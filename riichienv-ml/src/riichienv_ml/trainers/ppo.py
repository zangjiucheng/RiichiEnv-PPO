"""Online PPO Trainer with Ray distributed workers.

Supports two modes (controlled by cfg.async_rollout):

Synchronous (async_rollout=False, default):
  dispatch -> ray.get -> update -> sync -> dispatch -> ...

Double-buffered (async_rollout=True):
  dispatch_0 -> ray.get_0 -> dispatch_1 -> update_0 -> sync -> ray.get_1 -> ...
  Collection N+1 overlaps with PPO update N (1-step stale weights).
  Eval steps fall back to synchronous to ensure fresh weights.
"""
import sys
import os
import threading
import time

import ray
import wandb
import numpy as np
import torch
from loguru import logger

from riichienv_ml.trainers._ppo_worker import PPOWorker
from riichienv_ml.trainers._ppo_learner import PPOLearner


def _create_evaluator(cfg, model_config):
    """Create third-party evaluator if configured. Returns None otherwise."""
    ev = cfg.evaluator
    if ev.evaluator_name is not None:
        # Plugin-based evaluator (any player count)
        from riichienv_ml.evaluator import load_evaluator

        return load_evaluator(
            evaluator_name=ev.evaluator_name,
            model_path=ev.model_path,
            model_class=cfg.model_class,
            model_config=model_config,
            encoder_class=cfg.encoder_class,
            tile_dim=cfg.game.tile_dim,
            n_players=cfg.game.n_players,
            device=cfg.device,
            eval_device=ev.eval_device,
            opponents=[o.model_dump() for o in ev.opponents],
        )

    # Legacy: mortal evaluator (4P only, requires model_path)
    if ev.model_path is None or cfg.game.n_players != 4:
        return None

    from riichienv_ml.evaluator import load_evaluator

    return load_evaluator(
        evaluator_name="mortal",
        model_path=ev.model_path,
        model_class=cfg.model_class,
        model_config=model_config,
        encoder_class=cfg.encoder_class,
        tile_dim=cfg.game.tile_dim,
        device=cfg.device,
        eval_device=ev.eval_device,
    )


def evaluate_parallel(workers, hero_weights, baseline_weights, num_episodes):
    hero_ref = ray.put(hero_weights)
    baseline_ref = ray.put(baseline_weights)
    ray.get([w.update_weights.remote(hero_ref) for w in workers])
    ray.get([w.update_baseline_weights.remote(baseline_ref) for w in workers])

    eval_futures = [w.evaluate_episodes.remote() for w in workers]
    eval_results = ray.get(eval_futures)

    all_rewards = []
    all_ranks = []
    for worker_results in eval_results:
        for reward, rank in worker_results:
            all_rewards.append(reward)
            all_ranks.append(rank)
            if len(all_rewards) >= num_episodes:
                break
        if len(all_rewards) >= num_episodes:
            break

    all_rewards = all_rewards[:num_episodes]
    all_ranks = all_ranks[:num_episodes]

    mean_reward = float(np.mean(all_rewards))
    mean_rank = float(np.mean(all_ranks))
    rank_se = float(np.std(all_ranks) / np.sqrt(len(all_ranks)))

    return mean_reward, mean_rank, rank_se


def run_ppo_training(cfg):
    """Main online PPO training loop."""
    python_path = ":".join(sys.path)
    src_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

    worker_env_vars = {
        "PYTHONPATH": python_path,
        "PATH": os.environ["PATH"],
    }
    # Ray actors don't reliably inherit the driver's arbitrary OS env when a
    # runtime_env is specified, so explicitly forward the RIICHIENV_*/
    # TORCHINDUCTOR_* toggles that PPOWorker.__init__ reads (e.g.
    # RIICHIENV_DISABLE_WORKER_COMPILE, which gates the torch.compile that
    # deadlocks GPU workers at scale). Without this the flags are dead code
    # in the worker process.
    for _k, _v in os.environ.items():
        if _k.startswith(("RIICHIENV_", "TORCHINDUCTOR_", "TORCH_")):
            worker_env_vars.setdefault(_k, _v)

    runtime_env = {
        "working_dir": src_dir,
        "excludes": [".git", ".venv", "wandb", "__pycache__", "pyproject.toml"],
        "env_vars": worker_env_vars,
    }

    ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    game = cfg.game
    model_config = cfg.model.model_dump()

    learner = PPOLearner(
        device=cfg.device,
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ppo_clip=cfg.ppo_clip,
        ppo_epochs=cfg.ppo_epochs,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        alpha_kl=cfg.alpha_kl,
        alpha_kl_warmup_steps=cfg.alpha_kl_warmup_steps,
        batch_size=cfg.batch_size,
        model_config=model_config,
        model_class=cfg.model_class,
        teacher_model=cfg.teacher_model,
    )

    if cfg.load_model:
        learner.load_weights(cfg.load_model)
    if cfg.alpha_kl > 0 and cfg.teacher_model is not None:
        logger.info(f"KL teacher checkpoint: {cfg.teacher_model}")
    elif cfg.alpha_kl > 0 and cfg.load_model is not None:
        logger.info(f"KL teacher defaults to loaded checkpoint: {cfg.load_model}")
    elif cfg.teacher_model is not None and cfg.alpha_kl <= 0:
        logger.warning("teacher_model is set but alpha_kl <= 0, so teacher KL loss is disabled.")

    tp_evaluator = _create_evaluator(cfg, model_config)

    worker_kwargs = dict(
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        num_envs=cfg.num_envs_per_worker,
        model_config=model_config,
        model_class=cfg.model_class,
        encoder_class=cfg.encoder_class,
        encoder_config=cfg.encoder,
        grp_model=cfg.grp_model,
        pts_weight=cfg.pts_weight,
        n_players=game.n_players,
        game_mode=game.game_mode,
        tile_dim=game.tile_dim,
        starting_scores=game.starting_scores,
    )
    if cfg.worker_device == "cuda":
        # Bring CUDA workers up ONE AT A TIME. Creating all num_workers actors
        # at once makes them race to initialize CUDA contexts on the single
        # shared GPU, which intermittently deadlocks (~85% of runs: GPU pinned
        # at 0% util, no worker memory ever allocated). Blocking on each
        # worker's ready() -- which completes only after its __init__ +
        # first .to("cuda") + a cuda sync -- before creating the next means
        # contexts are created serially, eliminating the race. Costs ~1-2 min
        # of extra startup, paid once.
        workers = []
        for i in range(cfg.num_workers):
            w = PPOWorker.options(num_gpus=cfg.gpu_per_worker).remote(
                i, "cuda", **worker_kwargs,
            )
            ray.get(w.ready.remote())
            workers.append(w)
        logger.info(f"Brought up {len(workers)} CUDA workers serially")
    else:
        workers = [
            PPOWorker.remote(i, "cpu", **worker_kwargs)
            for i in range(cfg.num_workers)
        ]

    hero_weights = {k: v.cpu() for k, v in learner.get_weights().items()}

    # Load baseline (opponent) weights: either from a separate model or a copy of the hero
    if cfg.baseline_model:
        logger.info(f"Loading fixed baseline opponent from {cfg.baseline_model}")
        _bl_state = torch.load(cfg.baseline_model, map_location="cpu", weights_only=True)
        # Unwrap wrapped checkpoints (e.g. {'state_dict': ...})
        if isinstance(_bl_state, dict) and "state_dict" in _bl_state:
            _bl_state = _bl_state["state_dict"]
        elif isinstance(_bl_state, dict) and "model_state_dict" in _bl_state:
            _bl_state = _bl_state["model_state_dict"]
        # Auto-convert QNetwork keys to ActorCriticNetwork format
        _has_a = any(k.startswith("a_head.") for k in _bl_state)
        _has_v = any(k.startswith("v_head.") for k in _bl_state)
        _has_head = any(k.startswith("head.") for k in _bl_state)
        if _has_head and not _has_a:
            # Single-head QNetwork → ActorCriticNetwork
            _new = {}
            for k, v in _bl_state.items():
                if k.startswith("head."):
                    _new[k.replace("head.", "actor_head.")] = v
                elif k.startswith("aux_head."):
                    continue
                else:
                    _new[k] = v
            _bl_state = _new
        elif _has_a and _has_v:
            # Dueling QNetwork → ActorCriticNetwork
            _new = {}
            for k, v in _bl_state.items():
                if k.startswith("a_head."):
                    _new[k.replace("a_head.", "actor_head.")] = v
                elif k.startswith("v_head."):
                    _new[k.replace("v_head.", "critic_head.")] = v
                elif k.startswith("aux_head."):
                    continue
                else:
                    _new[k] = v
            _bl_state = _new
        baseline_weights = _bl_state
        logger.info(f"Loaded baseline model ({len(baseline_weights)} keys)")
    else:
        baseline_weights = {k: v.cpu() for k, v in learner.get_weights().items()}

    hero_ref = ray.put(hero_weights)
    baseline_ref = ray.put(baseline_weights)

    for w in workers:
        w.update_weights.remote(hero_ref)
        w.update_baseline_weights.remote(baseline_ref)

    step = 0
    episodes = 0
    # wandb is initialized by init_wandb() in the script

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    logger.info(f"Running dry-run evaluation ({cfg.eval_episodes} episodes, parallel)...")
    try:
        eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
            workers, hero_weights, baseline_weights,
            num_episodes=cfg.eval_episodes)
        logger.info(f"Dry-run evaluation passed: Reward={eval_reward:.2f}, "
                     f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
    except Exception as e:
        logger.error(f"Dry-run evaluation failed: {e}")
        raise e

    total_envs = cfg.num_workers * cfg.num_envs_per_worker

    prefetched_futures = None
    if cfg.async_rollout:
        logger.info("Async rollout enabled: using double buffering")
        prefetched_futures = [w.collect_episodes.remote() for w in workers]

    try:
        while step < cfg.num_steps:
            t_collect_start = time.time()
            if prefetched_futures is not None:
                all_results = ray.get(prefetched_futures)
                prefetched_futures = None
            else:
                futures = [w.collect_episodes.remote() for w in workers]
                all_results = ray.get(futures)
            t_collect = time.time() - t_collect_start

            batch_parts = []
            worker_stats_list = []
            for transitions, stats in all_results:
                if transitions:
                    batch_parts.append(transitions)
                if stats:
                    worker_stats_list.append(stats)
            episodes += total_envs

            if not batch_parts:
                if cfg.async_rollout:
                    prefetched_futures = [w.collect_episodes.remote() for w in workers]
                continue

            t_prep_start = time.time()
            rollout_batch = {
                "features": torch.from_numpy(np.concatenate([b["features"] for b in batch_parts])),
                "masks": torch.from_numpy(np.concatenate([b["mask"] for b in batch_parts])),
                "actions": torch.from_numpy(np.concatenate([b["action"] for b in batch_parts])),
                "old_log_probs": torch.from_numpy(np.concatenate([b["log_prob"] for b in batch_parts])),
                "advantages": torch.from_numpy(np.concatenate([b["advantage"] for b in batch_parts])),
                "returns": torch.from_numpy(np.concatenate([b["return"] for b in batch_parts])),
            }
            n_trans = len(rollout_batch["actions"])
            t_prep = time.time() - t_prep_start

            is_eval_step = (step + 1) > 0 and (step + 1) % cfg.eval_interval == 0
            if cfg.async_rollout and not is_eval_step and (step + 1) < cfg.num_steps:
                prefetched_futures = [w.collect_episodes.remote() for w in workers]

            t_update_start = time.time()
            metrics = learner.update(rollout_batch)
            t_update = time.time() - t_update_start
            step += 1
            metrics["time/collect_s"] = t_collect
            metrics["time/prep_s"] = t_prep
            metrics["time/update_s"] = t_update

            if worker_stats_list:
                for key in worker_stats_list[0]:
                    vals = [s[key] for s in worker_stats_list if key in s]
                    metrics[f"rollout/{key}"] = float(np.mean(vals))

            log_msg = (
                f"Step {step}: loss={metrics['loss']:.4f}, "
                f"pi={metrics['policy_loss']:.4f}, v={metrics['value_loss']:.4f}, "
                f"ent={metrics['entropy']:.4f}, kl={metrics['approx_kl']:.4f}, "
                f"kl_ref={metrics.get('kl_ref', 0):.4f}, "
                f"clip={metrics['clip_frac']:.3f}, "
                f"adv={metrics.get('adv/raw_mean', 0):.3f}\u00b1{metrics.get('adv/raw_std', 0):.3f}, "
                f"ret={metrics.get('return/mean', 0):.3f}, "
                f"ev={metrics.get('explained_variance', 0):.3f}, "
                f"ep={episodes}, trans={n_trans}"
            )
            if worker_stats_list:
                log_msg += (
                    f", rew={metrics.get('rollout/reward_mean', 0):.2f}"
                    f", rank={metrics.get('rollout/rank_mean', 0):.2f}"
                )
                if "rollout/kyoku_reward_mean" in metrics:
                    log_msg += (
                        f", k_rew={metrics['rollout/kyoku_reward_mean']:.3f}"
                        f"\u00b1{metrics.get('rollout/kyoku_reward_std', 0):.3f}"
                        f", k_len={metrics.get('rollout/kyoku_length_mean', 0):.1f}"
                    )
            log_msg += (
                f", t_collect={t_collect:.1f}s, t_prep={t_prep:.1f}s, t_update={t_update:.1f}s"
                f" (data={metrics.get('time/data_transfer_s', 0):.1f}s,"
                f" teacher_fwd={metrics.get('time/teacher_fwd_s', 0):.1f}s,"
                f" main_fwd_bwd={metrics.get('time/main_fwd_bwd_s', 0):.1f}s,"
                f" n_batches={metrics.get('time/n_batches', 0):.0f})"
            )
            logger.info(log_msg)
            wandb.log(metrics, step=step)

            if cfg.checkpoint_interval > 0 and step % cfg.checkpoint_interval == 0:
                save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
                save_weights = {k: v.cpu().clone() for k, v in learner.get_weights().items()}
                threading.Thread(target=torch.save, args=(save_weights, save_path), daemon=True).start()
                logger.info(f"Step {step}: saved checkpoint to {save_path} (async)")

            if step > 0 and step % cfg.eval_interval == 0:
                logger.info(f"Step {step}: Evaluating...")
                sys.stdout.flush()

                try:
                    hw = {k: v.cpu() for k, v in learner.get_weights().items()}
                    eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
                        workers, hw, baseline_weights,
                        num_episodes=cfg.eval_episodes)
                    logger.info(f"Evaluation (vs Baseline): Reward={eval_reward:.2f}, "
                                f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
                    wandb.log({
                        "eval/reward": eval_reward,
                        "eval/rank": eval_rank,
                        "eval/rank_se": eval_rank_se,
                    }, step=step)
                except Exception as e:
                    logger.error(f"Evaluation failed at step {step}: {e}")

                if tp_evaluator is not None:
                    try:
                        hw = {k: v.cpu() for k, v in learner.get_weights().items()}
                        tp_metrics = tp_evaluator.evaluate(
                            hw, num_episodes=cfg.evaluator.eval_episodes)
                        logline = tp_evaluator.metrics_to_logline(tp_metrics)
                        logger.info(f"TP Eval @ step {step}: {logline}")
                        wandb.log(tp_metrics, step=step)
                    except Exception as e:
                        logger.error(f"TP evaluation failed at step {step}: {e}")

            t_sync_start = time.time()
            weights = {k: v.cpu() for k, v in learner.get_weights().items()}

            has_nan = False
            for name, param in weights.items():
                if torch.isnan(param).any():
                    logger.error(f"NaN detected in learner weights {name} at step {step}")
                    has_nan = True
            if has_nan:
                logger.critical("Cannot sync NaN weights to workers. Stopping training.")
                break

            weight_ref = ray.put(weights)
            for w in workers:
                w.update_weights.remote(weight_ref)
            t_sync = time.time() - t_sync_start
            logger.info(f"Step {step}: t_sync={t_sync:.1f}s")

            # Periodic baseline update for self-play curriculum
            # (skip when using a fixed external baseline_model)
            if (cfg.baseline_model is None
                    and cfg.baseline_update_interval > 0
                    and step > 0
                    and step % cfg.baseline_update_interval == 0):
                baseline_weights = {k: v.clone() for k, v in weights.items()}
                baseline_ref = ray.put(baseline_weights)
                ray.get([w.update_baseline_weights.remote(baseline_ref) for w in workers])
                logger.info(f"Updated baseline weights at step {step}")

            if cfg.async_rollout and prefetched_futures is None and step < cfg.num_steps:
                prefetched_futures = [w.collect_episodes.remote() for w in workers]

    except KeyboardInterrupt:
        logger.info("Stopping...")

    logger.info(f"Final Step {step} (episodes={episodes}): Saving snapshot and Evaluating...")
    sys.stdout.flush()

    save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
    torch.save(learner.get_weights(), save_path)
    logger.info(f"Saved snapshot to {save_path}")

    try:
        hw = {k: v.cpu() for k, v in learner.get_weights().items()}
        eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
            workers, hw, baseline_weights,
            num_episodes=cfg.eval_episodes)
        logger.info(f"Final Evaluation (vs Baseline): Reward={eval_reward:.2f}, "
                    f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
        wandb.log({
            "eval/reward": eval_reward,
            "eval/rank": eval_rank,
            "eval/rank_se": eval_rank_se,
        }, step=step)
    except Exception as e:
        logger.error(f"Final Evaluation failed: {e}")

    if tp_evaluator is not None:
        try:
            hw = {k: v.cpu() for k, v in learner.get_weights().items()}
            tp_metrics = tp_evaluator.evaluate(
                hw, num_episodes=cfg.evaluator.eval_episodes)
            logline = tp_evaluator.metrics_to_logline(tp_metrics)
            logger.info(f"Final TP Eval: {logline}")
            wandb.log(tp_metrics, step=step)
        except Exception as e:
            logger.error(f"Final TP evaluation failed: {e}")

    ray.shutdown()
