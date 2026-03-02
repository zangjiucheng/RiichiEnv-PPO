"""Online BC trainer: teacher model plays games via Ray workers, student clones actions.

The student (ActorCriticNetwork) lives only in the main process (GPU).
Workers are pure data generators — no student model, no weight sync.
"""
import math
import os
import sys
import time
from typing import Literal

import numpy as np
import ray
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from loguru import logger

import wandb

from riichienv_ml.config import import_class
from riichienv_ml.teacher import load_teacher_worker_class
from riichienv_ml.trainers.bc_logs import _create_evaluator


class BCModelTrainer:
    def __init__(
        self,
        grp_model_path: str,
        pts_weight: list,
        device_str: str = "cuda",
        gamma: float = 1.0,
        batch_size: int = 128,
        lr: float = 5e-4,
        lr_min: float = 1e-5,
        weight_decay: float = 0.1,
        value_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        model_config: dict | None = None,
        model_class: str = "riichienv_ml.models.actor_critic.ActorCriticNetwork",
        encoder_class: str = "riichienv_ml.features.feat_v1.ObservationEncoder",
        n_players: int = 4,
        tile_dim: int = 34,
        game_mode: str = "4p-red-half",
        evaluator_config=None,

        # Online teacher settings (extensions)
        teacher_model_name: Literal["kanachan", "mortal"] = "kanachan",
        teacher_model_path: str | None = None,
        num_ray_workers: int = 4,
        num_envs_per_worker: int = 16,
        num_steps: int = 500,
        train_epochs: int = 3,
        warmup_steps: int = 0,
        worker_device: str = "cpu",
        gpu_per_worker: float = 0.0,
    ):
        self.grp_model_path = grp_model_path
        self.pts_weight = pts_weight
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.model_config = model_config or {}
        self.model_class = model_class
        self.encoder_class = encoder_class
        self.n_players = n_players
        self.tile_dim = tile_dim
        self.game_mode = game_mode

        # Online teacher settings
        self.teacher_model_name = teacher_model_name
        self.teacher_model_path = teacher_model_path
        self.num_ray_workers = num_ray_workers
        self.num_envs_per_worker = num_envs_per_worker
        self.num_steps = num_steps
        self.train_epochs = train_epochs
        self.warmup_steps = warmup_steps
        self.worker_device = worker_device
        self.gpu_per_worker = gpu_per_worker

        if evaluator_config is None:
            from riichienv_ml.config import EvaluatorConfig
            evaluator_config = EvaluatorConfig()
        self.evaluator_config = evaluator_config

        self.tp_evaluator = _create_evaluator(
            cfg_kwargs=dict(
                model_path=evaluator_config.model_path,
                eval_device=evaluator_config.eval_device,
                model_class=model_class,
                encoder_class=encoder_class,
                tile_dim=tile_dim,
                device_str=device_str,
                n_players=n_players,
            ),
            model_config=self.model_config,
        )

    def _create_cosine_scheduler(self, optimizer, total_steps, warmup_steps=0):
        """Linear warmup + one-way cosine decay from lr to lr_min."""
        lr_min_ratio = self.lr_min / self.lr

        def lr_lambda(step):
            # Phase 1: linear warmup  0 → lr
            if step < warmup_steps:
                return step / warmup_steps
            # Phase 2: cosine decay  lr → lr_min
            decay_steps = total_steps - warmup_steps
            progress = (step - warmup_steps) / max(1, decay_steps)
            if progress >= 1.0:
                return lr_min_ratio
            return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _combine_transitions(self, results):
        """Combine transitions from all workers into a single batch."""
        all_features = []
        all_masks = []
        all_actions = []
        all_targets = []

        all_stats = []

        for transitions, stats in results:
            if transitions:
                all_features.append(transitions["features"])
                all_masks.append(transitions["mask"])
                all_actions.append(transitions["action"])
                all_targets.append(transitions["target"])
            all_stats.append(stats)

        if not all_features:
            return None, all_stats

        batch = {
            "features": np.concatenate(all_features),
            "mask": np.concatenate(all_masks),
            "action": np.concatenate(all_actions),
            "target": np.concatenate(all_targets),
        }
        return batch, all_stats

    def _run_eval(self, model, output_path, step):
        """Run periodic evaluation."""
        if self.tp_evaluator is None:
            return
        try:
            ckpt_path = output_path.replace(".pth", f"_step{step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

            hw = {k: v.cpu() for k, v in model.state_dict().items()}
            model.eval()
            metrics_ = self.tp_evaluator.evaluate(
                hw, num_episodes=self.evaluator_config.eval_episodes)
            model.train()
            logline = self.tp_evaluator.metrics_to_logline(metrics_)
            logger.info(f"Eval @ step {step}: {logline}")
            wandb.log(metrics_, step=step)
        except Exception as e:
            logger.error(f"Mortal evaluation failed at step {step}: {e}")

    def train(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        python_path = ":".join(sys.path)
        src_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        runtime_env = {
            "working_dir": src_dir,
            "excludes": [".git", ".venv", "wandb", "__pycache__",
                         "pyproject.toml"],
            "env_vars": {
                "PYTHONPATH": python_path,
                "PATH": os.environ["PATH"],
            },
        }
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

        try:
            self._train_loop(output_path)
        finally:
            ray.shutdown()

    def _train_loop(self, output_path: str) -> None:
        # 1. Discover teacher worker class via entry_points
        WorkerClass = load_teacher_worker_class()

        # 2. Launch Ray workers (with optional GPU)
        worker_options = {}
        if self.gpu_per_worker > 0:
            worker_options["num_gpus"] = self.gpu_per_worker
        RemoteWorker = WorkerClass.options(**worker_options) if worker_options else WorkerClass

        workers = [
            RemoteWorker.remote(
                worker_id=i,
                teacher_model_path=self.teacher_model_path,
                teacher_model_name=self.teacher_model_name,
                encoder_class=self.encoder_class,
                tile_dim=self.tile_dim,
                n_players=self.n_players,
                game_mode=self.game_mode,
                grp_model=self.grp_model_path,
                pts_weight=self.pts_weight,
                gamma=self.gamma,
                num_envs=self.num_envs_per_worker,
                device=self.worker_device,
            )
            for i in range(self.num_ray_workers)
        ]
        logger.info(f"Launched {self.num_ray_workers} teacher workers "
                     f"({self.num_envs_per_worker} envs each, "
                     f"device={self.worker_device}, "
                     f"gpu_per_worker={self.gpu_per_worker})")

        # 3. Student model + optimizer (main process, GPU)
        ModelClass = import_class(self.model_class)
        model = ModelClass(**self.model_config).to(self.device)
        optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Scheduler will be created after first batch (need actual transition count)
        scheduler = None
        model.train()

        global_step = 0
        total_transitions = 0

        # 4. Async double-buffered training loop
        # Pre-dispatch first collection
        t0 = time.time()
        pending_futures = [w.collect_episodes.remote() for w in workers]

        for step in range(self.num_steps):
            # Wait for current batch
            results = ray.get(pending_futures)
            collect_time = time.time() - t0

            batch, worker_stats = self._combine_transitions(results)

            # Dispatch next collection immediately (overlaps with training)
            t0 = time.time()
            if step < self.num_steps - 1:
                pending_futures = [w.collect_episodes.remote() for w in workers]

            if batch is None:
                logger.warning(f"Step {step}: no transitions collected, skipping")
                continue

            N = len(batch["action"])

            # Create scheduler after first batch using actual transition count
            if scheduler is None:
                mb_per_step = (N + self.batch_size - 1) // self.batch_size
                total_opt_steps = mb_per_step * self.train_epochs * self.num_steps
                warmup_opt_steps = mb_per_step * self.train_epochs * self.warmup_steps
                scheduler = self._create_cosine_scheduler(
                    optimizer, total_opt_steps, warmup_opt_steps)
                warmup_msg = f", warmup {warmup_opt_steps} steps" if warmup_opt_steps else ""
                logger.info(
                    f"LR scheduler: cosine {self.lr} -> {self.lr_min} "
                    f"over {total_opt_steps} steps "
                    f"({mb_per_step} mb/step x {self.train_epochs} epochs "
                    f"x {self.num_steps} steps{warmup_msg})")
            total_transitions += N

            # Aggregate worker stats
            agg_stats = {}
            valid_stats = [s for s in worker_stats if s]
            if valid_stats:
                for key in valid_stats[0]:
                    vals = [s[key] for s in valid_stats if key in s]
                    agg_stats[key] = float(np.mean(vals))

            # Move to GPU
            features = torch.from_numpy(batch["features"]).to(self.device)
            masks = torch.from_numpy(batch["mask"]).float().to(self.device)
            actions = torch.from_numpy(batch["action"]).long().to(self.device)
            targets = torch.from_numpy(batch["target"]).float().to(self.device)

            # Mini-batch BC update (multiple epochs over collected batch)
            t1 = time.time()
            model.train()
            step_policy_loss = 0.0
            step_value_loss = 0.0
            step_loss = 0.0
            num_minibatches = 0

            for epoch in range(self.train_epochs):
                indices = torch.randperm(N, device=self.device)
                for start in range(0, N, self.batch_size):
                    idx = indices[start:start + self.batch_size]

                    logits, values = model(features[idx])
                    logits = logits.masked_fill(~masks[idx].bool(), -1e9)

                    policy_loss = F.cross_entropy(logits, actions[idx])
                    value_loss = F.mse_loss(values, targets[idx])
                    loss = policy_loss + self.value_coef * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    step_policy_loss += policy_loss.item()
                    step_value_loss += value_loss.item()
                    step_loss += loss.item()
                    num_minibatches += 1
                    global_step += 1

            train_time = time.time() - t1

            # Log metrics
            if num_minibatches > 0:
                avg_policy = step_policy_loss / num_minibatches
                avg_value = step_value_loss / num_minibatches
                avg_loss = step_loss / num_minibatches

                log_dict = {
                    "step": step,
                    "loss": avg_loss,
                    "policy_loss": avg_policy,
                    "value_loss": avg_value,
                    "transitions": N,
                    "total_transitions": total_transitions,
                    "collect_time": collect_time,
                    "train_time": train_time,
                    "lr": scheduler.get_last_lr()[0],
                }
                log_dict.update({f"worker/{k}": v for k, v in agg_stats.items()})
                wandb.log(log_dict, step=global_step)

                logger.info(
                    f"Step {step}/{self.num_steps}: "
                    f"loss={avg_loss:.4f} (policy={avg_policy:.4f}, "
                    f"value={avg_value:.4f}), "
                    f"{N} trans, {num_minibatches} mb, "
                    f"collect={collect_time:.1f}s, train={train_time:.1f}s")

            # Periodic eval
            eval_interval = self.evaluator_config.eval_interval
            if (self.tp_evaluator is not None
                    and step > 0
                    and step % eval_interval == 0):
                self._run_eval(model, output_path, global_step)

            # Periodic save
            if step > 0 and step % 50 == 0:
                torch.save(model.state_dict(), output_path)
                logger.info(f"Saved model to {output_path}")

        # Final save
        torch.save(model.state_dict(), output_path)
        logger.info(f"Saved final model to {output_path}")

        # Final evaluation
        if self.tp_evaluator is not None:
            self._run_eval(model, output_path, global_step)

        wandb.finish()
