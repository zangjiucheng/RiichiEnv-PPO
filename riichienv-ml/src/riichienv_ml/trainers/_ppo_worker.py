import random
import time
from pathlib import Path

import ray
import torch
import numpy as np
from riichienv import RiichiEnv

from riichienv_ml.config import import_class, GAME_PARAMS
from riichienv_ml.models.grp_model import RewardPredictor


@ray.remote
class PPOWorker:
    """PPO rollout worker. Parameterized by n_players, game_mode, tile_dim for 3P/4P support."""

    def __init__(self, worker_id: int, device: str = "cpu",
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_envs: int = 16,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.actor_critic.ActorCriticNetwork",
                 encoder_class: str = "riichienv_ml.features.feat_v1.ObservationEncoder",
                 grp_model: str | None = None,
                 pts_weight: list[float] | None = None,
                 n_players: int = 4,
                 game_mode: str = "4p-red-half",
                 tile_dim: int = 34,
                 starting_scores: list[int] | None = None):
        torch.set_num_threads(1)
        self.worker_id = worker_id
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.n_players = n_players
        self.game_mode = game_mode
        self.tile_dim = tile_dim
        self.starting_scores = starting_scores or GAME_PARAMS[n_players]["starting_scores"]
        self.envs = [RiichiEnv(game_mode=game_mode) for _ in range(num_envs)]
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        mc = model_config or {}
        ModelClass = import_class(model_class)

        self.model = ModelClass(**mc).to(self.device)
        self.model.eval()

        self.baseline_model = ModelClass(**mc).to(self.device)
        self.baseline_model.eval()

        if self.device.type == "cuda":
            self.model = torch.compile(self.model)
            self.baseline_model = torch.compile(self.baseline_model)

        EncoderClass = import_class(encoder_class)
        self.encoder = EncoderClass(tile_dim=tile_dim)
        self._compiled_warmup = False

        pw = pts_weight or GAME_PARAMS[n_players].get("pts_weight", [10.0, 4.0, -4.0, -10.0])
        if grp_model:
            grp_model_path = Path(grp_model).expanduser()
            if not grp_model_path.is_absolute():
                grp_model_path = (Path.cwd() / grp_model_path).resolve()
            else:
                grp_model_path = grp_model_path.resolve()
            self.reward_predictor = RewardPredictor(
                str(grp_model_path), pw, n_players=n_players, device="cpu")
        else:
            self.reward_predictor = None

    def _apply_state_dict(self, model, state_dict):
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        for name, param in target.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
        for name, buf in target.named_buffers():
            if name in state_dict:
                buf.data.copy_(state_dict[name])

    def update_weights(self, state_dict):
        self._apply_state_dict(self.model, state_dict)

    def update_baseline_weights(self, state_dict):
        self._apply_state_dict(self.baseline_model, state_dict)

    def _warmup_compile(self):
        if self._compiled_warmup:
            return
        target = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if hasattr(target, "backbone") and hasattr(target.backbone, "conv_in"):
            # CNN model (ActorCriticNetwork / QNetwork)
            in_ch = target.backbone.conv_in.in_channels
            dummy = torch.randn(1, in_ch, self.tile_dim, device=self.device)
        elif hasattr(self.encoder, "PACKED_SIZE"):
            # Packed sequence encoder (TransformerActorCritic etc.)
            dummy = torch.zeros(1, self.encoder.PACKED_SIZE, device=self.device)
        else:
            self._compiled_warmup = True
            return
        with torch.no_grad():
            self.model(dummy)
            self.baseline_model(dummy)
        self._compiled_warmup = True

    def _compute_kyoku_rewards(self, prev_scores, cur_scores,
                               round_wind, oya, honba, riichi_sticks) -> list[float]:
        if self.reward_predictor is None:
            return [0.0] * self.n_players

        n = self.n_players
        deltas = [cur_scores[i] - prev_scores[i] for i in range(n)]
        grp_features = {}
        for i in range(n):
            grp_features[f"p{i}_init_score"] = prev_scores[i]
            grp_features[f"p{i}_end_score"] = cur_scores[i]
            grp_features[f"p{i}_delta_score"] = deltas[i]
        grp_features["chang"] = round_wind
        grp_features["ju"] = oya
        grp_features["ben"] = honba
        grp_features["liqibang"] = riichi_sticks
        return self.reward_predictor.calc_all_player_rewards(grp_features)

    def collect_episodes(self):
        t_start = time.time()

        if not self._compiled_warmup:
            self._warmup_compile()

        n = self.n_players
        obs_dicts = [
            env.reset(scores=list(self.starting_scores),
                      round_wind=0, oya=0, honba=0, kyotaku=0)
            for env in self.envs
        ]

        active = [True] * self.num_envs
        hero_pids = [random.randint(0, n - 1) for _ in range(self.num_envs)]

        kyoku_buffers = [[] for _ in range(self.num_envs)]
        completed_kyokus = [[] for _ in range(self.num_envs)]

        prev_kyoku_idx = [env.kyoku_idx for env in self.envs]
        kyoku_start_scores = [list(env.scores()) for env in self.envs]
        kyoku_start_meta = [(env.round_wind, env.oya, env.honba, env.riichi_sticks)
                            for env in self.envs]
        kyoku_count = [0] * self.num_envs

        while any(active):
            hero_items = []
            opp_items = []

            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                for pid, obs in obs_dicts[ei].items():
                    la = obs.legal_actions()
                    if la:
                        if pid == hero_pids[ei]:
                            hero_items.append((ei, obs, la))
                        else:
                            opp_items.append((ei, pid, obs, la))

            if not hero_items and not opp_items:
                for ei in range(self.num_envs):
                    if active[ei] and self.envs[ei].done():
                        active[ei] = False
                break

            env_steps = {ei: {} for ei in range(self.num_envs)}

            if hero_items:
                feat_list = []
                mask_list = []
                for _, obs, _ in hero_items:
                    feat_list.append(self.encoder.encode(obs))
                    mask_list.append(torch.from_numpy(
                        np.frombuffer(obs.mask(), dtype=np.uint8).copy()))

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    logits, values = self.model(feat_batch)
                    mask_bool = mask_batch.bool()
                    logits = logits.masked_fill(~mask_bool, -1e9)

                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)
                    log_probs_all = torch.log_softmax(logits, dim=-1)
                    log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

                actions_cpu = actions.cpu().numpy()
                log_probs_cpu = log_probs.cpu().numpy()
                values_cpu = values.cpu().numpy()
                feat_cpu = feat_batch.cpu().numpy()
                mask_cpu = mask_batch.cpu().numpy()

                for idx, (ei, obs, la) in enumerate(hero_items):
                    action_idx = int(actions_cpu[idx])
                    kyoku_buffers[ei].append({
                        "features": feat_cpu[idx],
                        "mask": mask_cpu[idx],
                        "action": action_idx,
                        "log_prob": float(log_probs_cpu[idx]),
                        "value": float(values_cpu[idx]),
                    })
                    found_action = obs.find_action(action_idx)
                    if found_action is None:
                        found_action = la[0]
                    env_steps[ei][hero_pids[ei]] = found_action

            if opp_items:
                feat_list = []
                mask_list = []
                for _, _, obs, _ in opp_items:
                    feat_list.append(self.encoder.encode(obs))
                    mask_list.append(torch.from_numpy(
                        np.frombuffer(obs.mask(), dtype=np.uint8).copy()))

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    output = self.baseline_model(feat_batch)
                    if isinstance(output, tuple):
                        opp_logits = output[0]
                    else:
                        opp_logits = output
                    opp_logits = opp_logits.masked_fill(~mask_batch.bool(), -1e9)
                    opp_actions = opp_logits.argmax(dim=1)

                opp_actions_cpu = opp_actions.cpu().numpy()
                for idx, (ei, pid, obs, la) in enumerate(opp_items):
                    action_idx = int(opp_actions_cpu[idx])
                    found_action = obs.find_action(action_idx)
                    if found_action is None:
                        found_action = la[0]
                    env_steps[ei][pid] = found_action

            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                if env_steps[ei]:
                    obs_dicts[ei] = self.envs[ei].step(env_steps[ei])

                env = self.envs[ei]

                cur_kyoku_idx = env.kyoku_idx
                if cur_kyoku_idx != prev_kyoku_idx[ei] and kyoku_buffers[ei]:
                    cur_scores = list(env.scores())
                    rw, oya, honba, rsticks = kyoku_start_meta[ei]
                    all_rewards = self._compute_kyoku_rewards(
                        kyoku_start_scores[ei], cur_scores, rw, oya, honba, rsticks)
                    reward = all_rewards[hero_pids[ei]]
                    completed_kyokus[ei].append((kyoku_buffers[ei], reward))
                    kyoku_buffers[ei] = []
                    kyoku_count[ei] += 1

                    prev_kyoku_idx[ei] = cur_kyoku_idx
                    kyoku_start_scores[ei] = cur_scores
                    kyoku_start_meta[ei] = (env.round_wind, env.oya,
                                            env.honba, env.riichi_sticks)

                if env.done():
                    if kyoku_buffers[ei]:
                        cur_scores = list(env.scores())
                        rw, oya, honba, rsticks = kyoku_start_meta[ei]
                        all_rewards = self._compute_kyoku_rewards(
                            kyoku_start_scores[ei], cur_scores, rw, oya, honba, rsticks)
                        reward = all_rewards[hero_pids[ei]]
                        completed_kyokus[ei].append((kyoku_buffers[ei], reward))
                        kyoku_buffers[ei] = []
                    active[ei] = False

        feat_list = []
        mask_list = []
        action_list = []
        log_prob_list = []
        advantage_list = []
        return_list = []
        episode_rewards = []
        episode_ranks = []
        kyoku_lengths = []
        kyoku_rewards = []
        value_predictions = []

        for ei in range(self.num_envs):
            ranks = self.envs[ei].ranks()
            rank = ranks[hero_pids[ei]]
            final_reward = 0.0
            if rank == 1: final_reward = 10.0
            elif rank == 2: final_reward = 4.0
            elif rank == 3: final_reward = -4.0
            elif rank == 4: final_reward = -10.0

            episode_rewards.append(final_reward)
            episode_ranks.append(rank)

            for traj, kyoku_reward in completed_kyokus[ei]:
                T = len(traj)
                if T == 0:
                    continue

                kyoku_lengths.append(T)
                kyoku_rewards.append(kyoku_reward)

                values = [step["value"] for step in traj]
                value_predictions.extend(values)
                advantages = [0.0] * T
                returns = [0.0] * T

                gae = 0.0
                for t in reversed(range(T)):
                    if t == T - 1:
                        reward = kyoku_reward
                        next_value = 0.0
                    else:
                        reward = 0.0
                        next_value = values[t + 1]

                    delta = reward + self.gamma * next_value - values[t]
                    gae = delta + self.gamma * self.gae_lambda * gae
                    advantages[t] = gae
                    returns[t] = gae + values[t]

                for t, step in enumerate(traj):
                    feat_list.append(step["features"])
                    mask_list.append(step["mask"])
                    action_list.append(step["action"])
                    log_prob_list.append(step["log_prob"])
                    advantage_list.append(advantages[t])
                    return_list.append(returns[t])

        n_transitions = len(feat_list)
        if n_transitions > 0:
            transitions = {
                "features": np.stack(feat_list),
                "mask": np.stack(mask_list),
                "action": np.array(action_list, dtype=np.int64),
                "log_prob": np.array(log_prob_list, dtype=np.float32),
                "advantage": np.array(advantage_list, dtype=np.float32),
                "return": np.array(return_list, dtype=np.float32),
            }
        else:
            transitions = {}

        stats = {}
        if episode_rewards:
            stats["reward_mean"] = float(np.mean(episode_rewards))
            stats["reward_std"] = float(np.std(episode_rewards))
            stats["rank_mean"] = float(np.mean(episode_ranks))
            stats["value_pred_mean"] = float(np.mean(value_predictions)) if value_predictions else 0.0
            stats["value_pred_std"] = float(np.std(value_predictions)) if value_predictions else 0.0
        if kyoku_lengths:
            stats["kyoku_length_mean"] = float(np.mean(kyoku_lengths))
            stats["kyoku_reward_mean"] = float(np.mean(kyoku_rewards))
            stats["kyoku_reward_std"] = float(np.std(kyoku_rewards))
            stats["kyokus_per_hanchan"] = float(len(kyoku_lengths) / self.num_envs)

        t_end = time.time()
        episode_time = t_end - t_start
        if n_transitions > 0:
            from loguru import logger
            logger.info(f"Worker {self.worker_id}: {self.num_envs} hanchan, "
                        f"{len(kyoku_lengths)} kyokus took {episode_time:.3f}s, "
                        f"{n_transitions} transitions, {n_transitions/episode_time:.1f} trans/s")

        return transitions, stats

    def evaluate_episodes(self):
        if not self._compiled_warmup:
            self._warmup_compile()

        n = self.n_players
        obs_dicts = [
            env.reset(scores=list(self.starting_scores),
                      round_wind=0, oya=0, honba=0, kyotaku=0)
            for env in self.envs
        ]
        active = [True] * self.num_envs

        while any(active):
            hero_items = []
            opp_items = []

            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                for pid, obs in obs_dicts[ei].items():
                    la = obs.legal_actions()
                    if la:
                        if pid == 0:
                            hero_items.append((ei, obs, la))
                        else:
                            opp_items.append((ei, pid, obs, la))

            if not hero_items and not opp_items:
                for ei in range(self.num_envs):
                    if active[ei] and self.envs[ei].done():
                        active[ei] = False
                break

            env_steps = {ei: {} for ei in range(self.num_envs)}

            if hero_items:
                feat_list = [self.encoder.encode(obs) for _, obs, _ in hero_items]
                mask_list = [torch.from_numpy(
                    np.frombuffer(obs.mask(), dtype=np.uint8).copy())
                    for _, obs, _ in hero_items]

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    logits, _ = self.model(feat_batch)
                    logits = logits.masked_fill(~mask_batch.bool(), -1e9)
                    actions = logits.argmax(dim=1)

                actions_cpu = actions.cpu().numpy()
                for idx, (ei, obs, la) in enumerate(hero_items):
                    action_idx = int(actions_cpu[idx])
                    found_action = obs.find_action(action_idx)
                    if found_action is None:
                        found_action = la[0]
                    env_steps[ei][0] = found_action

            if opp_items:
                feat_list = [self.encoder.encode(obs) for _, _, obs, _ in opp_items]
                mask_list = [torch.from_numpy(
                    np.frombuffer(obs.mask(), dtype=np.uint8).copy())
                    for _, _, obs, _ in opp_items]

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    output = self.baseline_model(feat_batch)
                    if isinstance(output, tuple):
                        opp_logits = output[0]
                    else:
                        opp_logits = output
                    opp_logits = opp_logits.masked_fill(~mask_batch.bool(), -1e9)
                    opp_actions = opp_logits.argmax(dim=1)

                opp_actions_cpu = opp_actions.cpu().numpy()
                for idx, (ei, pid, obs, la) in enumerate(opp_items):
                    action_idx = int(opp_actions_cpu[idx])
                    found_action = obs.find_action(action_idx)
                    if found_action is None:
                        found_action = la[0]
                    env_steps[ei][pid] = found_action

            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                if env_steps[ei]:
                    obs_dicts[ei] = self.envs[ei].step(env_steps[ei])
                if self.envs[ei].done():
                    active[ei] = False

        results = []
        for ei in range(self.num_envs):
            ranks = self.envs[ei].ranks()
            rank = ranks[0]
            reward = 0.0
            if rank == 1: reward = 10.0
            elif rank == 2: reward = 4.0
            elif rank == 3: reward = -4.0
            elif rank == 4: reward = -10.0
            results.append((reward, rank))
        return results
