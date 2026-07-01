import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from riichienv_ml.config import import_class


class PPOLearner:
    def __init__(self,
                 device: str = "cuda",
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ppo_clip: float = 0.2,
                 ppo_epochs: int = 4,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 weight_decay: float = 0.0,
                 alpha_kl: float = 0.0,
                 alpha_kl_warmup_steps: int = 0,
                 batch_size: int = 128,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.actor_critic.ActorCriticNetwork",
                 teacher_model: str | None = None):

        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.alpha_kl = alpha_kl
        self.alpha_kl_warmup_steps = alpha_kl_warmup_steps
        self.batch_size = batch_size
        self.teacher_model_path = teacher_model

        mc = model_config or {}
        ModelClass = import_class(model_class)
        self.model = ModelClass(**mc).to(self.device)

        self.ref_model = None
        if alpha_kl > 0:
            self.ref_model = ModelClass(**mc).to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
            if self.teacher_model_path:
                self.load_teacher_weights(self.teacher_model_path)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.total_steps = 0

    def get_weights(self):
        return self.model.state_dict()

    @staticmethod
    def _normalize_state_dict(state: dict) -> dict:
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint format: {type(state)}")
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {
                (k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v
                for k, v in state.items()
            }
        return state

    def _load_weights_compat(self, target_model: nn.Module, state: dict, path: str, role: str) -> tuple[list, list]:
        state = self._normalize_state_dict(state)

        has_actor = any(k.startswith("actor_head.") for k in state.keys())
        has_critic = any(k.startswith("critic_head.") for k in state.keys())
        has_v_head = any(k.startswith("v_head.") for k in state.keys())
        has_a_head = any(k.startswith("a_head.") for k in state.keys())

        if has_actor and has_critic:
            missing, unexpected = target_model.load_state_dict(state, strict=False)
            logger.info(f"Loaded ActorCriticNetwork {role} weights from {path}")
            return missing, unexpected

        if has_v_head and has_a_head:
            new_state = {}
            for k, v in state.items():
                if k.startswith("a_head."):
                    new_state[k.replace("a_head.", "actor_head.")] = v
                elif k.startswith("v_head."):
                    new_state[k.replace("v_head.", "critic_head.")] = v
                elif k.startswith("aux_head."):
                    continue
                else:
                    new_state[k] = v
            missing, unexpected = target_model.load_state_dict(new_state, strict=False)
            logger.info(
                f"Loaded dueling QNetwork {role} weights from {path} "
                f"(a_head -> actor_head, v_head -> critic_head)"
            )
            return missing, unexpected

        if any(k.startswith("head.") for k in state.keys()):
            new_state = {}
            for k, v in state.items():
                if k.startswith("head."):
                    new_state[k.replace("head.", "actor_head.")] = v
                else:
                    new_state[k] = v
            missing, unexpected = target_model.load_state_dict(new_state, strict=False)
            logger.info(
                f"Loaded QNetwork {role} weights from {path} "
                f"(head -> actor_head, critic head random init if missing)"
            )
            return missing, unexpected

        missing, unexpected = target_model.load_state_dict(state, strict=False)
        logger.info(f"Loaded {role} weights from {path} (best effort)")
        return missing, unexpected

    @staticmethod
    def _log_missing_unexpected(missing: list, unexpected: list, role: str) -> None:
        if missing:
            logger.warning(f"{role} missing keys: {missing}")
        if unexpected:
            logger.warning(f"{role} unexpected keys: {unexpected}")

    def load_teacher_weights(self, path: str) -> None:
        if self.ref_model is None:
            raise RuntimeError("Cannot load teacher weights when alpha_kl <= 0 (reference model not initialized).")
        state = torch.load(path, map_location=self.device)
        missing, unexpected = self._load_weights_compat(
            target_model=self.ref_model,
            state=state,
            path=path,
            role="teacher",
        )
        self._log_missing_unexpected(missing, unexpected, role="teacher")
        self.teacher_model_path = path
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        logger.info("Loaded frozen teacher model for KL regularization")

    def load_weights(self, path: str):
        """Load weights with backward compatibility for QNetwork checkpoints."""
        state = torch.load(path, map_location=self.device)
        missing, unexpected = self._load_weights_compat(
            target_model=self.model,
            state=state,
            path=path,
            role="student",
        )
        self._log_missing_unexpected(missing, unexpected, role="student")

        if self.ref_model is not None and self.teacher_model_path is None:
            self.ref_model.load_state_dict(self.model.state_dict())
            logger.info("Initialized frozen teacher from current student checkpoint")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def update(self, rollout_batch: dict) -> dict:
        """PPO update over a batch of on-policy trajectory data."""
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

        self._sync()
        t_data_start = time.time()
        features = rollout_batch["features"].to(self.device)
        masks = rollout_batch["masks"].to(self.device)
        actions = rollout_batch["actions"].long().to(self.device)
        old_log_probs = rollout_batch["old_log_probs"].to(self.device)
        advantages = rollout_batch["advantages"].to(self.device)
        returns = rollout_batch["returns"].to(self.device)
        self._sync()
        t_data_transfer = time.time() - t_data_start

        adv_raw_mean = advantages.mean().item()
        adv_raw_std = advantages.std().item()
        return_mean = returns.mean().item()
        return_std = returns.std().item()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Accumulate as GPU tensors and defer .item() to a single point at
        # the end of the loop, avoiding a host-device sync on every one of
        # the ~13 per-minibatch metrics. Measured: didn't move the needle on
        # its own (still ~9.5 min/step), so the update's cost is elsewhere --
        # t_data_transfer/t_teacher_fwd/t_main_fwd_bwd below (with explicit
        # torch.cuda.synchronize() calls, since CUDA ops are async and a bare
        # time.time() around them would just measure how fast the CPU can
        # enqueue work, not how long the GPU takes) are here to find out
        # where. Remove the sync calls once the bottleneck is identified --
        # they're deliberately reintroducing the per-batch sync cost we just
        # removed above, purely for this one-off profiling pass.
        sum_keys = ["policy_loss", "value_loss", "entropy", "loss", "approx_kl",
                    "clip_frac", "ratio/mean", "ratio/std", "value/predicted_mean",
                    "grad_norm", "kl_ref"]
        max_keys = ["ratio/max", "kl/max"]
        metrics_gpu = {k: torch.zeros((), device=self.device) for k in sum_keys}
        metrics_gpu.update({k: torch.zeros((), device=self.device) for k in max_keys})

        N = features.shape[0]
        last_epoch_values = torch.zeros(N, device=self.device)
        n_batches_run = 0
        t_teacher_fwd = 0.0
        t_main_fwd_bwd = 0.0

        for epoch in range(self.ppo_epochs):
            perm = torch.randperm(N, device=self.device)

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                idx = perm[start:end]

                batch_features = features[idx]
                batch_masks = masks[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                self._sync()
                t_batch_start = time.time()
                t_teacher_this_batch = 0.0

                logits, values = self.model(batch_features)

                mask_bool = batch_masks.bool()
                logits = logits.masked_fill(~mask_bool, -1e9)

                log_probs_all = torch.log_softmax(logits, dim=-1)
                log_probs = log_probs_all.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * log_probs_all).sum(dim=-1)
                entropy = entropy.mean()

                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)

                kl_ref_val = torch.zeros((), device=self.device)
                kl_ref_loss = 0.0
                effective_kl = self.alpha_kl
                if self.alpha_kl_warmup_steps > 0:
                    effective_kl = self.alpha_kl * min(1.0, self.total_steps / self.alpha_kl_warmup_steps)
                if self.ref_model is not None and effective_kl > 0:
                    self._sync()
                    t_teacher0 = time.time()
                    with torch.no_grad():
                        ref_logits, _ = self.ref_model(batch_features)
                        ref_logits = ref_logits.masked_fill(~mask_bool, -1e9)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    kl_per_action = probs * (log_probs_all - ref_log_probs)
                    kl_per_action = kl_per_action.masked_fill(~mask_bool, 0.0)
                    kl_ref_term = kl_per_action.sum(dim=-1).mean()
                    kl_ref_val = kl_ref_term.detach()
                    kl_ref_loss = effective_kl * kl_ref_term
                    self._sync()
                    t_teacher_this_batch = time.time() - t_teacher0
                    t_teacher_fwd += t_teacher_this_batch

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + kl_ref_loss

                if torch.isnan(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self._sync()
                t_main_fwd_bwd += (time.time() - t_batch_start) - t_teacher_this_batch

                if epoch == self.ppo_epochs - 1:
                    last_epoch_values[idx] = values.detach()

                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean()
                    kl_max = (batch_old_log_probs - log_probs).max()
                    clip_frac = ((ratio - 1.0).abs() > self.ppo_clip).float().mean()

                grad_norm_t = grad_norm if isinstance(grad_norm, torch.Tensor) else torch.as_tensor(grad_norm, device=self.device)

                metrics_gpu["policy_loss"] += policy_loss.detach()
                metrics_gpu["value_loss"] += value_loss.detach()
                metrics_gpu["entropy"] += entropy.detach()
                metrics_gpu["loss"] += loss.detach()
                metrics_gpu["approx_kl"] += approx_kl
                metrics_gpu["clip_frac"] += clip_frac
                metrics_gpu["kl_ref"] += kl_ref_val
                metrics_gpu["ratio/mean"] += ratio.mean().detach()
                metrics_gpu["ratio/std"] += ratio.std().detach()
                metrics_gpu["ratio/max"] = torch.maximum(metrics_gpu["ratio/max"], ratio.max().detach())
                metrics_gpu["kl/max"] = torch.maximum(metrics_gpu["kl/max"], kl_max)
                metrics_gpu["value/predicted_mean"] += values.mean().detach()
                metrics_gpu["grad_norm"] += grad_norm_t.detach()
                n_batches_run += 1

        # Single sync point: pull every accumulated metric back to Python floats.
        total_metrics = {k: v.item() for k, v in metrics_gpu.items()}

        num_batches = max(1, n_batches_run)
        avg_keys = ["policy_loss", "value_loss", "entropy", "loss", "approx_kl",
                     "clip_frac", "ratio/mean", "ratio/std", "value/predicted_mean", "grad_norm",
                     "kl_ref"]
        for k in avg_keys:
            total_metrics[k] /= num_batches

        total_metrics["adv/raw_mean"] = adv_raw_mean
        total_metrics["adv/raw_std"] = adv_raw_std
        total_metrics["return/mean"] = return_mean
        total_metrics["return/std"] = return_std
        total_metrics["return/target_mean"] = returns.mean().item()

        var_returns = returns.var()
        if var_returns < 1e-8:
            total_metrics["explained_variance"] = 0.0
        else:
            total_metrics["explained_variance"] = (1.0 - (returns - last_epoch_values).var() / var_returns).item()

        total_metrics["time/data_transfer_s"] = t_data_transfer
        total_metrics["time/teacher_fwd_s"] = t_teacher_fwd
        total_metrics["time/main_fwd_bwd_s"] = t_main_fwd_bwd
        total_metrics["time/n_batches"] = n_batches_run

        self.total_steps += 1
        return total_metrics
