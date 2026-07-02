"""Online-learning PPO mini-update (beta).

Spawned detached by `bot.py` once `ONLINE_LEARNING_HANCHAN_PER_UPDATE`
hanchan of live trajectory data have accumulated under
`<model_path>_live/pending/`. Runs as its own process rather than inline in
the bot bridge because Akagi force-kills the bot process ~500ms after
end_game (see Akagi's `runner.rs::reset()`), far too short a window to run
even a small PPO step.

Not part of the mjai stdin/stdout protocol -- reads/writes only files under
`--live_dir`, logs to stdout/stderr (captured by bot.py into
`<live_dir>/update.log`), and exits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bot  # noqa: E402  (reuses REPO_ROOT/_resolve_checkpoint/notify; also
            # puts riichienv-ml/src on sys.path as a side effect of import)
from riichienv_ml.trainers._ppo_learner import PPOLearner  # noqa: E402

# Fixed, conservative hyperparams for the beta -- not exposed as bot
# settings (only lr/alpha_kl are, via meta.json/manifest.toml), since these
# secondary knobs matter far less than getting the KL anchor and cadence
# right, and a smaller settings surface is easier to reason about while
# this is still experimental.
PPO_CLIP = 0.1
PPO_EPOCHS = 1
ENTROPY_COEF = 0.001
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 64


def _next_version(live_dir: Path) -> int:
    existing = list(live_dir.glob("model_*.pth"))
    if not existing:
        return 1

    def _n(p: Path) -> int:
        try:
            return int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            return 0
    return max(_n(p) for p in existing) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live_dir", required=True)
    args = parser.parse_args()
    live_dir = Path(args.live_dir)
    lock_path = live_dir / ".update.lock"

    try:
        meta = json.loads((live_dir / "meta.json").read_text())
        pending_dir = live_dir / "pending"
        # Snapshot now: a fresh bot.py process for the *next* hanchan may
        # write new pending files concurrently while this update runs --
        # those are picked up by a future update, not this one.
        batch_files = sorted(pending_dir.glob("hanchan_*.npz"))
        if not batch_files:
            bot.notify("warn", "Online learning: nothing to update", "no pending batches found")
            return

        arrays = [np.load(f) for f in batch_files]
        n_transitions = sum(len(a["action"]) for a in arrays)
        rollout_batch = {
            "features": torch.from_numpy(np.concatenate([a["features"] for a in arrays])),
            "masks": torch.from_numpy(np.concatenate([a["mask"] for a in arrays])),
            "actions": torch.from_numpy(np.concatenate([a["action"] for a in arrays])),
            "old_log_probs": torch.from_numpy(np.concatenate([a["log_prob"] for a in arrays])),
            "advantages": torch.from_numpy(np.concatenate([a["advantage"] for a in arrays])),
            "returns": torch.from_numpy(np.concatenate([a["return_"] for a in arrays])),
        }
        for a in arrays:
            a.close()

        device = meta.get("device", "cpu")
        learner = PPOLearner(
            device=device, lr=meta["lr"], gamma=meta["gamma"], gae_lambda=meta["gae_lambda"],
            ppo_clip=PPO_CLIP, ppo_epochs=PPO_EPOCHS, entropy_coef=ENTROPY_COEF,
            value_coef=VALUE_COEF, max_grad_norm=MAX_GRAD_NORM, weight_decay=0.0,
            alpha_kl=meta["alpha_kl"], alpha_kl_warmup_steps=0,
            batch_size=min(BATCH_SIZE, n_transitions),
            model_config=meta["model_config"], model_class=meta["model_class"],
            teacher_model=meta["base_checkpoint_path"],
        )

        current_ckpt = (bot._resolve_checkpoint(live_dir) if any(live_dir.glob("model_*.pth"))
                        else Path(meta["base_checkpoint_path"]))
        learner.load_weights(str(current_ckpt))

        metrics = learner.update(rollout_batch)

        version = _next_version(live_dir)
        save_path = live_dir / f"model_{version}.pth"
        torch.save(learner.get_weights(), save_path)

        for f in batch_files:
            f.unlink(missing_ok=True)

        summary = (
            f"online update: {n_transitions} transitions from {len(batch_files)} hanchan "
            f"-> {save_path.name} | loss={metrics.get('loss', 0):.4f} "
            f"policy_loss={metrics.get('policy_loss', 0):.4f} "
            f"value_loss={metrics.get('value_loss', 0):.4f} "
            f"kl_ref={metrics.get('kl_ref', 0):.4f}"
        )
        print(summary)
        bot.notify("success", "Online learning: update complete", save_path.name)
    except Exception as e:
        print(f"online update failed: {e}", file=sys.stderr)
        bot.notify("error", "Online learning update failed", str(e))
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
