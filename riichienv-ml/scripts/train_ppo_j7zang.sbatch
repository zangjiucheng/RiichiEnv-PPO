#!/bin/bash
# SLURM launcher for Stage 3 (Online PPO) on the j7zang MFCF gpu-gen node.
#
# Hardware/QOS reference (MFCF gpu-gen partition, qos_rtx6000_max):
#   1x RTX6000 GPU (QOS caps this at 1 regardless of cpus_per_task/mem),
#   up to 25 CPU cores / 200G mem (requesting 24/160G, leaving headroom
#   below the ceiling), single node, MaxWall=24:00:00, MaxJobsPU=1 (only
#   one job runs at a time), MaxSubmitPU=3.
#
# Usage:
#   sbatch riichienv-ml/scripts/train_ppo_j7zang.sbatch
#   CONFIG=.../ppo_transformer.yml sbatch riichienv-ml/scripts/train_ppo_j7zang.sbatch
#
#SBATCH --job-name=riichi-ppo
#SBATCH --partition=gpu-gen
#SBATCH --account=normal
#SBATCH --qos=qos_rtx6000_max
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j7zang@uwaterloo.ca
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Editable settings (override via env var when submitting) ---------------
REPO_ROOT="${REPO_ROOT:-$HOME/RiichiEnv-PPO}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$REPO_ROOT/.venv/bin/activate}"
# CNN model (ppo_v2.yml) is the recommended default: it's cheap enough per
# inference step that CPU-only rollout workers keep up even at this scale.
# The transformer config (ppo_transformer.yml) does ~600-token self-attention
# per step and will still bottleneck CPU workers despite the extra cores --
# only switch to it if you also pass worker_device=cuda / gpu_per_worker>0
# below so inference batches onto the same RTX6000 as the learner (the QOS
# caps this node at 1 GPU, so hero+baseline workers would be sharing it).
CONFIG="${CONFIG:-$REPO_ROOT/riichienv-ml/src/riichienv_ml/configs/4p/ppo_v2.yml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/artifacts/4p/ppo_v2/checkpoints}"
WORKER_DEVICE="${WORKER_DEVICE:-cpu}"
GPU_PER_WORKER="${GPU_PER_WORKER:-0.0}"
# num_workers / num_envs_per_worker are overridable so a GPU-worker run can
# trade throughput for GPU memory: each cuda worker keeps its own model +
# frozen baseline model resident on the single shared GPU, so 20 workers can
# exceed the 48 GiB card (OOM mid-step). Drop NUM_WORKERS for the transformer
# (e.g. 12) to leave headroom for the learner's training-step allocations.
NUM_WORKERS="${NUM_WORKERS:-20}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-32}"
# Optional override for the config's own num_steps (e.g. to fit a wall-clock
# budget); empty means "use whatever the config specifies".
NUM_STEPS="${NUM_STEPS:-}"

cd "$REPO_ROOT"
mkdir -p logs "$CHECKPOINT_DIR"

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# NOT setting OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS globally
# here: PPOWorker.__init__ already calls torch.set_num_threads(1) itself, so
# the N Ray rollout worker processes are already correctly single-threaded
# for torch regardless of these env vars. Setting them at the shell level
# would ALSO throttle the main driver process's own CPU-bound work between
# rollouts (np.concatenate over the ~100k+ transition batch, the PPO update's
# tensor prep) down to 1 thread, for no benefit -- that work happens after
# all workers are idle, so the driver can use the freed-up cores. Measured
# ~9.5 min/step with the blanket throttle in place (worker collection itself
# was only ~2 min); if this doesn't meaningfully close that gap, the
# bottleneck is elsewhere (GPU-bound PPO update, KL-vs-teacher forward pass,
# Ray dispatch overhead) and these should be reinstated for worker safety.

# Confirmed the login node (tsubmit) reaches api.wandb.ai; the gpu-gen compute
# nodes weren't directly verified (an in-job srun --overlap probe hit an
# unrelated slurmd RPC error) but share the same network egress in practice,
# so default to online. If a compute node turns out to be air-gapped, wandb's
# init will fail/retry -- override with WANDB_MODE=offline when submitting
# and sync afterwards via `wandb sync <run_dir> --entity ... --project ...`
# (the entity/project flags matter -- a bare `wandb sync` can silently land
# in the wrong entity if the submitting account's default differs).
export WANDB_MODE="${WANDB_MODE:-online}"

# Resume from the latest checkpoint if one exists, so re-submitting this
# script after the 24h MaxWall (or a preemption) continues training instead
# of restarting from the config's `load_model`.
LATEST_CKPT=""
if [ -d "$CHECKPOINT_DIR" ]; then
  LATEST_CKPT=$(ls -v "$CHECKPOINT_DIR"/model_*.pth 2>/dev/null | tail -n 1 || true)
fi

RESUME_ARGS=()
if [ -n "$LATEST_CKPT" ]; then
  echo "Resuming from checkpoint: $LATEST_CKPT"
  RESUME_ARGS=(--load_model "$LATEST_CKPT")
else
  echo "No checkpoint found under $CHECKPOINT_DIR; using load_model from $CONFIG" \
       "(must already exist -- run Stage 1/2 GRP+CQL first if it doesn't)."
fi

# Meaningful W&B run name: config + what it resumed from + a timestamp for
# uniqueness across re-submissions (init_wandb() never sets `name` itself,
# so wandb.init() picks up WANDB_NAME automatically).
CONFIG_NAME=$(basename "$CONFIG" .yml)
if [ -n "$LATEST_CKPT" ]; then
  RESUME_TAG="resume$(basename "$LATEST_CKPT" .pth | sed 's/^model_//')"
else
  RESUME_TAG="fresh"
fi
export WANDB_NAME="${WANDB_NAME:-${CONFIG_NAME}-${RESUME_TAG}-$(date +%Y%m%d_%H%M%S)}"
echo "W&B run name: $WANDB_NAME"

# num_workers/num_envs_per_worker tuned for cpus_per_task=24: 20 Ray rollout
# worker processes (~1 core each, since PPOWorker pins torch to a single
# thread) + headroom for the Ray driver/GCS and the learner's CPU-side work
# (data transfer, Python overhead). GPU stays untouched by workers (worker_device=cpu,
# gpu_per_worker=0.0) since the QOS caps this node at 1 GPU, reserved for the
# learner. Watch each worker's "trans/s" log line (emitted every
# collect_episodes() call) and rebalance num_workers/num_envs_per_worker if
# collection is the bottleneck.
STEP_ARGS=()
if [ -n "$NUM_STEPS" ]; then
  STEP_ARGS=(--num_steps "$NUM_STEPS")
fi

python riichienv-ml/scripts/train_ppo.py \
  -c "$CONFIG" \
  --num_workers "$NUM_WORKERS" \
  --num_envs_per_worker "$NUM_ENVS_PER_WORKER" \
  --worker_device "$WORKER_DEVICE" \
  --gpu_per_worker "$GPU_PER_WORKER" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "${STEP_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
