# Transformer-on-GPU: debugging & optimization write-up

How the from-scratch PPO transformer (`configs/4p/ppo_transformer_scratch.yml`)
was brought from "hangs / OOMs on every submit" to a healthy GPU run on the
j7zang HPC node (single RTX6000 Ada, 47.5 GiB). Written as a post-mortem so
the misdiagnosis and the real fixes are both on record.

---

## TL;DR (中文速览)

- **真正的病根不是死锁**：`train_ppo_j7zang.sbatch` 默认 `WORKER_DEVICE=cpu`，并以命令行参数**覆盖**了配置里的 `worker_device: cuda`。所以那些"卡死"的作业其实一直在 **CPU 上慢慢跑**，GPU 空转。0% GPU 利用率 + 0.9% 显存被误读成了"CUDA 死锁"。
- **一行修复**：提交时加 `WORKER_DEVICE=cuda GPU_PER_WORKER=0.04`。GPU 立刻 100% 占用，dry-run ~13 分钟通过。
- **暴露出两个真实的 OOM**（48 GiB 的卡）：(1) 20 个 worker 各驻留两个模型 → 采集阶段爆；(2) learner 的 PPO 更新在 `batch_size=512` 下单进程占 20 GiB → 更新阶段爆。降到 **8 workers + batch_size 256 + `expandable_segments`** 解决。
- **提速**：开 TF32（`set_float32_matmul_precision('high')`），更新从 802s → 624s。
- **结果**：从零训练，step 50 eval **rank 1.04 / reward 9.74**（vs 冻结随机 baseline）。

---

## 1. The symptom

Every real submission of the transformer config "hung": GPU utilization sat
at **0%**, GPU memory at **~0.9%** (475 MiB), the process alive but making no
visible progress for 20–40 minutes, at which point it was killed. Affected
jobs: `5550, 5558, 5564, 5565, 5566, 5568, 5569, 5570, 5571`.

## 2. The false trail — a "CUDA deadlock" that never existed

The 0%-GPU signature looked exactly like a concurrent CUDA-init deadlock, so a
lot of effort went into ruling that out. **Nine diagnostic jobs** (`5551–5563`)
reproduced progressively more of the real conditions:

| Probe | Scale | Result |
|---|---|---|
| Ray + fractional-GPU actors touching CUDA | 20 workers | ✅ 2.7 s |
| Real `TransformerActorCritic` + `torch.compile` | 4→20 workers | ✅ 10–27 s |
| Dynamic-shape recompilation | single proc | ✅ no storm |
| Real `evaluate_episodes()` gameplay | 1 worker | ✅ 12 s |
| Full-scale `evaluate_episodes()` | 20×32 | ✅ 88 s |
| + driver's own learner model on GPU | 20×32 | ✅ 68 s |
| + `grp_model` / `pts_weight` | 20×32 | ✅ 64 s |
| Exact `evaluate_parallel()` | 20×32 | ✅ 66 s |
| Real `train_ppo.py` entrypoint + wandb, `num_envs=4` | 20 workers | ✅ passed |

**Every diagnostic passed** — because each one explicitly requested GPU workers
(`--worker_device cuda --gpu_per_worker 0.04`). The real submissions did not.
Three defensive changes made while chasing the phantom are correct but were
**not** the fix:

- `24dcc11` — `RIICHIENV_DISABLE_WORKER_COMPILE` toggle + `TORCHINDUCTOR_COMPILE_THREADS=1` (guards a *real* but different risk: 20 workers each forking a CPU-count Inductor compile pool).
- `73dd04a` — forward `RIICHIENV_*` / `TORCHINDUCTOR_*` env vars to Ray workers (they don't inherit the driver's arbitrary env once a `runtime_env` is set; the compile toggle was silently dead code in the worker until this).
- `d4c2362` — serialize CUDA worker startup via a `ready()` gate (still good hygiene against concurrent CUDA-context init).

Disabling compile proved compile was innocent: **zero Inductor activity in the
log, still 0% GPU**. That finally forced the right question — *are the workers
even on the GPU?*

## 3. The real root cause

`train_ppo_j7zang.sbatch` was written for the CNN model, which uses **CPU**
rollout workers. Its defaults:

```bash
WORKER_DEVICE="${WORKER_DEVICE:-cpu}"      # line 43
GPU_PER_WORKER="${GPU_PER_WORKER:-0.0}"    # line 44
```

are passed as `--worker_device` / `--gpu_per_worker`, which **override** the
config's `worker_device: cuda`. So every "hung" transformer job ran its
rollout workers on CPU. The tells, in hindsight:

- **GPU util 0%, GPU memory ~0.9%** = just the driver/learner's idle CUDA
  context; no worker ever created a GPU context.
- CPU-worker transformer dry-run takes **~90 min** (measured) — a hang if you
  kill it at 20–40 min.
- The one job that *worked* (diagnostic `5563`) explicitly passed
  `--worker_device cuda`.

**Fix:** submit with `WORKER_DEVICE=cuda GPU_PER_WORKER=0.04`. GPU immediately
went to **100% util / ~89% memory**, dry-run passed in ~13 min.

## 4. The real problems this exposed — two distinct OOMs

Once workers were actually on the GPU, the run OOM'd — twice, in two different
phases, because each GPU rollout worker keeps **its own model + a frozen
baseline model** resident on the single shared 48 GiB card.

**OOM #1 — collection phase, 20 workers.** 20 × ~2.2 GiB (worker models) +
~4.4 GiB (driver) ≈ 48 GiB > 47.5. Fixed by parameterizing worker count
(`fcdee5e`) and dropping to 12.

**OOM #2 — learner update phase, 12 workers.** Collection now fit, but the
first PPO update OOM'd: the **learner alone used 20.18 GiB** doing the
backward over `batch_size=512` minibatches (~600-token self-attention is
activation-heavy), on top of ~26 GiB of resident worker models. Fixed by
halving the learner minibatch to `batch_size: 256` (`1c5ef1f`) and dropping to
8 workers.

**Final memory budget (job 5576):** 8 × ~2.2 GiB workers (~17.6) + ~12 GiB
learner (batch 256) ≈ **30 GiB**, ~17 GiB headroom, plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` against fragmentation.

## 5. Optimizations

| Change | Commit | Effect |
|---|---|---|
| Run rollout workers on the GPU | (submit flag) | the actual fix; CPU→GPU dry-run 90 min → ~13 min |
| `batch_size` 512 → 256 | `1c5ef1f` | halves learner update activation memory (fixes OOM #2) |
| Parameterize `NUM_WORKERS` / `NUM_ENVS_PER_WORKER` | `fcdee5e` | trade throughput for GPU memory headroom |
| **TF32 matmuls** (`set_float32_matmul_precision('high')`) | `b366c00` | learner update **802 s → 624 s** (~22% faster) |
| Serialize CUDA worker startup (`ready()` gate) | `d4c2362` | one-at-a-time context init; ~1–2 min extra startup, paid once |
| Forward `RIICHIENV_*`/`TORCHINDUCTOR_*`/`TORCH_*`/`PYTORCH_*` env to workers | `73dd04a`, `1c5ef1f` | worker-side toggles actually take effect |
| `RIICHIENV_DISABLE_WORKER_COMPILE` + serial Inductor compile | `24dcc11` | guards the compile-pool oversubscription risk |
| `checkpoint_interval: 25` | `1550a28` | mid-run snapshots for resume safety + pull/test |

Note on TF32's modest 22%: with `batch_size=256` the update runs ~1024
minibatches, so per-minibatch Python/masking/softmax overhead dilutes the
matmul speedup. `async_rollout` already overlaps collection with the previous
update (`t_collect` → 0 from step 2), so the effective step time is ≈ the
update time (~624 s ≈ **10.4 min/step**; 110 steps ≈ **~19 h**, fits the 24 h
wall clock — fp32 would have needed ~33 h).

## 6. Canonical submit command

```bash
WORKER_DEVICE=cuda GPU_PER_WORKER=0.04 NUM_WORKERS=8 \
sbatch --export=ALL,NUM_STEPS=110,WORKER_DEVICE=cuda,GPU_PER_WORKER=0.04,NUM_WORKERS=8,\
RIICHIENV_DISABLE_CUDNN=0,RIICHIENV_DISABLE_CUDNN_V8=0,\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  riichienv-ml/scripts/train_ppo_j7zang.sbatch
```

- `WORKER_DEVICE`/`GPU_PER_WORKER`/`NUM_WORKERS` — the memory-critical trio.
- `RIICHIENV_DISABLE_CUDNN(_V8)=0` — re-enable cuDNN (the README default
  disables it "for cluster stability"; it's a big CNN speedup and harmless here).
- Health check: `system.gpu.0.gpu` should be ~100% once dry-run starts. **0% +
  ~0.9% memory = workers are on CPU** (missing `WORKER_DEVICE=cuda`).

## 7. Results (job 5576, from random init)

| Step | entropy | explained_var | rollout rank | note |
|---|---|---|---|---|
| 1–2 | 2.00 | 0.80 | 3.2 | random policy |
| 13 | 1.81 | 0.93 | 2.34 | crossed 2.5 (beats baseline) |
| 31 | 1.51 | 0.94 | 1.46 | plateau broke, strong learning |
| 50 | 1.48 | 0.92 | 1.28 | — |
| **50 eval** | — | — | **1.04 ± 0.02** | **reward 9.74** vs baseline |

Healthy, fast, statistically solid learning. **Caveat:** the eval "baseline"
is a *frozen copy of the model's own random init*, so `rank 1.04` means "crushes
a random network," not absolute strength — and this eval is near-saturated
(can't beat random by much more than 1.0x), so it stops being informative from
here. See §8.

---

## 8. Next: evaluate against a real opponent (transformer vs CNN `model_100`)

The vs-random-baseline eval has served its purpose (confirmed learning) and is
now saturated. To measure **real relative strength**, play the trained
transformer against the existing CNN checkpoint `model_100.pth` (from the
`ppo.yml` lineage, eval rank ~2.28 vs *its* baseline — a genuinely competent
opponent).

**Approach — 1 hero vs 3 fixed opponents, seat-rotated:**

- **Hero:** transformer `model_110.pth` + `ppo_transformer_scratch.yml`
  (`TransformerActorCritic` / `SequenceFeaturePackedEncoder`).
- **Opponents (×3):** CNN `model_100.pth` + `ppo.yml`
  (`ActorCriticNetwork` / `feat_v1`, 74-channel).
- Play N (e.g. 200) hanchan, rotating the hero across all 4 seats for fairness;
  report hero mean rank / reward / 1st-rate.
- **Interpretation:** hero rank **< 2.5** ⇒ transformer is stronger than the CNN;
  **> 2.5** ⇒ weaker. This is an *absolute-ish* signal because the opponent is a
  real trained policy, not random.

**Implementation options (both already in the codebase):**

1. **`AgentEvaluator` plugin** (`riichienv_ml.agent_eval`, registered as the
   `riichienv` evaluator entry-point) — already does hero-vs-fixed-`Agent`
   opponents with seat rotation. Configure `evaluator.evaluator_name: riichienv`
   with `opponents: [{config_path: .../ppo.yml, model_path: .../model_100.pth}]`.
   Mixed architectures are fine: the hero loads from the transformer config, each
   opponent `Agent` loads from the CNN config independently.
2. **Standalone head-to-head script** — a small driver that loads both models
   into one `RiichiEnv` and plays N games. Cleaner for a one-off comparison
   outside a training run; can also be pointed at Mortal or any other agent.

**Plan:** run this the moment the transformer `model_110.pth` lands (a checkpoint
watcher is armed to auto-pull it). Also worth a symmetric run with roles
swapped (CNN hero vs 3 transformers) to cancel any seat/first-mover bias.

**Akagi note:** the same `model_110.pth` + `ppo_transformer_scratch.yml` can be
wired into the Akagi bot bridge (`akagi_bot/`), which reads `model_class` /
`encoder_class` / `encoder` kwargs straight from the config YAML, so the
transformer loads with no bridge changes — just point `config_path_4p` /
`model_path_4p` at it.

---

## Appendix — commit reference

| Commit | Title |
|---|---|
| `aa7f9be` | Run transformer rollout inference on GPU instead of CPU |
| `24dcc11` | Guard against torch.compile deadlock in GPU rollout workers |
| `73dd04a` | Forward RIICHIENV_/TORCHINDUCTOR_ env vars to Ray workers |
| `d4c2362` | Serialize CUDA rollout worker startup to avoid init deadlock |
| `fcdee5e` | Parameterize NUM_WORKERS/NUM_ENVS_PER_WORKER in the j7zang sbatch |
| `1c5ef1f` | Cut transformer learner batch_size 512→256; forward PYTORCH_ env |
| `b366c00` | Enable TF32 matmuls for transformer learner + workers |
| `1550a28` | Add checkpoint_interval=25 to transformer config for resume safety |
