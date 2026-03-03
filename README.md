# riichienv-ml

Mahjong RL training pipeline for RiichiEnv.

## Dataset

- Dataset/conversion project: [NikkeTryHard/tenhou-to-mjai](https://github.com/NikkeTryHard/tenhou-to-mjai)

Expected relative data layout (from repo root):

```text
data/
  mjsoul/
    mjsoul-4p/
      train/.../*.jsonl
      val/.../*.jsonl
    mjsoul-3p/
      train/.../*.jsonl
      val/.../*.jsonl
```

Create the local directories first:

```sh
mkdir -p data/mjsoul/mjsoul-4p/train data/mjsoul/mjsoul-4p/val
mkdir -p data/mjsoul/mjsoul-3p/train data/mjsoul/mjsoul-3p/val
mkdir -p artifacts/4p artifacts/3p artifacts/teacher
```

## Stage 0: Environment Setup

Run from repo root (`/Users/jiucheng/Dev/RiichiEnv`):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Optional (disable W&B cloud sync):

```sh
export WANDB_MODE=offline
```

Optional (dataset cache for BC/CQL):

```sh
export RIICHIENV_ML_ENABLE_CACHE=1
export RIICHIENV_ML_CACHE_DIR=artifacts/cache/mc
export RIICHIENV_ML_CACHE_DTYPE=float16
export RIICHIENV_ML_QUARANTINE_BAD_REPLAYS=1
export RIICHIENV_ML_BAD_REPLAY_DIR=artifacts/cache/bad_replays
export RIICHIENV_ML_MAX_ERROR_LOGS=1
```

Optional (some clusters with cuDNN v8 symbol mismatch):

```sh
export RIICHIENV_DISABLE_CUDNN_V8=1
# If it still crashes, disable cuDNN entirely (slower but stable):
export RIICHIENV_DISABLE_CUDNN=1
```

## Stage 0.5: Import Downloaded `.zip` Data

Convert zip files containing `.mjson` into local `train/val` `.jsonl` files:

```sh
# 4-player: import one year zip
python riichienv-ml/scripts/import_mjai_zip.py --players 4p ~/Downloads/2024.zip

# 4-player: import multiple zips
python riichienv-ml/scripts/import_mjai_zip.py --players 4p ~/Downloads/2023.zip ~/Downloads/2024.zip

# 3-player
python riichienv-ml/scripts/import_mjai_zip.py --players 3p ~/Downloads/2024_3p.zip
```

Notes:
- Default validation split is `5%` (`--val-ratio 0.05`), deterministic by file name.
- Use `--overwrite` if you want to regenerate already imported files.
- Use `--dry-run` to preview output paths without writing files.
- Import now validates UTF-8 by default (`--encoding-errors strict`).
- If your source has minor broken bytes, you can force salvage with `--encoding-errors replace`.

If training logs contain errors like `Read error: stream did not contain valid UTF-8`, rebuild files:

```sh
python riichienv-ml/scripts/import_mjai_zip.py \
  --players 4p \
  --overwrite \
  ./.cache/riichi-data/2023.zip \
  ./.cache/riichi-data/2024.zip \
  ./.cache/riichi-data/2025.zip
```

If training logs contain many `Replay desync` errors, quarantine bad files first:

```sh
python riichienv-ml/scripts/quarantine_bad_replays.py \
  --glob "data/mjsoul/mjsoul-4p/train/**/*.jsonl" \
  --players 4 \
  --rule mjsoul
```

## Stage 1: Train GRP (Reward Model)

This trains `artifacts/{3p|4p}/grp_model.pth`.

```sh
# 4-player
python riichienv-ml/scripts/train_grp.py -c riichienv-ml/src/riichienv_ml/configs/4p/grp.yml

# 3-player
python riichienv-ml/scripts/train_grp.py -c riichienv-ml/src/riichienv_ml/configs/3p/grp.yml
```

## Stage 2: Offline Policy Training (Choose One)

### Option A: CQL

This trains `artifacts/{3p|4p}/cql_model.pth` and uses GRP from Stage 1.
On first run, replay features are cached to `artifacts/cache/mc`; later runs reuse cache.

```sh
# 4-player
python riichienv-ml/scripts/train_cql.py -c riichienv-ml/src/riichienv_ml/configs/4p/cql.yml

# 3-player
python riichienv-ml/scripts/train_cql.py -c riichienv-ml/src/riichienv_ml/configs/3p/cql.yml
```

### Option B: BC (Offline logs BC)

This trains `artifacts/{3p|4p}/bc_model.pth`.

```sh
# 4-player
python riichienv-ml/scripts/train_bc.py -c riichienv-ml/src/riichienv_ml/configs/4p/bc_logs.yml

# 3-player
python riichienv-ml/scripts/train_bc.py -c riichienv-ml/src/riichienv_ml/configs/3p/bc_logs.yml
```

## Stage 3: Online PPO Finetuning

### PPO after CQL

```sh
# 4-player
python riichienv-ml/scripts/train_ppo.py -c riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml

# 3-player
python riichienv-ml/scripts/train_ppo.py -c riichienv-ml/src/riichienv_ml/configs/3p/ppo.yml
```

### PPO after BC (4p)

```sh
python riichienv-ml/scripts/train_ppo.py -c riichienv-ml/src/riichienv_ml/configs/4p/bc_ppo.yml
```

## Optional: Online Teacher BC (4p)

Requires an external teacher plugin (not included in this repo):

```sh
python riichienv-ml/scripts/train_bc.py -c riichienv-ml/src/riichienv_ml/configs/4p/bc_model.yml
```

## CPU-Only Note

If CUDA is unavailable, override key flags, for example:

```sh
python riichienv-ml/scripts/train_grp.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/grp.yml \
  --device cpu --num_workers 4
```
