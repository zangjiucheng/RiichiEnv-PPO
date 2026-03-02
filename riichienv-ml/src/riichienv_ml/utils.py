import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import wandb
from loguru import logger


def setup_logging(output_dir: str, script_name: str) -> Path:
    """Configure loguru to log to both stderr and a file.

    Log file is created in ``output_dir`` with name
    ``{script_name}_{YYYYMMDD_HHMMSS}.log``.

    Returns the log file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{script_name}_{timestamp}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="DEBUG")
    logger.info(f"Logging to {log_path}")
    return log_path


def _git_info() -> dict:
    """Collect git revision information."""
    info = {}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        info["git_dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _git_diff() -> str | None:
    """Return the combined staged + unstaged diff, or None."""
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        )
        return diff if diff.strip() else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def init_wandb(
    cfg,
    config_path: str | None = None,
    extra_config: dict | None = None,
    wandb_dir: str | None = None,
) -> wandb.sdk.wandb_run.Run:
    """Initialize a W&B run with traceability artifacts.

    - Logs full config to ``wandb.config``
    - Records git commit, branch, dirty status
    - Uploads config YAML as artifact (``config``)
    - Uploads ``git diff HEAD`` as artifact (``code-diff``) if dirty

    Args:
        cfg: A pydantic config object (GrpConfig / BcConfig / PpoConfig)
             that has ``wandb_entity``, ``wandb_project``, ``wandb_tags``,
             ``wandb_group`` fields (inherited from WandbConfig).
        config_path: Path to the YAML config file (uploaded as artifact).
        extra_config: Additional key-values to merge into wandb.config.
        wandb_dir: Directory for W&B local run data. If None, uses the
                   output/checkpoint directory from the config.
    """
    git = _git_info()

    run_config = cfg.model_dump()
    run_config.update(git)
    if extra_config:
        run_config.update(extra_config)

    # Determine wandb local data directory
    if wandb_dir is None:
        if hasattr(cfg, "checkpoint_dir"):
            wandb_dir = cfg.checkpoint_dir
        elif hasattr(cfg, "output"):
            wandb_dir = str(Path(cfg.output).parent)
    if wandb_dir:
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)

    init_kwargs = dict(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        tags=cfg.wandb_tags or None,
        group=cfg.wandb_group,
        config=run_config,
        save_code=True,
        dir=wandb_dir,
    )
    try:
        run = wandb.init(**init_kwargs)
    except wandb.errors.CommError as e:
        logger.warning(f"W&B online init failed ({e}). Falling back to offline mode.")
        run = wandb.init(**init_kwargs, mode="offline")

    # Upload config YAML as artifact
    if config_path and os.path.isfile(config_path):
        art_config = wandb.Artifact("config", type="config")
        art_config.add_file(config_path)
        run.log_artifact(art_config)
        logger.info(f"Uploaded config artifact: {config_path}")

    # Upload git diff as artifact if there are uncommitted changes.
    # NOTE: This may include sensitive data from uncommitted changes.
    diff = _git_diff()
    if diff:
        logger.warning("Uploading uncommitted git diff to W&B. "
                       "Ensure no secrets/credentials are in your working tree.")
        art_diff = wandb.Artifact("code-diff", type="code")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".diff", prefix="git_diff_", delete=False,
        ) as f:
            f.write(diff)
            diff_path = f.name
        try:
            art_diff.add_file(diff_path, name="git_diff.diff")
            run.log_artifact(art_diff)
            logger.info("Uploaded code-diff artifact (uncommitted changes)")
        finally:
            os.unlink(diff_path)

    logger.info(
        f"W&B run: project={cfg.wandb_project} tags={cfg.wandb_tags} "
        f"commit={git.get('git_commit', 'N/A')[:8]} dirty={git.get('git_dirty', 'N/A')}"
    )
    return run


def resolve_train_device(requested: str | None) -> str:
    """Resolve torch training device with macOS-aware fallback.

    Priority:
    - requested == cuda* and CUDA available -> cuda*
    - requested == mps and MPS available -> mps
    - requested == cpu -> cpu
    - otherwise:
      - on macOS with MPS -> mps
      - else -> cpu
    """
    req = (requested or "cuda").lower()

    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            logger.warning("CUDA is unavailable. Falling back to MPS on macOS.")
            return "mps"
        logger.warning("CUDA is unavailable. Falling back to CPU.")
        return "cpu"

    if req == "mps":
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS is unavailable. Falling back to CPU.")
        return "cpu"

    if req == "cpu":
        return "cpu"

    return req


def resolve_worker_device(requested: str | None) -> str:
    """Resolve Ray worker device (cpu/cuda only)."""
    req = (requested or "cpu").lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("worker_device=cuda but CUDA is unavailable. Falling back to CPU workers.")
        return "cpu"
    return "cpu"


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
