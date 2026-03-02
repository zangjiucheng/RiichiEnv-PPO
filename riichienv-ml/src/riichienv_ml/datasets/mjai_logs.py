import glob
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from riichienv import MjaiReplay


def load_mjai_replay(file_path: str, replay_rule: str):
    """Load replay with compatibility across riichienv versions.

    Newer versions accept ``rule=...`` while older ones do not.
    """
    try:
        return MjaiReplay.from_jsonl(file_path, rule=replay_rule)
    except TypeError as e:
        if "unexpected keyword argument 'rule'" in str(e):
            return MjaiReplay.from_jsonl(file_path)
        raise


def _compute_rank(end_scores: list, player_id: int, n_players: int) -> int:
    """Compute rank (0=1st, n-1=last) from end-of-kyoku scores."""
    scores = np.array(end_scores[:n_players], dtype=np.float64)
    return int((-scores).argsort(kind='stable').argsort(kind='stable')[player_id])


class GrpFeatureEncoder:
    """Extracts GRP features from a kyoku, parameterized by n_players."""

    def __init__(self, kyoku, n_players: int = 4):
        self.kyoku = kyoku
        self.n_players = n_players

    def encode(self) -> dict:
        feat = self.kyoku.take_grp_features()
        n = self.n_players
        row = {}
        for i in range(n):
            row[f"p{i}_init_score"] = feat["round_initial_scores"][i]
            row[f"p{i}_end_score"] = feat["round_end_scores"][i]
            row[f"p{i}_delta_score"] = feat["round_delta_scores"][i]
        row["chang"] = feat["chang"]
        row["ju"] = feat["ju"]
        row["ben"] = feat["ben"]
        row["liqibang"] = feat["liqibang"]
        return row


class BaseDataset(IterableDataset):
    def __init__(self, data_sources, reward_predictor=None, gamma=0.99,
                 is_train=True, n_players=4, replay_rule="mjsoul", encoder=None):
        self.data_sources = data_sources
        self.reward_predictor = reward_predictor
        self.gamma = gamma
        self.is_train = is_train
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.encoder = encoder

    def _get_files(self):
        if isinstance(self.data_sources, list):
            return self.data_sources
        elif isinstance(self.data_sources, str):
            return glob.glob(self.data_sources, recursive=True)
        return []


class MCDataset(BaseDataset):
    """Yields (features, action_id, return, mask, rank).

    Target: Monte-Carlo Return (G_t), decayed.
    Uses MjaiReplay.from_jsonl() for replay parsing.
    """
    CACHE_VERSION = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_dir = os.getenv("RIICHIENV_ML_CACHE_DIR", "artifacts/cache/mc")
        self.cache_dir = Path(cache_dir)
        self.enable_cache = os.getenv("RIICHIENV_ML_ENABLE_CACHE", "1").lower() not in ("0", "false", "no")
        self.cache_dtype = os.getenv("RIICHIENV_ML_CACHE_DTYPE", "float16").lower()
        if self.cache_dtype not in ("float16", "float32"):
            self.cache_dtype = "float16"

    def _cache_key(self, file_path: str) -> str:
        stat = os.stat(file_path)
        encoder_name = type(self.encoder).__name__ if self.encoder is not None else "None"
        encoder_dim = getattr(self.encoder, "tile_dim", None) if self.encoder is not None else None

        rp_model_path = getattr(self.reward_predictor, "model_path", "")
        rp_model_sig = ""
        if rp_model_path:
            try:
                rp_stat = os.stat(rp_model_path)
                rp_model_sig = f"{rp_model_path}|{rp_stat.st_mtime_ns}|{rp_stat.st_size}"
            except OSError:
                rp_model_sig = rp_model_path
        rp_pts = getattr(self.reward_predictor, "pts_weight", None)

        key_payload = {
            "version": self.CACHE_VERSION,
            "file_path": str(file_path),
            "file_mtime_ns": stat.st_mtime_ns,
            "file_size": stat.st_size,
            "gamma": float(self.gamma),
            "n_players": int(self.n_players),
            "replay_rule": str(self.replay_rule),
            "encoder_name": encoder_name,
            "encoder_tile_dim": encoder_dim,
            "reward_model": rp_model_sig,
            "reward_pts_weight": rp_pts,
            "cache_dtype": self.cache_dtype,
        }
        return hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_path(self, file_path: str) -> Path:
        digest = self._cache_key(file_path)
        return self.cache_dir / f"{digest}.npz"

    def _write_cache(self, cache_path: Path, samples: list[tuple]) -> None:
        if not samples:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        feats = np.stack([
            s[0].numpy() if isinstance(s[0], torch.Tensor) else s[0]
            for s in samples
        ])
        if self.cache_dtype == "float16":
            feats = feats.astype(np.float16, copy=False)
        else:
            feats = feats.astype(np.float32, copy=False)

        actions = np.asarray([s[1] for s in samples], dtype=np.int16)
        targets = np.asarray([s[2] for s in samples], dtype=np.float32)
        masks = np.stack([s[3] for s in samples]).astype(np.uint8, copy=False)
        ranks = np.asarray([s[4] for s in samples], dtype=np.uint8)

        tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp.npz")
        np.savez(tmp_path, features=feats, actions=actions, targets=targets, masks=masks, ranks=ranks)
        os.replace(tmp_path, cache_path)

    def _iter_from_cache(self, cache_path: Path):
        with np.load(cache_path, allow_pickle=False) as data:
            features = data["features"]
            actions = data["actions"]
            targets = data["targets"]
            masks = data["masks"]
            ranks = data["ranks"]
            for i in range(actions.shape[0]):
                feat = torch.from_numpy(features[i].astype(np.float32, copy=False))
                yield feat, int(actions[i]), float(targets[i]), masks[i], int(ranks[i])

    def _build_samples_from_replay(self, file_path: str) -> list[tuple]:
        replay = load_mjai_replay(file_path, self.replay_rule)
        samples = []

        for kyoku in replay.take_kyokus():
            grp_features = GrpFeatureEncoder(kyoku, self.n_players).encode()

            assert self.reward_predictor is not None
            all_rewards = self.reward_predictor.calc_all_player_rewards(grp_features)
            end_scores = [grp_features[f"p{i}_end_score"] for i in range(self.n_players)]

            for player_id in range(self.n_players):
                trajectory = []
                final_reward = all_rewards[player_id]
                rank = _compute_rank(end_scores, player_id, self.n_players)

                for obs, action in kyoku.steps(player_id):
                    features = self.encoder.encode(obs)
                    action_id = action.encode()

                    mask_bytes = obs.mask()
                    mask = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
                    assert 0 <= action_id < mask.shape[0], f"action_id should be in [0, {mask.shape[0]})"
                    assert mask[action_id] == 1, f"action_id {action_id} should be legal"
                    trajectory.append((features, action_id, mask))

                T = len(trajectory)
                for t, (feat, act, mask) in enumerate(trajectory):
                    decayed = final_reward * (self.gamma ** (T - t - 1))
                    samples.append((feat, act, decayed, mask, rank))
        return samples

    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)

        # Shard files across DataLoader workers to avoid duplicated work
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        for file_path in files:
            cache_path = self._cache_path(file_path) if self.enable_cache else None

            try:
                if cache_path is not None and cache_path.exists():
                    yield from self._iter_from_cache(cache_path)
                    continue

                buffer = self._build_samples_from_replay(file_path)
                if self.is_train:
                    random.shuffle(buffer)

                if cache_path is not None:
                    self._write_cache(cache_path, buffer)

                yield from buffer
            except Exception as e:
                # Skip malformed or corrupted replay files (e.g., broken gzip stream).
                print(f"Error processing replay: {file_path}: {e}")
                if cache_path is not None and cache_path.exists():
                    try:
                        cache_path.unlink()
                    except OSError:
                        pass
                continue


class DiscardHistoryDataset(MCDataset):
    """MCDataset with discard history decay features (78 channels)."""
    pass


class DiscardHistoryShantenDataset(MCDataset):
    """MCDataset with discard history + shanten features (94 channels)."""
    pass


class ExtendedDataset(MCDataset):
    """MCDataset with extended features (215 channels)."""
    pass
