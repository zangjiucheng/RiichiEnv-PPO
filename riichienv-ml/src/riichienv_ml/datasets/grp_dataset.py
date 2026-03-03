import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from riichienv_ml.datasets.mjai_logs import GrpFeatureEncoder, _compute_rank, load_mjai_replay


class GrpReplayDataset(IterableDataset):
    """GRP dataset that reads .jsonl replay files via MjaiReplay.

    For each kyoku in each replay, extracts GRP features and predicts
    the final hanchan ranking for each player. Yields (features_tensor, rank_one_hot).
    """

    def __init__(
        self,
        data_glob: str,
        n_players: int = 4,
        replay_rule: str = "mjsoul",
        is_train: bool = True,
    ):
        self.data_glob = data_glob
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.is_train = is_train

    def _get_files(self) -> list[str]:
        files = sorted(glob.glob(self.data_glob, recursive=True))
        return files

    def _encode_features(self, grp_features: dict, player_idx: int) -> torch.Tensor:
        n = self.n_players
        score_norm = 35000.0 if n == 3 else 25000.0

        scores = np.array(
            [grp_features[f"p{i}_init_score"] / score_norm for i in range(n)]
            + [grp_features[f"p{i}_end_score"] / score_norm for i in range(n)]
            + [grp_features[f"p{i}_delta_score"] / 12000.0 for i in range(n)],
            dtype=np.float32,
        )
        round_meta = np.array([
            grp_features["chang"] / 3.0,
            grp_features["ju"] / 3.0,
            grp_features["ben"] / 4.0,
            grp_features["liqibang"] / 4.0,
        ], dtype=np.float32)
        player = np.zeros(n, dtype=np.float32)
        player[player_idx] = 1.0

        x = np.concatenate([scores, round_meta, player])
        return torch.from_numpy(x)

    def _encode_label(self, final_scores: list, player_idx: int) -> torch.Tensor:
        n = self.n_players
        rank = _compute_rank(final_scores, player_idx, n)
        y = np.zeros(n, dtype=np.float32)
        y[rank] = 1.0
        return torch.from_numpy(y)

    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)

        # Shard files across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        worker_label = "main"
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]
            worker_label = str(worker_info.id)

        try:
            max_error_logs = int(os.getenv("RIICHIENV_ML_MAX_ERROR_LOGS", "1"))
        except ValueError:
            max_error_logs = 1
        error_count = 0

        buffer = []
        for file_path in files:
            try:
                replay = load_mjai_replay(file_path, self.replay_rule, n_players=self.n_players)
                # Collect all kyoku features first to get final hanchan scores
                kyoku_features_list = []
                for kyoku in replay.take_kyokus():
                    grp_features = GrpFeatureEncoder(kyoku, self.n_players).encode()
                    kyoku_features_list.append(grp_features)

                if not kyoku_features_list:
                    continue

                # Final hanchan ranking from the last kyoku's end scores
                last_features = kyoku_features_list[-1]
                final_scores = [last_features[f"p{i}_end_score"] for i in range(self.n_players)]

                for grp_features in kyoku_features_list:
                    for player_idx in range(self.n_players):
                        x = self._encode_features(grp_features, player_idx)
                        y = self._encode_label(final_scores, player_idx)
                        buffer.append((x, y))
            except Exception as e:
                error_count += 1
                if error_count <= max_error_logs or error_count % 100 == 0:
                    print(
                        f"[GRP worker={worker_label}] Error processing {file_path}: {e} "
                        f"(errors={error_count})"
                    )
                continue

            # Flush buffer periodically to limit memory usage
            if len(buffer) >= 10000:
                if self.is_train:
                    random.shuffle(buffer)
                yield from buffer
                buffer.clear()

        # Flush remaining
        if buffer:
            if self.is_train:
                random.shuffle(buffer)
            yield from buffer

        if error_count > max_error_logs:
            print(
                f"[GRP worker={worker_label}] Suppressed {error_count - max_error_logs} "
                "additional replay errors."
            )
