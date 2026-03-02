import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RankPredictor(nn.Module):
    def __init__(self, input_dim: int = 20, n_players: int = 4):
        super().__init__()
        self.n_players = n_players
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_players)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class RewardPredictor:
    def __init__(self, model_path: str, pts_weight: list[float],
                 n_players: int = 4, input_dim: int | None = None,
                 device: str = "cuda"):
        self.model_path: str = model_path
        self.device: str = device
        self.n_players: int = n_players
        self.pts_weight: list[float] = pts_weight
        self._score_norm = 35000.0 if n_players == 3 else 25000.0

        if input_dim is None:
            input_dim = n_players * 4 + 4

        self.model: RankPredictor = RankPredictor(input_dim, n_players)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(torch.device(device))
        self.model = self.model.eval()

    def _calc_pts(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return torch.softmax(self.model(x), dim=1) @ torch.tensor(self.pts_weight, device=self.device).float()

    def calc_pts_reward(self, row: dict, player_idx: int) -> np.ndarray:
        n = self.n_players
        scores = np.array([row[f"p{i}_init_score"] for i in range(n)]
                          + [row[f"p{i}_end_score"] for i in range(n)])
        delta_scores = np.array([row[f"p{i}_delta_score"] for i in range(n)])
        scores = scores / self._score_norm
        delta_scores = delta_scores / 12000.0
        round_meta = np.array([
            row["chang"] / 3.0, row["ju"] / 3.0, row["ben"] / 4.0, row["liqibang"] / 4.0
        ])
        player = np.zeros(n)
        player[player_idx] = 1.0

        x = np.concatenate([scores, delta_scores, round_meta, player])
        return x

    def calc_all_player_rewards(self, grp_features: dict) -> list[float]:
        """Compute final rewards for all players in one batched forward pass."""
        n = self.n_players
        scores = np.array(
            [grp_features[f"p{i}_init_score"] for i in range(n)]
            + [grp_features[f"p{i}_end_score"] for i in range(n)]
        ) / self._score_norm
        delta_scores = np.array(
            [grp_features[f"p{i}_delta_score"] for i in range(n)]
        ) / 12000.0
        round_meta = np.array([
            grp_features["chang"] / 3.0, grp_features["ju"] / 3.0,
            grp_features["ben"] / 4.0, grp_features["liqibang"] / 4.0,
        ])
        base = np.concatenate([scores, delta_scores, round_meta])

        input_dim = n * 4 + 4
        xs = np.zeros((n, input_dim), dtype=np.float32)
        for pid in range(n):
            xs[pid, :len(base)] = base
            xs[pid, len(base) + pid] = 1.0

        xs_t = torch.from_numpy(xs).to(self.device)
        mean_pts = float(np.mean(self.pts_weight))
        with torch.inference_mode():
            pts_w = torch.tensor(self.pts_weight, device=self.device).float()
            pts = torch.softmax(self.model(xs_t), dim=1) @ pts_w
        rewards = (pts - mean_pts).cpu().tolist()
        return rewards

    def calc_pts_rewards(self, kyoku_features: list[dict], player_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs = []
        for row in kyoku_features:
            x = self.calc_pts_reward(row, player_idx)
            xs.append(x)

        xs = torch.from_numpy(np.array(xs)).float().to(self.device)
        pts = torch.concat([
            torch.tensor([np.mean(self.pts_weight)], device=self.device).float(),
            self._calc_pts(xs)
        ], dim=0)

        rewards = pts[1:] - pts[:-1]
        return pts[1:], rewards
