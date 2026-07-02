"""Shared per-kyoku reward + GAE math.

Extracted from `_ppo_worker.PPOWorker` so the same on-policy trajectory
processing can run outside a Ray actor -- specifically, the Akagi bot
bridge's online-learning path (`scripts/akagi_bot/`), which builds one
hanchan's trajectory in a plain Python process instead of a
`collect_episodes()` rollout batch.
"""
from __future__ import annotations


def compute_kyoku_rewards(reward_predictor, prev_scores, cur_scores,
                           round_wind, oya, honba, riichi_sticks,
                           n_players: int) -> list[float]:
    """Potential-based per-kyoku reward via the trained GRP rank predictor.

    Returns one reward per seat. `reward_predictor` may be None (no GRP
    model configured), in which case all rewards are 0.0.
    """
    if reward_predictor is None:
        return [0.0] * n_players

    deltas = [cur_scores[i] - prev_scores[i] for i in range(n_players)]
    grp_features = {}
    for i in range(n_players):
        grp_features[f"p{i}_init_score"] = prev_scores[i]
        grp_features[f"p{i}_end_score"] = cur_scores[i]
        grp_features[f"p{i}_delta_score"] = deltas[i]
    grp_features["chang"] = round_wind
    grp_features["ju"] = oya
    grp_features["ben"] = honba
    grp_features["liqibang"] = riichi_sticks
    return reward_predictor.calc_all_player_rewards(grp_features)


def compute_kyoku_gae(traj: list[dict], kyoku_reward: float,
                       gamma: float, gae_lambda: float) -> tuple[list[float], list[float]]:
    """GAE over one kyoku's trajectory, treated as its own episode: reward
    is 0 for every step except the last (which gets `kyoku_reward`), and
    the value bootstrap at the terminal step is 0 (no next kyoku carry-over).

    `traj` is a list of dicts each containing at least a "value" key (the
    critic's estimate at that step). Returns (advantages, returns), same
    length/order as `traj`.
    """
    T = len(traj)
    values = [step["value"] for step in traj]
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

        delta = reward + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns
