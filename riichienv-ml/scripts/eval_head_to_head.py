"""Head-to-head evaluation: 1 hero model vs 3 fixed opponents, seat-rotated.

Unlike training's "vs frozen self-baseline" eval (which saturates once the
policy beats random), this pits two *different trained policies* against each
other for an absolute-ish strength comparison -- e.g. the from-scratch
transformer vs the CNN `model_100.pth`.

Self-contained agent loader (mirrors scripts/akagi_bot/bot.py): reads
model_class / encoder_class / model dims / encoder kwargs from each config YAML
and passes `**cfg.encoder` to the encoder -- which the shared
riichienv_ml.agents.Agent / AgentEvaluator do NOT do, so they can't correctly
load the transformer (its SequenceFeaturePackedEncoder needs
max_prog_len/max_cand_len).

Usage:
  python riichienv-ml/scripts/eval_head_to_head.py \
    --hero-config    riichienv-ml/src/riichienv_ml/configs/4p/ppo_transformer_scratch.yml \
    --hero-model     artifacts/4p/ppo_transformer_scratch/checkpoints/model_110.pth \
    --opp-config     riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml \
    --opp-model      artifacts/4p/ppo/checkpoints/model_100.pth \
    --games 200 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "riichienv-ml" / "src"))

from riichienv import RiichiEnv  # noqa: E402
from riichienv_ml.config import GAME_PARAMS, import_class, load_config  # noqa: E402


def _strip_compile_prefix(state: dict) -> dict:
    if not any(k.startswith("_orig_mod.") for k in state):
        return state
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()}


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    def _step(p: Path) -> int:
        try:
            return int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1
    cands = sorted(path.glob("model_*.pth"), key=_step)
    if not cands:
        raise FileNotFoundError(f"no model_*.pth under {path}")
    return cands[-1]


class HeroAgent:
    """Greedy (argmax) agent loaded from a training config + checkpoint,
    passing the config's encoder kwargs so transformer encoders build with the
    right PACKED_SIZE."""

    def __init__(self, config_path: str, model_path: str, tile_dim: int, device: str):
        self.device = torch.device(device)
        cfg = load_config(config_path).ppo
        model_cls = import_class(cfg.model_class)
        encoder_cls = import_class(cfg.encoder_class)
        self.model = model_cls(**cfg.model.model_dump()).to(self.device)
        self.encoder = encoder_cls(tile_dim=tile_dim, **cfg.encoder)

        resolved = _resolve_checkpoint(Path(model_path))
        state = torch.load(resolved, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        state = _strip_compile_prefix(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [warn] {resolved.name}: missing={len(missing)} unexpected={len(unexpected)}",
                  file=sys.stderr)
        self.model.eval()
        self.name = f"{cfg.model_class.rsplit('.', 1)[-1]}:{resolved.name}"

    @torch.inference_mode()
    def act(self, obs):
        feat = self.encoder.encode(obs).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).to(self.device).bool()
        out = self.model(feat)
        logits = out[0] if isinstance(out, tuple) else out
        idx = int(logits.masked_fill(~mask, -1e9).argmax(dim=-1).item())
        action = obs.find_action(idx)
        if action is not None:
            return action
        legal = obs.legal_actions()
        return legal[0] if legal else None


def play_game(env: RiichiEnv, agents: dict, hero_seat: int, starting_scores):
    obs_dict = env.reset(scores=list(starting_scores))
    while not env.done():
        actions = {}
        for pid, obs in obs_dict.items():
            if obs.legal_actions():
                a = agents[pid].act(obs)
                if a is not None:
                    actions[pid] = a
        if not actions:
            break
        obs_dict = env.step(actions)
    return env.ranks()[hero_seat], env.scores()[hero_seat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero-config", required=True)
    ap.add_argument("--hero-model", required=True)
    ap.add_argument("--opp-config", required=True)
    ap.add_argument("--opp-model", required=True)
    ap.add_argument("--games", type=int, default=200, help="games per hero seat is games//4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-players", type=int, default=4)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable, using cpu", file=sys.stderr)
        device = "cpu"

    params = GAME_PARAMS[args.n_players]
    tile_dim = params["tile_dim"]
    game_mode = params["game_mode"]
    starting_scores = params["starting_scores"]

    print(f"Loading hero:      {args.hero_model}")
    hero = HeroAgent(args.hero_config, args.hero_model, tile_dim, device)
    print(f"  -> {hero.name}")
    print(f"Loading opponent:  {args.opp_model}")
    opp = HeroAgent(args.opp_config, args.opp_model, tile_dim, device)
    print(f"  -> {opp.name}")

    env = RiichiEnv(game_mode=game_mode)
    per_seat = max(1, args.games // args.n_players)
    ranks, scores = [], []
    for hero_seat in range(args.n_players):
        for _ in range(per_seat):
            agents = {s: (hero if s == hero_seat else opp) for s in range(args.n_players)}
            r, sc = play_game(env, agents, hero_seat, starting_scores)
            ranks.append(r)
            scores.append(sc)
        print(f"  seat {hero_seat}: {per_seat} games done "
              f"(running mean rank {np.mean(ranks):.3f})", file=sys.stderr)

    ranks = np.array(ranks, dtype=np.float64)
    n = len(ranks)
    print("\n=== HEAD-TO-HEAD RESULT ===")
    print(f"hero      : {hero.name}")
    print(f"opponents : 3x {opp.name}")
    print(f"games     : {n}")
    print(f"mean rank : {ranks.mean():.3f} ± {ranks.std()/np.sqrt(n):.3f}   (lower=better, 2.5=even)")
    print(f"mean score: {np.mean(scores):.0f}")
    for r in range(1, args.n_players + 1):
        print(f"  {r}位 rate: {(ranks == r).mean():.1%}")
    verdict = "HERO STRONGER" if ranks.mean() < 2.5 else "HERO WEAKER"
    print(f"verdict   : {verdict} than opponent (rank {ranks.mean():.3f} vs even 2.5)")


if __name__ == "__main__":
    main()
