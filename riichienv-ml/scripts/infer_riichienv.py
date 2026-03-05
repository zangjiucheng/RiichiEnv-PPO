"""Run local inference in RiichiEnv and print per-step model decisions.

Example:
  python riichienv-ml/scripts/infer_riichienv.py \
    --config riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml \
    --model artifacts/4p/ppo/checkpoints/model_50.pth \
    --episodes 1 --hero-seat 0
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
from riichienv import RiichiEnv

from riichienv_ml.config import load_config, import_class
from riichienv_ml.utils import resolve_train_device

if os.getenv("RIICHIENV_DISABLE_CUDNN_V8", "1").lower() not in ("0", "false", "no"):
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
import torch
if os.getenv("RIICHIENV_DISABLE_CUDNN", "1").lower() not in ("0", "false", "no"):
    torch.backends.cudnn.enabled = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference demo in RiichiEnv")
    parser.add_argument(
        "--config",
        type=str,
        default="riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml",
        help="Config YAML path (reads game/model/encoder settings)",
    )
    parser.add_argument(
        "--section",
        type=str,
        choices=["ppo", "cql", "bc"],
        default="ppo",
        help="Config section used for model/encoder settings",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Checkpoint path, e.g. artifacts/4p/ppo/checkpoints/model_50.pth",
    )
    parser.add_argument("--episodes", type=int, default=1, help="How many hanchan to play")
    parser.add_argument("--hero-seat", type=int, default=0, help="Seat controlled by printed policy")
    parser.add_argument(
        "--opponent-policy",
        type=str,
        choices=["random", "model"],
        default="random",
        help="How opponents act",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu/cuda/mps; default resolves automatically",
    )
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of greedy")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k legal actions for hero")
    parser.add_argument("--max-hero-decisions", type=int, default=0, help="0 means no limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _resolve_path(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith("_orig_mod.") for k in state.keys()):
        return state
    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            new_state[k.replace("_orig_mod.", "", 1)] = v
        else:
            new_state[k] = v
    return new_state


def _load_weights_compat(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    state = _strip_compile_prefix(state)

    has_actor = any(k.startswith("actor_head.") for k in state.keys())
    has_critic = any(k.startswith("critic_head.") for k in state.keys())
    has_v_head = any(k.startswith("v_head.") for k in state.keys())
    has_a_head = any(k.startswith("a_head.") for k in state.keys())

    if has_actor and has_critic:
        model.load_state_dict(state, strict=False)
        return

    if has_v_head and has_a_head:
        mapped = {}
        for k, v in state.items():
            if k.startswith("a_head."):
                mapped[k.replace("a_head.", "actor_head.")] = v
            elif k.startswith("v_head."):
                mapped[k.replace("v_head.", "critic_head.")] = v
            elif k.startswith("aux_head."):
                continue
            else:
                mapped[k] = v
        model.load_state_dict(mapped, strict=False)
        return

    if any(k.startswith("head.") for k in state.keys()):
        mapped = {}
        for k, v in state.items():
            if k.startswith("head."):
                mapped[k.replace("head.", "actor_head.")] = v
            else:
                mapped[k] = v
        model.load_state_dict(mapped, strict=False)
        return

    model.load_state_dict(state, strict=False)


def _describe_action(action_obj) -> str:
    return repr(action_obj)


def _choose_model_action(
    model: torch.nn.Module,
    encoder,
    obs,
    device: torch.device,
    sample: bool,
    temperature: float,
) -> tuple[int, torch.Tensor]:
    feat = encoder.encode(obs).unsqueeze(0).to(device)
    mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).to(device).bool()

    with torch.no_grad():
        output = model(feat)
        if isinstance(output, tuple):
            values = output[0][0]
        else:
            values = output[0]

        masked_values = values.masked_fill(~mask, -1e9)
        if sample:
            t = max(1e-4, float(temperature))
            probs = torch.softmax(masked_values / t, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())
        else:
            action_idx = int(torch.argmax(masked_values).item())

    return action_idx, masked_values.detach().cpu()


def _print_hero_decision(
    episode_idx: int,
    decision_idx: int,
    env,
    action_idx: int,
    action_obj,
    masked_values_cpu: torch.Tensor,
    topk: int,
) -> None:
    legal_idx = torch.where(masked_values_cpu > -1e8)[0]
    if legal_idx.numel() == 0:
        print(f"[ep={episode_idx}] hero_decision={decision_idx} no legal action")
        return

    k = min(int(topk), int(legal_idx.numel()))
    top_vals, top_ids = torch.topk(masked_values_cpu, k=k)
    probs = torch.softmax(masked_values_cpu, dim=-1)

    top_items = []
    for i in range(k):
        aid = int(top_ids[i].item())
        score = float(top_vals[i].item())
        p = float(probs[aid].item())
        top_items.append(f"{aid}(score={score:.3f},p={p:.3f})")

    round_wind = getattr(env, "round_wind", "?")
    oya = getattr(env, "oya", "?")
    honba = getattr(env, "honba", "?")
    kyotaku = getattr(env, "riichi_sticks", "?")
    kyoku_idx = getattr(env, "kyoku_idx", "?")

    print(
        f"[ep={episode_idx}] d={decision_idx} kyoku_idx={kyoku_idx} "
        f"wind={round_wind} oya={oya} honba={honba} kyotaku={kyotaku} "
        f"choose={action_idx} { _describe_action(action_obj) }"
    )
    print(f"  top{k}: " + ", ".join(top_items))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg_root = load_config(args.config)
    cfg = getattr(cfg_root, args.section)
    game = cfg.game

    if not (0 <= args.hero_seat < game.n_players):
        raise ValueError(f"--hero-seat must be in [0, {game.n_players - 1}]")

    model_path = _resolve_path(args.model)
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device_str = resolve_train_device(args.device or cfg.device)
    device = torch.device(device_str)

    ModelClass = import_class(cfg.model_class)
    EncoderClass = import_class(cfg.encoder_class)
    model = ModelClass(**cfg.model.model_dump()).to(device)
    encoder = EncoderClass(tile_dim=game.tile_dim)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)}")
    _load_weights_compat(model, state)
    model.eval()

    print(
        f"Inference start: section={args.section} model={model_path} "
        f"device={device} game_mode={game.game_mode} n_players={game.n_players}"
    )
    print(f"Hero seat={args.hero_seat}, opponents={args.opponent_policy}, episodes={args.episodes}")

    ranks_all = []
    scores_all = []

    for ep in range(args.episodes):
        env = RiichiEnv(game_mode=game.game_mode)
        obs_dict = env.reset(
            scores=list(game.starting_scores),
            round_wind=0,
            oya=0,
            honba=0,
            kyotaku=0,
        )

        hero_decisions = 0
        while not env.done():
            env_actions = {}
            for pid, obs in obs_dict.items():
                legal = obs.legal_actions()
                if not legal:
                    continue

                use_model = pid == args.hero_seat or args.opponent_policy == "model"
                if use_model:
                    action_idx, masked_values_cpu = _choose_model_action(
                        model=model,
                        encoder=encoder,
                        obs=obs,
                        device=device,
                        sample=args.sample,
                        temperature=args.temperature,
                    )
                    action_obj = obs.find_action(action_idx)
                    if action_obj is None:
                        action_obj = legal[0]
                        action_idx = int(action_obj.encode())
                    env_actions[pid] = action_obj

                    if pid == args.hero_seat:
                        hero_decisions += 1
                        _print_hero_decision(
                            episode_idx=ep,
                            decision_idx=hero_decisions,
                            env=env,
                            action_idx=action_idx,
                            action_obj=action_obj,
                            masked_values_cpu=masked_values_cpu,
                            topk=args.topk,
                        )
                else:
                    env_actions[pid] = random.choice(legal)

            if not env_actions:
                break

            obs_dict = env.step(env_actions)
            if args.max_hero_decisions > 0 and hero_decisions >= args.max_hero_decisions:
                print(f"[ep={ep}] reached --max-hero-decisions={args.max_hero_decisions}, stop early")
                break

        final_scores = list(env.scores())
        final_ranks = list(env.ranks())
        hero_rank = final_ranks[args.hero_seat]
        hero_score = final_scores[args.hero_seat]
        ranks_all.append(float(hero_rank))
        scores_all.append(float(hero_score))

        print(
            f"[ep={ep}] done={env.done()} hero_rank={hero_rank} hero_score={hero_score} "
            f"ranks={final_ranks} scores={final_scores}"
        )

    print(
        f"Summary: episodes={args.episodes} hero_rank_mean={np.mean(ranks_all):.3f} "
        f"hero_score_mean={np.mean(scores_all):.1f}"
    )


if __name__ == "__main__":
    main()
