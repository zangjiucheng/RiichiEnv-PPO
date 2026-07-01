"""Akagi mjai-bot bridge for RiichiEnv-PPO trained models.

Speaks Akagi's line-based mjai bot protocol (see
../../../Akagi/mjai_bot/README.md): read one JSON array of mjai events per
stdin line, track game state with riichienv.RiichiEnv.observe_event(), run
the configured checkpoint on whichever Observation comes back, and write
exactly one JSON mjai reaction per stdout line.

Settings (see manifest.toml) select which training config + checkpoint to
load, separately for 4p and 3p, since Akagi's active_4p/active_3p slots are
independent and either mode's start_game can arrive in this process.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

NOTIFY_PREFIX = "@@AKAGI_NOTIFY@@ "


def notify(level: str, title: str, body: str | None = None, *, sticky: bool = False, id: str | None = None) -> None:
    payload: dict[str, Any] = {"level": level, "title": title}
    if body is not None:
        payload["body"] = body
    if sticky:
        payload["sticky"] = True
    if id is not None:
        payload["id"] = id
    sys.stderr.write(NOTIFY_PREFIX + json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stderr.flush()


# bot.py lives at <repo-root>/riichienv-ml/scripts/akagi_bot/bot.py (whether
# reached directly or through a symlink under Akagi/mjai_bot/<name>/ --
# .resolve() follows the symlink to this physical location either way).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "riichienv-ml" / "src"))

from riichienv import RiichiEnv, MeldType  # noqa: E402
from riichienv.convert import tid_to_mjai  # noqa: E402
from riichienv_ml.config import GAME_PARAMS, import_class, load_config  # noqa: E402

DEFAULT_SETTINGS = {
    "config_path_4p": "riichienv-ml/src/riichienv_ml/configs/4p/ppo_v2.yml",
    "model_path_4p": "artifacts/4p/ppo_v2/checkpoints",
    "config_path_3p": "riichienv-ml/src/riichienv_ml/configs/3p/ppo.yml",
    "model_path_3p": "artifacts/3p/ppo/checkpoints",
    "device": "cpu",
    "sample": False,
    "temperature": 1.0,
}


def _load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    cfg_path = os.environ.get("AKAGI_BOT_CONFIG")
    if cfg_path and Path(cfg_path).is_file():
        with open(cfg_path) as f:
            settings.update(json.load(f))
    return settings


def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")

    def _step(p: Path) -> int:
        stem = p.stem  # "model_123"
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    candidates = sorted(path.glob("model_*.pth"), key=_step)
    if not candidates:
        raise FileNotFoundError(f"no model_*.pth checkpoints found under {path}")
    return candidates[-1]


def _strip_compile_prefix(state: dict) -> dict:
    if not any(k.startswith("_orig_mod.") for k in state):
        return state
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}


class _Agent:
    """Loads one checkpoint and picks an action for an Observation."""

    def __init__(self, config_path: Path, model_path: Path, tile_dim: int,
                 device_str: str, sample: bool, temperature: float):
        if device_str == "cuda" and not torch.cuda.is_available():
            notify("warn", "CUDA unavailable", "Falling back to CPU inference.")
            device_str = "cpu"
        self.device = torch.device(device_str)

        cfg = load_config(str(config_path)).ppo
        model_cls = import_class(cfg.model_class)
        encoder_cls = import_class(cfg.encoder_class)

        self.model = model_cls(**cfg.model.model_dump()).to(self.device)
        self.encoder = encoder_cls(tile_dim=tile_dim, **cfg.encoder)

        resolved = _resolve_checkpoint(model_path)
        state = torch.load(resolved, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        state = _strip_compile_prefix(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            notify("warn", "Checkpoint key mismatch",
                   f"missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()

        self.sample = sample
        self.temperature = temperature
        notify("success", "Model loaded", f"{resolved.name} ({cfg.model_class.rsplit('.', 1)[-1]})")

    @torch.inference_mode()
    def act(self, obs: Any):
        feat = self.encoder.encode(obs).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).to(self.device).bool()

        output = self.model(feat)
        logits = output[0] if isinstance(output, tuple) else output
        masked = logits.masked_fill(~mask, -1e9)

        if self.sample:
            temp = max(1e-4, float(self.temperature))
            probs = torch.softmax(masked / temp, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())
        else:
            action_idx = int(torch.argmax(masked, dim=-1).item())

        action = obs.find_action(action_idx)
        if action is not None:
            return action
        legal = obs.legal_actions()
        return legal[0] if legal else None


def _build_agent(n_players: int, settings: dict) -> _Agent:
    suffix = "4p" if n_players == 4 else "3p"
    params = GAME_PARAMS[n_players]
    config_path = _resolve(settings[f"config_path_{suffix}"])
    model_path = _resolve(settings[f"model_path_{suffix}"])
    return _Agent(
        config_path, model_path, params["tile_dim"],
        settings["device"], bool(settings["sample"]), float(settings["temperature"]),
    )


def _for_engine(ev: dict) -> dict:
    """Adapt an incoming Akagi/mjai event for riichienv's MjaiEvent parser.

    riichienv-core's own StartGame variant declares `id: Option<String>`,
    but Akagi (and the mjai spec) send the seat index as an integer -- so
    passing the event through unchanged makes observe_event()/apply_event()
    raise a JSON type-mismatch error. Everything else riichienv doesn't
    recognize (num_players, kyoku_first, aka_flag, ...) is silently ignored
    by serde, so only `id` needs coercing.
    """
    if ev.get("type") == "start_game" and isinstance(ev.get("id"), int):
        ev = {**ev, "id": str(ev["id"])}
    return ev


def _enrich(action: Any, raw: dict, seat: int, trigger: dict, env: RiichiEnv) -> dict:
    """Fill in mjai fields riichienv's Action.to_mjai() doesn't emit.

    riichienv's to_mjai() covers type/actor/pai/consumed but omits a few
    fields Akagi's schema (src/schema/mjai/mod.rs) requires: dahai.tsumogiri,
    chi/pon/daiminkan.target, kakan.consumed, hora.target.
    """
    t = raw.get("type")

    if t == "dahai":
        raw["tsumogiri"] = bool(
            trigger.get("type") == "tsumo"
            and trigger.get("actor") == seat
            and trigger.get("pai") == raw.get("pai")
        )
    elif t in ("chi", "pon", "daiminkan"):
        raw["target"] = trigger.get("actor")
    elif t == "kakan":
        added_tid = action.tile
        for meld in env.melds[seat]:
            if meld.meld_type == MeldType.Pon and meld.tiles and meld.tiles[0] // 4 == added_tid // 4:
                raw["consumed"] = [tid_to_mjai(x) for x in meld.tiles]
                break
    elif t == "hora":
        if trigger.get("type") == "tsumo" and trigger.get("actor") == seat:
            raw["target"] = seat
        else:
            raw["target"] = trigger.get("actor")
        raw.setdefault("pai", trigger.get("pai"))
    elif t == "kita":
        raw.setdefault("pai", "N")

    return raw


def _initial_seat() -> int:
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass
    env_val = os.environ.get("AKAGI_PLAYER_ID")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return 0


class _Session:
    """Holds the one RiichiEnv + Agent for this process's single game."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.seat = _initial_seat()
        self.env: RiichiEnv | None = None
        self.agent: _Agent | None = None

    def react(self, events: list[dict]) -> dict:
        if not events:
            return {"type": "none"}

        trigger = events[-1]
        last_obs = None

        for ev in events:
            if ev.get("type") == "start_game":
                if ev.get("id") is not None:
                    self.seat = ev["id"]
                n_players = ev.get("num_players", 4)
                game_mode = GAME_PARAMS[n_players]["game_mode"]
                self.env = RiichiEnv(game_mode=game_mode)
                self.agent = _build_agent(n_players, self.settings)
                notify("info", "Game started", f"seat={self.seat} players={n_players}")

            if self.env is None:
                continue
            obs = self.env.observe_event(_for_engine(ev), self.seat)
            if obs is not None:
                last_obs = obs

        if last_obs is None or self.agent is None:
            return {"type": "none"}

        action = self.agent.act(last_obs)
        if action is None:
            return {"type": "none"}

        raw = json.loads(action.to_mjai())
        return _enrich(action, raw, self.seat, trigger, self.env)


def main() -> None:
    settings = _load_settings()
    session = _Session(settings)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            events = json.loads(line)
        except Exception as e:
            sys.stderr.write(f"bad event batch: {e}\n")
            events = []

        try:
            reaction = session.react(events)
        except Exception as e:  # never desync the stdout protocol
            sys.stderr.write(f"bot error: {e}\n")
            notify("error", "Bot error", str(e))
            reaction = {"type": "none"}

        sys.stdout.write(json.dumps(reaction, separators=(",", ":")) + "\n")
        sys.stdout.flush()

        if any(isinstance(ev, dict) and ev.get("type") == "end_game" for ev in events):
            break


if __name__ == "__main__":
    main()
