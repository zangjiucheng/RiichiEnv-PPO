"""Run a local browser UI for human-vs-3AI play in RiichiEnv.

Example:
  python riichienv-ml/scripts/human_vs_ai_web.py --host 127.0.0.1 --port 8000 \
    --config riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml \
    --model checkpoints/model_50.pth
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from http import cookies
from itertools import count
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

import numpy as np
from riichienv import RiichiEnv
from riichienv.convert import tid_to_mjai_list
from riichienv.visualizer import GameViewer

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_SRC = SCRIPT_PATH.parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from riichienv_ml.config import import_class, load_config
from riichienv_ml.utils import resolve_train_device

if os.getenv("RIICHIENV_DISABLE_CUDNN_V8", "1").lower() not in ("0", "false", "no"):
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import torch

if os.getenv("RIICHIENV_DISABLE_CUDNN", "1").lower() not in ("0", "false", "no"):
    torch.backends.cudnn.enabled = False


HUMAN_SEAT = 0
DEFAULT_GAME_MODE = "4p-red-half"
DEFAULT_CONFIG_PATH = "riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml"
DEFAULT_MODEL_PATH = "checkpoints/v1.pth"
SESSION_COOKIE = "riichi_browser_session"

ACTION_TYPE_LABELS = {
    "dahai": "Discard",
    "chi": "Chi",
    "pon": "Pon",
    "daiminkan": "Open Kan",
    "ankan": "Closed Kan",
    "kakan": "Added Kan",
    "reach": "Riichi",
    "hora": "Win",
    "ryukyoku": "Draw",
    "none": "Pass",
    "nukidora": "Kita",
}

SHORTCUTS = list("1234567890qwertyuiopasdfghjklzxcvbnm")
VIEWER_REVISION = count(1)


def _resolve_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _build_player_names(human_seat: int, n_players: int, ai_model_path: str) -> list[str]:
    ai_name = Path(ai_model_path).stem
    names: list[str] = []
    ai_idx = 1
    for seat in range(n_players):
        if seat == human_seat:
            names.append("Human")
        else:
            names.append(f"{ai_name} #{ai_idx}")
            ai_idx += 1
    return names


def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state.keys()):
        return state

    stripped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("_orig_mod."):
            stripped[key.replace("_orig_mod.", "", 1)] = value
        else:
            stripped[key] = value
    return stripped


def _load_weights_compat(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    state = _strip_compile_prefix(state)

    has_actor = any(key.startswith("actor_head.") for key in state.keys())
    has_critic = any(key.startswith("critic_head.") for key in state.keys())
    has_v_head = any(key.startswith("v_head.") for key in state.keys())
    has_a_head = any(key.startswith("a_head.") for key in state.keys())

    if has_actor and has_critic:
        model.load_state_dict(state, strict=False)
        return

    if has_v_head and has_a_head:
        mapped: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("a_head."):
                mapped[key.replace("a_head.", "actor_head.")] = value
            elif key.startswith("v_head."):
                mapped[key.replace("v_head.", "critic_head.")] = value
            elif key.startswith("aux_head."):
                continue
            else:
                mapped[key] = value
        model.load_state_dict(mapped, strict=False)
        return

    if any(key.startswith("head.") for key in state.keys()):
        mapped = {}
        for key, value in state.items():
            if key.startswith("head."):
                mapped[key.replace("head.", "actor_head.")] = value
            else:
                mapped[key] = value
        model.load_state_dict(mapped, strict=False)
        return

    model.load_state_dict(state, strict=False)


class PretrainedAgent:
    def __init__(
        self,
        *,
        config_path: str,
        model_path: str,
        section: str = "ppo",
        device: str | None = None,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> None:
        config_path = _resolve_path(config_path)
        model_path = _resolve_path(model_path)

        cfg_root = load_config(config_path)
        cfg = getattr(cfg_root, section)
        game = cfg.game

        device_str = resolve_train_device(device or cfg.device)
        self.device = torch.device(device_str)
        self.sample = sample
        self.temperature = temperature
        self.game_mode = game.game_mode
        self.n_players = game.n_players

        model_cls = import_class(cfg.model_class)
        encoder_cls = import_class(cfg.encoder_class)
        self.model = model_cls(**cfg.model.model_dump()).to(self.device)
        self.encoder = encoder_cls(tile_dim=game.tile_dim)

        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint format: {type(state)}")
        _load_weights_compat(self.model, state)
        self.model.eval()

    @torch.inference_mode()
    def act(self, obs):
        feat = self.encoder.encode(obs).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).to(self.device).bool()

        output = self.model(feat)
        if isinstance(output, tuple):
            values = output[0][0]
        else:
            values = output[0]

        masked_values = values.masked_fill(~mask, -1e9)
        if self.sample:
            temp = max(1e-4, float(self.temperature))
            probs = torch.softmax(masked_values / temp, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())
        else:
            action_idx = int(torch.argmax(masked_values).item())

        action = obs.find_action(action_idx)
        if action is not None:
            return action

        legal = obs.legal_actions()
        if legal:
            return legal[0]
        raise ValueError(f"No legal action for action_id={action_idx}")

    def reset(self) -> None:
        """Stateless inference hook."""


@dataclass
class SessionState:
    game_mode: str = DEFAULT_GAME_MODE
    human_seat: int = HUMAN_SEAT
    env: RiichiEnv | None = None
    obs_dict: dict[int, Any] | None = None
    human_obs: Any = None
    pending_ai_actions: dict[int, Any] = field(default_factory=dict)
    ai_agent: PretrainedAgent | None = None
    player_names: list[str] = field(default_factory=list)
    viewer_revision: int = field(default_factory=lambda: next(VIEWER_REVISION))

    def reset(self) -> None:
        assert self.ai_agent is not None
        self.env = RiichiEnv(game_mode=self.game_mode)
        self.obs_dict = self.env.reset()
        self.human_obs = None
        self.pending_ai_actions = {}
        self.ai_agent.reset()
        self.advance_until_human_turn()
        self.viewer_revision = next(VIEWER_REVISION)

    def advance_until_human_turn(self) -> None:
        assert self.env is not None
        assert self.obs_dict is not None

        obs_dict = self.obs_dict
        while not self.env.done():
            ai_actions: dict[int, Any] = {}
            human_obs = None

            for pid, obs in obs_dict.items():
                legal = obs.legal_actions()
                if not legal:
                    continue
                if pid == self.human_seat:
                    human_obs = obs
                else:
                    ai_actions[pid] = self.ai_agent.act(obs)

            if human_obs is not None:
                self.human_obs = human_obs
                self.pending_ai_actions = ai_actions
                self.obs_dict = obs_dict
                return

            if not ai_actions:
                break

            obs_dict = self.env.step(ai_actions)

        self.human_obs = None
        self.pending_ai_actions = {}
        self.obs_dict = obs_dict

    def play_action(self, action_idx: int) -> None:
        if self.env is None or self.human_obs is None or self.env.done():
            return

        legal = self.human_obs.legal_actions()
        if action_idx < 0 or action_idx >= len(legal):
            return

        actions = dict(self.pending_ai_actions)
        actions[self.human_seat] = legal[action_idx]
        self.obs_dict = self.env.step(actions)
        self.human_obs = None
        self.pending_ai_actions = {}
        self.advance_until_human_turn()
        self.viewer_revision = next(VIEWER_REVISION)

    def play_random(self) -> None:
        if self.human_obs is None:
            return
        legal = self.human_obs.legal_actions()
        if legal:
            self.play_action(secrets.randbelow(len(legal)))


SESSIONS: dict[str, SessionState] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser UI for human-vs-3AI RiichiEnv play")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Model config YAML path")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Checkpoint path")
    parser.add_argument("--section", choices=["ppo", "cql", "bc"], default="ppo", help="Config section to load")
    parser.add_argument("--device", default=None, help="Inference device override")
    parser.add_argument("--sample", action="store_true", help="Sample AI actions instead of greedy argmax")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when --sample is set")
    parser.add_argument("--game-mode", default=None, help="RiichiEnv game_mode override; defaults to config game mode")
    parser.add_argument("--human-seat", type=int, default=HUMAN_SEAT, help="Seat controlled by the browser user")
    return parser.parse_args()


def _redirect(location: str, extra_headers: list[tuple[str, str]] | None = None) -> tuple[str, list[tuple[str, str]], bytes]:
    headers = [("Location", location)]
    if extra_headers:
        headers.extend(extra_headers)
    return "303 See Other", headers, b""


def _json_action(action: Any) -> dict[str, Any]:
    return json.loads(action.to_mjai())


def _format_action(action: Any) -> str:
    ev = _json_action(action)
    action_type = ev.get("type", "")
    label = ACTION_TYPE_LABELS.get(action_type, action_type or "Action")
    parts = [label]

    pai = ev.get("pai")
    if pai is not None:
        parts.append(str(pai))

    consumed = ev.get("consumed")
    if consumed:
        parts.append("(" + ",".join(str(x) for x in consumed) + ")")

    target = ev.get("target")
    if target is not None:
        parts.append(f"<-{target}")

    return " ".join(parts)


def _escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _seat_wind_label(seat: int) -> str:
    return ["East", "South", "West", "North"][seat] if 0 <= seat < 4 else f"Seat {seat}"


def _player_label(session: SessionState, seat: int) -> str:
    base = session.player_names[seat] if 0 <= seat < len(session.player_names) else f"Seat {seat}"
    return f"{base} ({_seat_wind_label(seat)})"


def _round_label(env: RiichiEnv) -> str:
    start_kyoku = None
    for ev in reversed(getattr(env, "mjai_log", [])):
        if ev.get("type") == "start_kyoku":
            start_kyoku = ev
            break
    if start_kyoku is None:
        return "Game start"
    bakaze = start_kyoku.get("bakaze", "?")
    kyoku = start_kyoku.get("kyoku", "?")
    honba = start_kyoku.get("honba", 0)
    kyotaku = start_kyoku.get("kyotaku", 0)
    return f"{bakaze}{kyoku} Honba {honba} Riichi {kyotaku}"


def _tile_badge(tile: str, extra_class: str = "") -> str:
    suit = "honor"
    if tile.endswith("m") or tile.endswith("mr"):
        suit = "man"
    elif tile.endswith("p") or tile.endswith("pr"):
        suit = "pin"
    elif tile.endswith("s") or tile.endswith("sr"):
        suit = "sou"
    return f'<span class="tile {suit} {extra_class}">{_escape(tile)}</span>'


def _render_hand(obs: Any) -> str:
    tiles = tid_to_mjai_list(obs.hand)
    return "".join(_tile_badge(tile) for tile in tiles)


def _render_discards(session: SessionState, obs: Any) -> str:
    blocks = []
    for seat, discards in enumerate(obs.discards):
        tile_html = "".join(_tile_badge(tile, "discard") for tile in tid_to_mjai_list(discards))
        if not tile_html:
            tile_html = '<span class="muted">none</span>'
        blocks.append(
            f"""
            <div class="seat-block">
              <div class="seat-title">{_escape(_player_label(session, seat))}</div>
              <div class="tile-row">{tile_html}</div>
            </div>
            """
        )
    return "".join(blocks)


def _public_event_text(session: SessionState, ev: dict[str, Any]) -> str | None:
    event_type = ev.get("type")
    actor = ev.get("actor")

    if event_type == "start_kyoku":
        return f"{ev.get('bakaze', '?')}{ev.get('kyoku', '?')} starts. Dora {ev.get('dora_marker', '?')}."

    if event_type == "tsumo":
        if actor == session.human_seat:
            return f"{session.player_names[actor]} draws {ev.get('pai', '?')}."
        return f"{session.player_names[actor]} draws a tile."

    if event_type == "dahai":
        if actor is None:
            return None
        return f"{session.player_names[actor]} discards {ev.get('pai', '?')}."

    if event_type in {"chi", "pon", "daiminkan", "ankan", "kakan", "reach", "nukidora"}:
        if actor is None:
            return None
        return f"{session.player_names[actor]}: {_format_public_action(ev)}."

    if event_type == "dora":
        return f"New dora indicator: {ev.get('dora_marker', '?')}."

    if event_type == "hora":
        if actor is None:
            return None
        target = ev.get("target")
        if target == actor:
            return f"{session.player_names[actor]} wins by tsumo."
        if target is not None:
            return f"{session.player_names[actor]} wins off {session.player_names[target]}."
        return f"{session.player_names[actor]} wins."

    if event_type == "ryukyoku":
        return "Round ends in draw."

    return None


def _format_public_action(ev: dict[str, Any]) -> str:
    action_type = ev.get("type", "")
    label = ACTION_TYPE_LABELS.get(action_type, action_type or "Action")
    parts = [label]
    pai = ev.get("pai")
    if pai is not None:
        parts.append(str(pai))
    consumed = ev.get("consumed")
    if consumed:
        parts.append("(" + ",".join(str(x) for x in consumed) + ")")
    target = ev.get("target")
    if target is not None:
        parts.append(f"<-{target}")
    return " ".join(parts)


def _render_recent_events(session: SessionState, limit: int = 12) -> str:
    rows: list[str] = []
    for ev in reversed(getattr(session.env, "mjai_log", [])):
        text = _public_event_text(session, ev)
        if text is None:
            continue
        rows.append(f"<li>{_escape(text)}</li>")
        if len(rows) >= limit:
            break
    rows.reverse()
    return "".join(rows) if rows else "<li><span class=\"muted\">no public events yet</span></li>"


def _render_action_buttons(obs: Any) -> str:
    legal = obs.legal_actions()
    events = [_json_action(action) for action in legal]

    last_event = obs.events[-1] if getattr(obs, "events", None) else None
    last_draw = None
    if isinstance(last_event, dict) and last_event.get("type") == "tsumo" and last_event.get("actor") == obs.player_id:
        last_draw = last_event.get("pai")

    draw_mark_used = False
    discard_buttons = []
    other_buttons = []
    for idx, ev in enumerate(events):
        shortcut = SHORTCUTS[idx] if idx < len(SHORTCUTS) else ""
        action_type = ev.get("type", "")
        prefix = f"[{shortcut}] " if shortcut else ""
        if action_type == "dahai":
            tile = str(ev.get("pai", "?"))
            badge_class = "drawn" if last_draw == tile and not draw_mark_used else ""
            if badge_class:
                draw_mark_used = True
            discard_buttons.append(
                f"""
                <button class="action-button discard-button" type="submit" name="idx" value="{idx}" data-shortcut="{_escape(shortcut)}">
                  <span class="shortcut">{_escape(shortcut)}</span>
                  {_tile_badge(tile, badge_class)}
                </button>
                """
            )
            continue

        label = prefix + _format_action(legal[idx])
        other_buttons.append(
            f"""
            <button class="action-button other-button" type="submit" name="idx" value="{idx}" data-shortcut="{_escape(shortcut)}">
              {_escape(label)}
            </button>
            """
        )

    discard_html = "".join(discard_buttons) if discard_buttons else '<div class="muted">no discard choices</div>'
    other_html = "".join(other_buttons) if other_buttons else '<div class="muted">no call / declaration actions</div>'
    return f"""
    <form method="post" action="/action" id="action-form">
      <div class="section-label">Discard</div>
      <div class="action-grid">{discard_html}</div>
      <div class="section-label">Calls And Special Actions</div>
      <div class="action-grid">{other_html}</div>
    </form>
    """


def _render_status(session: SessionState) -> str:
    assert session.env is not None
    env = session.env
    scores = ", ".join(str(score) for score in env.scores())
    ranks = ", ".join(str(rank) for rank in env.ranks())
    phase = "Game finished." if env.done() else "Your turn." if session.human_obs is not None else "Waiting for next human turn..."
    lineup = ", ".join(_player_label(session, seat) for seat in range(len(session.player_names)))
    return f"""
    <div class="status-grid">
      <div class="card">
        <div class="card-title">Round</div>
        <div class="big">{_escape(_round_label(env))}</div>
        <div class="muted">Seat: {_escape(_player_label(session, session.human_seat))}</div>
      </div>
      <div class="card">
        <div class="card-title">Phase</div>
        <div class="big">{_escape(phase)}</div>
        <div class="muted">Scores: {_escape(scores)}</div>
        <div class="muted">Ranks: {_escape(ranks)}</div>
      </div>
      <div class="card">
        <div class="card-title">Table</div>
        <div class="big">{_escape(session.player_names[session.human_seat])}</div>
        <div class="muted">{_escape(lineup)}</div>
      </div>
    </div>
    """


def _viewer_path(session: SessionState) -> str:
    return f"/viewer?rev={session.viewer_revision}"


def _viewer_log(session: SessionState) -> list[dict[str, Any]]:
    assert session.env is not None
    log = copy.deepcopy(getattr(session.env, "mjai_log", []))
    if log and log[0].get("type") == "start_game":
        log[0]["names"] = list(session.player_names)
    else:
        log.insert(0, {"type": "start_game", "names": list(session.player_names)})
    return log


def _render_viewer_document(session: SessionState) -> str:
    current_step = max(len(getattr(session.env, "mjai_log", [])) - 1, 0)
    viewer_html = GameViewer.from_list(_viewer_log(session)).show(
        step=current_step,
        perspective=session.human_seat,
        freeze=True,
    ).data
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: transparent;
    }}
    /* Hide only opponents' concealed tiles. Public melds stay visible. */
    .opp-tiles-inner .opp-tile,
    .opp-tiles-inner .opp-tile-rotated {{
      position: relative;
      overflow: hidden;
      border-radius: 3px;
    }}
    .opp-tiles-inner .opp-tile > *,
    .opp-tiles-inner .opp-tile-rotated > * {{
      opacity: 0 !important;
    }}
    .opp-tiles-inner .opp-tile::after,
    .opp-tiles-inner .opp-tile-rotated::after {{
      content: "";
      position: absolute;
      inset: 0;
      border-radius: 3px;
      background:
        linear-gradient(135deg, rgba(255,255,255,0.12), transparent 40%),
        linear-gradient(180deg, #c6b790 0%, #ab9460 100%);
      box-shadow:
        inset 0 0 0 1px rgba(78, 58, 26, 0.35),
        inset 0 -2px 0 rgba(110, 86, 41, 0.35);
      z-index: 999;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  {viewer_html}
</body>
</html>
"""


def _render_state_fragments(session: SessionState) -> dict[str, str]:
    hand_html = '<span class="muted">game over</span>'
    discard_panel = '<span class="muted">game over</span>'
    action_html = '<div class="muted">No legal actions available.</div>'
    if session.human_obs is not None:
        hand_html = _render_hand(session.human_obs)
        discard_panel = _render_discards(session, session.human_obs)
        action_html = _render_action_buttons(session.human_obs)

    return {
        "status_html": _render_status(session),
        "hand_html": hand_html,
        "actions_html": action_html,
        "discards_html": discard_panel,
        "events_html": _render_recent_events(session),
        "viewer_url": _viewer_path(session),
    }


def _render_page(session: SessionState) -> str:
    assert session.env is not None
    env = session.env
    fragments = _render_state_fragments(session)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RiichiEnv Browser UI</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --ink: #1f1f1f;
      --muted: #6b665f;
      --panel: #fffaf1;
      --panel-strong: #fffdf8;
      --line: #d5c7b3;
      --accent: #8f2d24;
      --accent-soft: #f3ddd6;
      --green: #1e6a57;
      --man: #3d6ea8;
      --pin: #b5402b;
      --sou: #2d7a4f;
      --honor: #4e4a43;
      --shadow: 0 12px 28px rgba(41, 31, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.85), transparent 28%),
        linear-gradient(135deg, #efe6d6 0%, #f8f3eb 45%, #e7ddcf 100%);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      padding: 22px 24px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.4));
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0;
      font-size: 32px;
      line-height: 1;
      letter-spacing: 0.02em;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .toolbar-form {{
      margin: 0;
    }}
    .toolbar-button {{
      border: 1px solid var(--line);
      background: var(--panel-strong);
      color: var(--ink);
      padding: 11px 16px;
      border-radius: 999px;
      font: inherit;
      cursor: pointer;
      box-shadow: var(--shadow);
    }}
    .toolbar-button.primary {{
      background: var(--accent);
      color: white;
      border-color: transparent;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .card, .panel {{
      border: 1px solid var(--line);
      background: rgba(255, 250, 241, 0.9);
      border-radius: 20px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .card-title, .panel h2 {{
      margin: 0 0 10px;
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .panel h2 {{
      font-size: 17px;
      letter-spacing: 0.06em;
      color: var(--ink);
    }}
    .big {{
      font-size: 26px;
      line-height: 1.2;
      margin-bottom: 6px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 2.1fr) minmax(320px, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .panel.viewer-panel {{
      padding: 14px;
      overflow: hidden;
    }}
    .viewer-frame {{
      width: 100%;
      min-height: 760px;
      border: 0;
      display: block;
      background: transparent;
    }}
    .side-stack {{
      display: grid;
      gap: 18px;
    }}
    .tile-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .tile {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 44px;
      height: 60px;
      padding: 0 10px;
      border-radius: 10px;
      border: 1px solid rgba(0,0,0,0.1);
      background: white;
      box-shadow: inset 0 -3px 0 rgba(0,0,0,0.05);
      font-size: 24px;
      font-weight: 700;
    }}
    .tile.man {{ color: var(--man); }}
    .tile.pin {{ color: var(--pin); }}
    .tile.sou {{ color: var(--sou); }}
    .tile.honor {{ color: var(--honor); }}
    .tile.discard {{
      min-width: 34px;
      height: 46px;
      font-size: 18px;
    }}
    .tile.drawn {{
      outline: 3px solid rgba(143, 45, 36, 0.2);
    }}
    .action-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .section-label {{
      margin: 12px 0 10px;
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .action-button {{
      border: 1px solid var(--line);
      background: var(--panel-strong);
      color: var(--ink);
      border-radius: 14px;
      cursor: pointer;
      box-shadow: var(--shadow);
    }}
    .discard-button {{
      padding: 10px;
      min-width: 74px;
      display: inline-flex;
      flex-direction: column;
      gap: 6px;
      align-items: center;
    }}
    .other-button {{
      padding: 12px 14px;
      font: inherit;
      font-size: 15px;
    }}
    .shortcut {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      min-height: 14px;
    }}
    .seat-block + .seat-block {{
      margin-top: 14px;
    }}
    .seat-title {{
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .muted {{
      color: var(--muted);
    }}
    ul.event-list {{
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 8px;
    }}
    code {{
      word-break: break-word;
      white-space: pre-wrap;
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 12px;
    }}
    .footer-note {{
      color: var(--muted);
      font-size: 14px;
    }}
    .is-busy {{
      pointer-events: none;
      opacity: 0.72;
    }}
    @media (max-width: 1024px) {{
      body {{ padding: 14px; }}
      .layout {{ grid-template-columns: 1fr; }}
      .hero {{
        flex-direction: column;
        align-items: start;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <h1>RiichiEnv Browser Table</h1>
        <p>Human seat {_escape(_seat_wind_label(session.human_seat))}. Click tiles in the browser instead of working through notebook widgets.</p>
      </div>
      <div class="control-row">
        <form class="toolbar-form" method="post" action="/random">
          <button class="toolbar-button" type="submit">Random Move</button>
        </form>
        <form class="toolbar-form" method="post" action="/reset">
          <button class="toolbar-button primary" type="submit">New Game</button>
        </form>
      </div>
    </section>

    <div id="status-panel">{fragments["status_html"]}</div>

    <div class="layout">
      <section class="panel viewer-panel">
        <iframe
          id="viewer-frame"
          class="viewer-frame"
          src="{_escape(_viewer_path(session))}"
          loading="eager"
          referrerpolicy="no-referrer"
        ></iframe>
      </section>

      <aside class="side-stack">
        <section class="panel">
          <h2>Your Hand</h2>
          <div class="tile-row" id="hand-panel">{fragments["hand_html"]}</div>
        </section>

        <section class="panel">
          <h2>Available Actions</h2>
          <div id="actions-panel">{fragments["actions_html"]}</div>
        </section>

        <section class="panel">
          <h2>Discard Pools</h2>
          <div id="discards-panel">{fragments["discards_html"]}</div>
        </section>

        <section class="panel">
          <h2>Recent Events</h2>
          <ul class="event-list" id="events-panel">{fragments["events_html"]}</ul>
        </section>
      </aside>
    </div>

    <div class="footer-note">
      Keyboard shortcuts follow the small labels on each action button. The browser app keeps the board view and controls on the same page.
    </div>
  </div>

  <script>
    let requestInFlight = false;

    const applyState = (payload) => {{
      document.getElementById("status-panel").innerHTML = payload.status_html;
      document.getElementById("hand-panel").innerHTML = payload.hand_html;
      document.getElementById("actions-panel").innerHTML = payload.actions_html;
      document.getElementById("discards-panel").innerHTML = payload.discards_html;
      document.getElementById("events-panel").innerHTML = payload.events_html;
      document.getElementById("viewer-frame").src = payload.viewer_url;
    }};

    const submitAsync = async (form, submitter) => {{
      if (requestInFlight) return;
      requestInFlight = true;
      document.body.classList.add("is-busy");

      try {{
        const formData = new FormData(form);
        if (submitter && submitter.name) {{
          formData.set(submitter.name, submitter.value);
        }}

        const response = await fetch(form.action, {{
          method: form.method || "POST",
          body: new URLSearchParams(formData),
          headers: {{
            "X-Requested-With": "fetch",
            "Accept": "application/json"
          }}
        }});

        if (!response.ok) {{
          throw new Error(`HTTP ${{response.status}}`);
        }}

        const payload = await response.json();
        applyState(payload);
      }} catch (error) {{
        console.error("Async update failed:", error);
        window.location.reload();
      }} finally {{
        requestInFlight = false;
        document.body.classList.remove("is-busy");
      }}
    }};

    document.addEventListener("submit", (event) => {{
      const form = event.target;
      if (!(form instanceof HTMLFormElement)) return;
      if ((form.method || "").toLowerCase() !== "post") return;
      event.preventDefault();
      submitAsync(form, event.submitter || null);
    }});

    window.addEventListener("keydown", (event) => {{
      if (requestInFlight) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const key = event.key.toLowerCase();
      const button = document.querySelector(`[data-shortcut="${{CSS.escape(key)}}"]`);
      if (!button) return;
      event.preventDefault();
      button.click();
    }});
  </script>
</body>
</html>
"""


def _get_or_create_session(
    environ: dict[str, Any],
    default_game_mode: str,
    default_human_seat: int,
    ai_agent: PretrainedAgent,
    player_names: list[str],
) -> tuple[str, SessionState, list[tuple[str, str]]]:
    raw_cookie = environ.get("HTTP_COOKIE", "")
    jar = cookies.SimpleCookie()
    if raw_cookie:
        jar.load(raw_cookie)

    session_id = jar[SESSION_COOKIE].value if SESSION_COOKIE in jar else None
    headers: list[tuple[str, str]] = []
    if session_id is None or session_id not in SESSIONS:
        session_id = secrets.token_hex(16)
        session = SessionState(
            game_mode=default_game_mode,
            human_seat=default_human_seat,
            ai_agent=ai_agent,
            player_names=list(player_names),
        )
        session.reset()
        SESSIONS[session_id] = session
        headers.append(("Set-Cookie", f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax"))
    return session_id, SESSIONS[session_id], headers


def _read_post_body(environ: dict[str, Any]) -> dict[str, list[str]]:
    try:
        content_length = int(environ.get("CONTENT_LENGTH") or "0")
    except ValueError:
        content_length = 0
    raw = environ["wsgi.input"].read(content_length) if content_length > 0 else b""
    return parse_qs(raw.decode("utf-8"))


def _wants_json(environ: dict[str, Any]) -> bool:
    requested_with = environ.get("HTTP_X_REQUESTED_WITH", "")
    accept = environ.get("HTTP_ACCEPT", "")
    return requested_with.lower() == "fetch" or "application/json" in accept.lower()


def _json_response(start_response, payload: dict[str, str], extra_headers: list[tuple[str, str]]) -> list[bytes]:
    body = json.dumps(payload).encode("utf-8")
    headers = [("Content-Type", "application/json; charset=utf-8"), *extra_headers]
    start_response("200 OK", headers)
    return [body]


def build_app(default_game_mode: str, default_human_seat: int, ai_agent: PretrainedAgent, player_names: list[str]):
    def app(environ: dict[str, Any], start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")

        _, session, session_headers = _get_or_create_session(
            environ,
            default_game_mode,
            default_human_seat,
            ai_agent,
            player_names,
        )

        if method == "POST" and path == "/reset":
            session.reset()
            if _wants_json(environ):
                return _json_response(start_response, _render_state_fragments(session), session_headers)
            status, headers, body = _redirect("/", session_headers)
            start_response(status, headers)
            return [body]

        if method == "POST" and path == "/random":
            session.play_random()
            if _wants_json(environ):
                return _json_response(start_response, _render_state_fragments(session), session_headers)
            status, headers, body = _redirect("/", session_headers)
            start_response(status, headers)
            return [body]

        if method == "POST" and path == "/action":
            form = _read_post_body(environ)
            try:
                action_idx = int((form.get("idx") or ["-1"])[0])
            except ValueError:
                action_idx = -1
            session.play_action(action_idx)
            if _wants_json(environ):
                return _json_response(start_response, _render_state_fragments(session), session_headers)
            status, headers, body = _redirect("/", session_headers)
            start_response(status, headers)
            return [body]

        if method == "GET" and path == "/viewer":
            page = _render_viewer_document(session).encode("utf-8")
            headers = [("Content-Type", "text/html; charset=utf-8"), *session_headers]
            start_response("200 OK", headers)
            return [page]

        if path != "/":
            start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8"), *session_headers])
            return [b"Not Found"]

        page = _render_page(session).encode("utf-8")
        headers = [("Content-Type", "text/html; charset=utf-8"), *session_headers]
        start_response("200 OK", headers)
        return [page]

    return app


def main() -> None:
    args = parse_args()
    model_path = _resolve_path(args.model)
    ai_agent = PretrainedAgent(
        config_path=args.config,
        model_path=model_path,
        section=args.section,
        device=args.device,
        sample=args.sample,
        temperature=args.temperature,
    )
    game_mode = args.game_mode or ai_agent.game_mode
    if not (0 <= args.human_seat < ai_agent.n_players):
        raise ValueError(f"--human-seat must be in [0, {ai_agent.n_players - 1}]")
    player_names = _build_player_names(args.human_seat, ai_agent.n_players, model_path)
    app = build_app(game_mode, args.human_seat, ai_agent, player_names)

    print(
        f"Serving RiichiEnv browser UI on http://{args.host}:{args.port} "
        f"using {args.model} ({args.section}) for AI opponents"
    )
    with make_server(args.host, args.port, app) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
