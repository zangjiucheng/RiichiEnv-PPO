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


def _early_repo_root_override() -> Path | None:
    """Peek at AKAGI_BOT_CONFIG for a `repo_root` override before any
    riichienv_ml import happens (see REPO_ROOT below for why this must
    run before sys.path is touched).
    """
    cfg_path = os.environ.get("AKAGI_BOT_CONFIG")
    if not cfg_path or not Path(cfg_path).is_file():
        return None
    try:
        with open(cfg_path) as f:
            data = json.load(f)
    except Exception:
        return None
    root = data.get("repo_root")
    if not root:
        return None
    return Path(root).expanduser().resolve()


# bot.py normally lives at <repo-root>/riichienv-ml/scripts/akagi_bot/bot.py
# (whether reached directly or through a symlink under Akagi/mjai_bot/<name>/
# -- .resolve() follows the symlink to this physical location either way),
# which is 3 directories below <repo-root>. That fixed-depth assumption
# breaks once this file is bundled inside a packaged Akagi.app (nested under
# Contents/MacOS/mjai_bot/<name>/ instead), so an explicit `repo_root`
# setting takes priority when present.
REPO_ROOT = _early_repo_root_override() or Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "riichienv-ml" / "src"))

from riichienv import RiichiEnv, MeldType  # noqa: E402
from riichienv.convert import tid_to_mjai  # noqa: E402
from riichienv_ml.config import GAME_PARAMS, import_class, load_config  # noqa: E402
from riichienv_ml.trainers.kyoku_reward import compute_kyoku_gae, compute_kyoku_rewards  # noqa: E402

# mjai event types that represent an actual decision our bot made (as
# opposed to passive/informational events like tsumo or start_kyoku that
# happen to carry `actor == our seat` without us having decided anything).
# Used by the online-learning divergence guard below.
_DECISION_TYPES = frozenset({
    "dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "reach", "hora", "kita",
})

DEFAULT_SETTINGS = {
    "repo_root": "",
    "config_path_4p": "riichienv-ml/src/riichienv_ml/configs/4p/ppo_v2.yml",
    "model_path_4p": "artifacts/4p/ppo_v2/checkpoints",
    "config_path_3p": "riichienv-ml/src/riichienv_ml/configs/3p/ppo.yml",
    "model_path_3p": "artifacts/3p/ppo/checkpoints",
    "device": "cpu",
    "sample": False,
    "temperature": 1.0,
    # --- Online learning (beta) -- see README's "Online learning" section
    # before enabling. Off by default: this fine-tunes the live model on
    # your own games, which only produces a correct training signal when
    # every hero decision it logs is actually what happened in the real
    # game (see _Session's divergence guard) -- i.e. it's only meaningful
    # with autoplay on, or a human who always follows the bot's suggestion.
    "online_learning_enabled": False,
    "online_learning_lr": 1e-6,
    "online_learning_alpha_kl": 0.2,
}

# Number of hanchan to accumulate before triggering one online PPO
# mini-update. Not exposed as a setting: fewer hanchan means noisier,
# higher-variance updates; more means slower adaptation. Fixed rather than
# configurable to keep the beta's behavior predictable.
ONLINE_LEARNING_HANCHAN_PER_UPDATE = 4


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
                 device_str: str, sample: bool, temperature: float,
                 n_players: int = 4):
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
        self.loaded_checkpoint_path = resolved

        # Online-learning bookkeeping (see README "Online learning"). Cheap
        # to keep on hand even when the feature is disabled -- just config
        # values, no extra compute happens unless _Session actually stores
        # act()'s returned extras.
        self.config_path = config_path
        self.model_class = cfg.model_class
        self.model_config = cfg.model.model_dump()
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.grp_model = cfg.grp_model
        self.pts_weight = cfg.pts_weight
        self.n_players = n_players

        notify("success", "Model loaded", f"{resolved.name} ({cfg.model_class.rsplit('.', 1)[-1]})")

    @torch.inference_mode()
    def act(self, obs: Any):
        """Pick an action. Returns (action, extra) where `extra` is a dict
        of the same forward pass's cheap side-products (features/mask/
        action_idx/log_prob/value) used for online learning -- computing
        these costs one extra log_softmax+gather over an already-computed
        logits tensor, negligible next to the forward pass itself.
        """
        feat = self.encoder.encode(obs).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).to(self.device).bool()

        output = self.model(feat)
        logits, value = output if isinstance(output, tuple) else (output, None)
        masked = logits.masked_fill(~mask, -1e9)

        if self.sample:
            temp = max(1e-4, float(self.temperature))
            probs = torch.softmax(masked / temp, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())
        else:
            action_idx = int(torch.argmax(masked, dim=-1).item())

        log_prob = float(torch.log_softmax(masked, dim=-1)[0, action_idx].item())
        extra = {
            "features": feat[0].cpu().numpy(),
            "mask": mask.cpu().numpy().astype(np.uint8),
            "action_idx": action_idx,
            "log_prob": log_prob,
            "value": float(value[0].item()) if value is not None else 0.0,
        }

        action = obs.find_action(action_idx)
        if action is not None:
            return action, extra
        legal = obs.legal_actions()
        return (legal[0] if legal else None), extra


def _live_checkpoint_dir(model_dir: Path) -> Path:
    """Where online-learning updates are written, kept separate from
    `model_dir` (which may be a shared/cluster-written directory) so a live
    fine-tune lineage never collides with or gets mixed into cluster
    training output. Rolling back is just pointing model_path_* elsewhere,
    or deleting this directory.
    """
    return model_dir.parent / f"{model_dir.name}_live"


def _build_agent(n_players: int, settings: dict) -> _Agent:
    suffix = "4p" if n_players == 4 else "3p"
    params = GAME_PARAMS[n_players]
    config_path = _resolve(settings[f"config_path_{suffix}"])
    model_path = _resolve(settings[f"model_path_{suffix}"])

    if settings.get("online_learning_enabled") and model_path.is_dir():
        live_dir = _live_checkpoint_dir(model_path)
        if live_dir.is_dir() and any(live_dir.glob("model_*.pth")):
            model_path = live_dir

    return _Agent(
        config_path, model_path, params["tile_dim"],
        settings["device"], bool(settings["sample"]), float(settings["temperature"]),
        n_players=n_players,
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


def _resolve_anchor(live_dir: Path, base_checkpoint_path: Path) -> Path:
    """The fixed KL-regularization reference for a live-learning lineage:
    pinned to whatever checkpoint the FIRST online update in this lineage
    started from, read from `anchor.json` if it already exists. Kept fixed
    across the whole lineage (rather than re-resolved each time) so
    repeated online updates can't drift arbitrarily far from the
    cluster-trained baseline -- same role `teacher_model` plays in normal
    PPO training.
    """
    anchor_file = live_dir / "anchor.json"
    if anchor_file.is_file():
        try:
            data = json.loads(anchor_file.read_text())
            path = Path(data["base_checkpoint_path"])
            if path.is_file():
                return path
        except Exception:
            pass
    live_dir.mkdir(parents=True, exist_ok=True)
    anchor_file.write_text(json.dumps({"base_checkpoint_path": str(base_checkpoint_path)}))
    return base_checkpoint_path


def _write_online_learning_meta(live_dir: Path, agent: "_Agent", anchor_path: Path) -> None:
    meta = {
        "config_path": str(agent.config_path),
        "model_class": agent.model_class,
        "model_config": agent.model_config,
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "n_players": agent.n_players,
        "base_checkpoint_path": str(anchor_path),
        "lr": float(agent.settings_lr),
        "alpha_kl": float(agent.settings_alpha_kl),
        "device": str(agent.device),
    }
    (live_dir / "meta.json").write_text(json.dumps(meta))


def _spawn_online_update(live_dir: Path) -> None:
    """Fire-and-forget a detached process to run the PPO mini-update.
    Must not block: Akagi kills this bot process ~500ms after it sees the
    end_game reaction (see Akagi's `runner.rs::reset()` RESET_GRACE_MS), far
    too short a window to run a training step in-process.
    """
    lock_path = live_dir / ".update.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        # An update is already in flight (or a stale lock from a crashed
        # run) -- skip this round rather than risk two concurrent updates
        # stepping on the same checkpoint lineage. Picked up again once
        # enough pending hanchan accumulate past the next update.
        return

    script = Path(__file__).resolve().parent / "online_update.py"
    log_path = live_dir / "update.log"
    import subprocess
    with open(log_path, "a") as log_f:
        subprocess.Popen(
            [sys.executable, str(script), "--live_dir", str(live_dir)],
            stdin=subprocess.DEVNULL, stdout=log_f, stderr=log_f,
            start_new_session=True,
        )
    notify("info", "Online learning: update started",
           f"{ONLINE_LEARNING_HANCHAN_PER_UPDATE} hanchan accumulated; fine-tuning in the background.")


class _Session:
    """Holds the one RiichiEnv + Agent for this process's single game."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.seat = _initial_seat()
        self.env: RiichiEnv | None = None
        self.agent: _Agent | None = None
        self._reset_online_learning_state()

    def _reset_online_learning_state(self) -> None:
        self.ol_enabled = False
        self.ol_live_dir: Path | None = None
        self.ol_reward_predictor = None
        self.ol_kyoku_buffer: list[dict] = []
        self.ol_completed_kyokus: list[tuple[list[dict], float]] = []
        self.ol_prev_kyoku_idx = None
        self.ol_kyoku_start_scores = None
        self.ol_kyoku_start_meta = None
        self.ol_pending_prediction: dict | None = None
        self.ol_diverged = False

    def _init_online_learning(self) -> None:
        """Called once per hanchan, right after self.agent is built."""
        if not self.settings.get("online_learning_enabled") or self.agent is None:
            return
        model_path = _resolve(
            self.settings[f"model_path_{'4p' if self.agent.n_players == 4 else '3p'}"])
        if not model_path.is_dir():
            notify("warn", "Online learning disabled this game",
                   "model_path must be a directory (auto-pick highest checkpoint) "
                   "for online learning to have somewhere to write live checkpoints.")
            return
        live_dir = _live_checkpoint_dir(model_path)
        # The anchor is the checkpoint this lineage's FIRST update started
        # from -- if we just loaded from the base dir (no live checkpoints
        # yet), that base checkpoint IS the anchor; otherwise reuse the
        # pinned one from a prior hanchan in this lineage.
        base_for_anchor = (self.agent.loaded_checkpoint_path
                            if not str(self.agent.loaded_checkpoint_path).startswith(str(live_dir))
                            else _resolve_checkpoint(model_path))
        anchor_path = _resolve_anchor(live_dir, base_for_anchor)

        try:
            from riichienv_ml.models.grp_model import RewardPredictor
            grp_path = _resolve(self.agent.grp_model) if self.agent.grp_model else None
            reward_predictor = (
                RewardPredictor(str(grp_path), self.agent.pts_weight or [10.0, 4.0, -4.0, -10.0],
                                 n_players=self.agent.n_players, device="cpu")
                if grp_path and grp_path.is_file() else None
            )
        except Exception as e:
            notify("warn", "Online learning: GRP reward model unavailable",
                   f"kyoku rewards will be 0 for this game ({e})")
            reward_predictor = None

        self.agent.settings_lr = self.settings.get("online_learning_lr", 1e-6)
        self.agent.settings_alpha_kl = self.settings.get("online_learning_alpha_kl", 0.2)
        _write_online_learning_meta(live_dir, self.agent, anchor_path)

        self.ol_enabled = True
        self.ol_live_dir = live_dir
        self.ol_reward_predictor = reward_predictor
        self.ol_prev_kyoku_idx = self.env.kyoku_idx
        self.ol_kyoku_start_scores = list(self.env.scores())
        self.ol_kyoku_start_meta = (self.env.round_wind, self.env.oya,
                                     self.env.honba, self.env.riichi_sticks)

    def _check_divergence(self, events: list[dict]) -> None:
        """If our last decision predicted X but the real game shows our
        seat actually did Y, this hanchan's trajectory no longer reflects
        what the policy actually caused (e.g. autoplay missed a click, or
        a human overrode the suggestion) -- the reward signal downstream
        would be attributed to the wrong action. Mark the whole hanchan
        invalid rather than try to untangle a partially-correct buffer.
        """
        if not self.ol_enabled or self.ol_pending_prediction is None:
            return
        for ev in events:
            if not isinstance(ev, dict) or ev.get("type") not in _DECISION_TYPES:
                continue
            if ev.get("actor") != self.seat:
                continue
            # `pai` is only compared when both sides actually specify one --
            # e.g. the engine's own mjai_log `hora` events carry no `pai`
            # field at all (just deltas/target/ura_markers), so requiring
            # an exact match there would always spuriously "diverge".
            matches = ev.get("type") == self.ol_pending_prediction.get("type")
            pred_pai = self.ol_pending_prediction.get("pai")
            actual_pai = ev.get("pai")
            if matches and pred_pai is not None and actual_pai is not None:
                matches = pred_pai == actual_pai
            self.ol_pending_prediction = None
            if not matches:
                self.ol_diverged = True
                notify("warn", "Online learning: this hanchan's data discarded",
                       "predicted action didn't match what actually happened in-game "
                       "(autoplay miss or manual override) -- likely if autoplay is off.")
            return

    def _record_decision(self, extra: dict, raw: dict) -> None:
        if not self.ol_enabled or self.ol_diverged:
            return
        self.ol_kyoku_buffer.append({
            "features": extra["features"], "mask": extra["mask"],
            "action": extra["action_idx"], "log_prob": extra["log_prob"],
            "value": extra["value"],
        })
        if raw.get("type") != "none":
            self.ol_pending_prediction = {"type": raw.get("type"), "pai": raw.get("pai")}

    def _maybe_close_kyoku(self, force: bool = False) -> None:
        if not self.ol_enabled or self.ol_diverged or self.env is None:
            return
        cur_kyoku_idx = self.env.kyoku_idx
        boundary = cur_kyoku_idx != self.ol_prev_kyoku_idx
        finished = self.env.done()
        if not boundary and not finished and not force:
            return
        if not self.ol_kyoku_buffer:
            if boundary:
                self.ol_prev_kyoku_idx = cur_kyoku_idx
                self.ol_kyoku_start_scores = list(self.env.scores())
                self.ol_kyoku_start_meta = (self.env.round_wind, self.env.oya,
                                             self.env.honba, self.env.riichi_sticks)
            return
        cur_scores = list(self.env.scores())
        rw, oya, honba, rsticks = self.ol_kyoku_start_meta
        rewards = compute_kyoku_rewards(
            self.ol_reward_predictor, self.ol_kyoku_start_scores, cur_scores,
            rw, oya, honba, rsticks, self.agent.n_players)
        reward = rewards[self.seat]
        self.ol_completed_kyokus.append((self.ol_kyoku_buffer, reward))
        self.ol_kyoku_buffer = []
        if boundary:
            self.ol_prev_kyoku_idx = cur_kyoku_idx
            self.ol_kyoku_start_scores = cur_scores
            self.ol_kyoku_start_meta = (self.env.round_wind, self.env.oya,
                                         self.env.honba, self.env.riichi_sticks)

    def _finalize_hanchan(self) -> None:
        if not self.ol_enabled:
            return
        try:
            self._maybe_close_kyoku(force=True)
            if self.ol_diverged or not self.ol_completed_kyokus:
                return

            feat_list, mask_list, action_list = [], [], []
            log_prob_list, advantage_list, return_list = [], [], []
            for traj, kyoku_reward in self.ol_completed_kyokus:
                if not traj:
                    continue
                advantages, returns = compute_kyoku_gae(
                    traj, kyoku_reward, self.agent.gamma, self.agent.gae_lambda)
                for t, step in enumerate(traj):
                    feat_list.append(step["features"])
                    mask_list.append(step["mask"])
                    action_list.append(step["action"])
                    log_prob_list.append(step["log_prob"])
                    advantage_list.append(advantages[t])
                    return_list.append(returns[t])
            if not feat_list:
                return

            self.ol_live_dir.mkdir(parents=True, exist_ok=True)
            pending_dir = self.ol_live_dir / "pending"
            pending_dir.mkdir(parents=True, exist_ok=True)
            batch_path = pending_dir / f"hanchan_{os.getpid()}_{id(self)}.npz"
            np.savez(
                batch_path,
                features=np.stack(feat_list), mask=np.stack(mask_list),
                action=np.array(action_list, dtype=np.int64),
                log_prob=np.array(log_prob_list, dtype=np.float32),
                advantage=np.array(advantage_list, dtype=np.float32),
                return_=np.array(return_list, dtype=np.float32),
            )

            n_pending = len(list(pending_dir.glob("hanchan_*.npz")))
            if n_pending >= ONLINE_LEARNING_HANCHAN_PER_UPDATE:
                _spawn_online_update(self.ol_live_dir)
        except Exception as e:  # never let online-learning bookkeeping break the bridge
            sys.stderr.write(f"online learning error: {e}\n")
            notify("warn", "Online learning error", str(e))

    def react(self, events: list[dict]) -> dict:
        if not events:
            return {"type": "none"}

        self._check_divergence(events)

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
                self._reset_online_learning_state()
                self._init_online_learning()
                notify("info", "Game started", f"seat={self.seat} players={n_players}")

            if self.env is None:
                continue
            obs = self.env.observe_event(_for_engine(ev), self.seat)
            if obs is not None:
                last_obs = obs

        self._maybe_close_kyoku()
        if any(isinstance(ev, dict) and ev.get("type") == "end_game" for ev in events):
            self._finalize_hanchan()

        if last_obs is None or self.agent is None:
            return {"type": "none"}

        action, extra = self.agent.act(last_obs)
        if action is None:
            return {"type": "none"}

        raw = json.loads(action.to_mjai())
        raw = _enrich(action, raw, self.seat, trigger, self.env)
        self._record_decision(extra, raw)
        return raw


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
