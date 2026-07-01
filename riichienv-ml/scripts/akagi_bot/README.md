# riichienv-ppo Akagi bot

Bridges a trained RiichiEnv-PPO checkpoint into [Akagi](https://github.com/shinkuan/Akagi)
so it can play live on Tenhou / Majsoul / Riichi City / Amatsuki through Akagi's
mjai bot protocol (`Akagi/mjai_bot/README.md`).

This folder is the source of truth; it isn't committed inside the `Akagi`
submodule (that's a separate upstream repo). Symlink it in instead:

```sh
mkdir -p Akagi/mjai_bot
ln -s ../../riichienv-ml/scripts/akagi_bot Akagi/mjai_bot/riichienv_ppo
```

Then in Akagi's Bots tab: Refresh -> Install environment (runs `uv sync`
against `pyproject.toml` here) -> activate for 4p and/or 3p.

## Configuring which model it plays

Akagi's Bots tab exposes the settings from `manifest.toml`:

- `config_path_4p` / `config_path_3p` -- which training YAML to read
  `model_class` / `encoder_class` / model dimensions from (e.g.
  `riichienv-ml/src/riichienv_ml/configs/4p/ppo_v2.yml`).
- `model_path_4p` / `model_path_3p` -- a specific `.pth`, or a directory (the
  highest-numbered `model_*.pth` in it is picked automatically, so pointing
  at a live `checkpoints/` dir keeps using the latest snapshot).
- `device` -- `cpu` or `cuda`.
- `sample` / `temperature` -- greedy argmax (default, recommended for live
  play) vs. temperature sampling.

Paths are relative to the RiichiEnv-PPO repo root unless absolute.

## What the bridge does

`bot.py` maintains one `riichienv.RiichiEnv` per game, fed via
`env.observe_event(event, seat)` (the same online-inference API used by
`human_vs_ai_web.py`). Whenever that returns an Observation, it runs the
configured checkpoint through the matching encoder, masks illegal actions,
and converts the chosen `Action` to mjai JSON via `Action.to_mjai()`.

`to_mjai()` doesn't emit every field Akagi's schema requires, so `_enrich()`
fills in the rest from context: `dahai.tsumogiri` (compares against the
triggering tsumo's tile), `chi`/`pon`/`daiminkan.target` (the discarder from
the triggering event), `hora.target` (self for tsumo, the discarder/kakan
actor for ron), and `kakan.consumed` (looked up from the existing Pon meld
being upgraded, via `env.melds`).

## Known limitations

- Only tested against replayed mjai logs, not a live Tenhou connection --
  verify a few real games before trusting it unattended.
- RiichiEnv's rule enforcement (kuitan, red fives, yaku, etc.) must match
  whatever ruleset the live table actually uses, or `legal_actions()` can
  disagree with what the server accepts. Configure `game_mode` (via the
  training YAML's `game.n_players`) accordingly.
- Model load happens on the first `start_game` batch, inside that reaction's
  ~5s budget. A very large checkpoint may need trimming `num_blocks`/
  `d_model` or moving to `device = "cuda"` if it doesn't load in time.
- If `observe_event()` raises mid-batch (e.g. a genuine desync), the bridge
  logs to stderr, toasts an error, and replies `{"type":"none"}` rather than
  crashing -- but internal state may drift out of sync with the real game
  for the rest of that hand.
