# riichienv-ppo Akagi bot

Bridges a trained RiichiEnv-PPO checkpoint into [Akagi](https://github.com/shinkuan/Akagi)
so it can play live on Tenhou / Majsoul / Riichi City / Amatsuki through Akagi's
mjai bot protocol (`Akagi/mjai_bot/README.md`).

This folder is the source of truth. Our [Akagi fork](https://github.com/zangjiucheng/Akagi)
carries a snapshot copy committed at `mjai_bot/riichienv_ppo/` so the bot
works out of a plain clone or a packaged standalone build; keep the two in
sync by hand after editing here. When developing against RiichiEnv-PPO as
the superproject (Akagi checked out as its `Akagi` submodule), symlink
instead of copying so edits stay live:

```sh
rm -rf Akagi/mjai_bot/riichienv_ppo
ln -s ../../riichienv-ml/scripts/akagi_bot Akagi/mjai_bot/riichienv_ppo
```

Then in Akagi's Bots tab: Refresh -> Install environment (runs `uv sync`
against `pyproject.toml` here) -> activate for 4p and/or 3p.

### Running from a packaged/standalone Akagi build

A packaged build (see `scripts/package-zip.sh` in the Akagi repo) can't
locate the RiichiEnv-PPO checkout by relative position the way the symlinked
dev setup can. Set the `repo_root` setting (Akagi's Bots tab, or directly in
this bot's `settings.toml`) to the checkout's absolute path -- everything
else (`config_path_4p`, `model_path_4p`, ...) then resolves relative to that,
same as in dev mode.

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

## Online learning (beta)

Off by default (`online_learning_enabled` in Akagi's Bots tab). When on,
the bridge fine-tunes the model on your own live games instead of just
running a fixed checkpoint:

- **Mechanism**: each hero decision's encoded observation, chosen action,
  log-prob and value estimate are cached in memory for the hanchan. At
  `end_game`, per-kyoku rewards are computed with the same GRP reward model
  used in cluster training (`compute_kyoku_rewards`) and turned into
  advantages/returns via the same per-kyoku GAE math
  (`riichienv_ml.trainers.kyoku_reward`) -- i.e. this replays the exact
  transition-processing logic `PPOWorker.collect_episodes()` uses, just
  sourced from one real game instead of a simulated rollout batch.
- **Update cadence**: every 4 hanchan (fixed, not configurable -- see
  `ONLINE_LEARNING_HANCHAN_PER_UPDATE` in `bot.py`), a detached
  `online_update.py` process runs one small PPO step (1 epoch, tight clip)
  and writes a new checkpoint to `<model_path>_live/model_N.pth`. It has to
  be a separate detached process: Akagi force-kills the bot subprocess
  ~500ms after sending `end_game` (`runner.rs::reset()`'s `RESET_GRACE_MS`),
  far too short a window to run a training step in-process.
- **Stability guardrail**: every update is KL-regularized toward the
  checkpoint the live lineage *started* from (pinned once in
  `<live_dir>/anchor.json`, not re-picked each time), so repeated updates
  can't drift arbitrarily far from the cluster-trained baseline. Rolling
  back is just pointing `model_path_*` elsewhere or deleting
  `<model_path>_live/`; nothing under `<model_path>_live/` is ever mixed
  into the original checkpoint directory.
- **Correctness requirement -- read this before enabling**: the reward
  signal is only valid if what got logged as "the action the policy took"
  is what *actually happened* in the real game. A per-hanchan divergence
  guard checks this (comparing each predicted decision's type/tile against
  the real mjai event that follows) and discards the whole hanchan's data
  if they don't match -- but if autoplay is off and a human frequently
  overrides the bot's suggestions, most/all hanchans will get discarded
  this way, so **online learning is only really useful with autoplay on**
  (see the autoplay section above, including its own ToS/ban-risk caveat --
  that risk applies here too, plus this feature's own risk that automated
  fine-tuning on live data doesn't guarantee monotonic improvement).
- **Tuning knobs**: `online_learning_lr` (default `1e-6`, deliberately tiny)
  and `online_learning_alpha_kl` (default `0.2`) are exposed as advanced
  settings; everything else (`ppo_clip`, `ppo_epochs`, `entropy_coef`,
  batch size) is fixed in `online_update.py` rather than exposed, to keep
  the settings surface small while this is still experimental.

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
