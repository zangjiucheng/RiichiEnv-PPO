"""Scan replay files and quarantine desync-prone files.

Example:
  python riichienv-ml/scripts/quarantine_bad_replays.py \
    --glob "data/mjsoul/mjsoul-4p/train/**/*.jsonl" \
    --players 4 \
    --mode steps
"""

from __future__ import annotations

import argparse
import glob

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from riichienv_ml.datasets.mjai_logs import (
    MCDataset,
    _is_replay_desync_error,
    load_mjai_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarantine replay files that desync in riichienv parser")
    parser.add_argument("--glob", required=True, help="Replay glob, e.g. data/mjsoul/mjsoul-4p/train/**/*.jsonl")
    parser.add_argument("--players", type=int, choices=[3, 4], default=4, help="Number of players")
    parser.add_argument("--rule", type=str, default="mjsoul", help="Replay rule hint")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["steps", "parse"],
        default="steps",
        help=(
            "Validation depth: "
            "'steps' matches training path (kyoku.steps + action/mask), "
            "'parse' only checks take_kyokus()"
        ),
    )
    return parser.parse_args()


def _exercise_steps(replay, n_players: int) -> None:
    """Run the same replay path used by training data pipelines."""
    for kyoku in replay.take_kyokus():
        for player_id in range(n_players):
            for obs, action in kyoku.steps(player_id):
                _ = action.encode()
                _ = obs.mask()


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {args.glob}")

    ds = MCDataset(
        data_sources=[],
        reward_predictor=None,
        n_players=args.players,
        replay_rule=args.rule,
        is_train=False,
        encoder=None,
    )

    seen = 0
    skipped_known_bad = 0
    quarantined = 0
    ok = 0
    other_errors = 0

    for file_path in tqdm(files, desc="scan", unit="file", dynamic_ncols=True):
        seen += 1
        if ds._is_known_bad_replay(file_path):
            skipped_known_bad += 1
            continue
        try:
            replay = load_mjai_replay(file_path, args.rule, n_players=args.players)
            if args.mode == "parse":
                # Shallow parse: only validates kyoku extraction.
                list(replay.take_kyokus())
            else:
                # Deep validation: matches training code path.
                _exercise_steps(replay, args.players)
            ok += 1
        except Exception as e:  # noqa: BLE001
            if _is_replay_desync_error(e):
                ds._mark_bad_replay(file_path, e)
                quarantined += 1
            else:
                other_errors += 1
                print(f"[non-desync] {file_path}: {e}")

    print(
        "Done. "
        f"seen={seen} ok={ok} quarantined={quarantined} "
        f"known_bad_skipped={skipped_known_bad} other_errors={other_errors}"
    )
    print(f"Bad replay markers: {ds.bad_replay_dir}")


if __name__ == "__main__":
    main()
