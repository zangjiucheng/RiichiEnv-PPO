"""Import MJAI .mjson files from zip archives into local training folders.

Example:
    python riichienv-ml/scripts/import_mjai_zip.py --players 4p ~/Downloads/2024.zip
    python riichienv-ml/scripts/import_mjai_zip.py --players 4p --val-ratio 0.05 ~/Downloads/2023.zip ~/Downloads/2024.zip
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MJAI zip files into data/mjsoul")
    parser.add_argument("zip_files", nargs="+", help="Path(s) to zip file(s) containing .mjson files")
    parser.add_argument(
        "--players",
        choices=["3p", "4p"],
        default="4p",
        help="Target dataset bucket: mjsoul-3p or mjsoul-4p",
    )
    parser.add_argument(
        "--output-root",
        default="data/mjsoul",
        help="Root output directory (default: data/mjsoul)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio in [0, 1], deterministic by file name (default: 0.05)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .jsonl.gz files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned outputs without writing files",
    )
    return parser.parse_args()


def is_val_split(rel_name: str, ratio: float) -> bool:
    # Deterministic split per replay file path.
    digest = hashlib.sha1(rel_name.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False) / 2**32
    return value < ratio


def normalize_member_path(member_name: str) -> PurePosixPath | None:
    p = PurePosixPath(member_name)
    if p.name == "":
        return None
    clean_parts = [x for x in p.parts if x not in ("", ".")]
    if any(x == ".." for x in clean_parts):
        return None
    if not clean_parts:
        return None
    return PurePosixPath(*clean_parts)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.val_ratio <= 1.0):
        raise ValueError("--val-ratio must be between 0 and 1")

    dataset_root = Path(args.output_root) / f"mjsoul-{args.players}"
    train_root = dataset_root / "train"
    val_root = dataset_root / "val"

    if not args.dry_run:
        train_root.mkdir(parents=True, exist_ok=True)
        val_root.mkdir(parents=True, exist_ok=True)

    total_seen = 0
    total_written = 0
    total_skipped = 0
    total_invalid = 0

    for zip_path_str in args.zip_files:
        zip_path = Path(zip_path_str)
        if not zip_path.is_file():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        written = 0
        skipped = 0
        invalid = 0
        seen = 0

        with ZipFile(zip_path) as zf:
            members = [i for i in zf.infolist() if not i.is_dir() and i.filename.endswith(".mjson")]
            for info in tqdm(
                members,
                desc=f"{zip_path.name}",
                unit="file",
                dynamic_ncols=True,
            ):
                seen += 1
                normalized = normalize_member_path(info.filename)
                if normalized is None:
                    invalid += 1
                    continue

                rel_out = Path(*normalized.parts).with_suffix(".jsonl.gz")
                split_root = val_root if is_val_split(normalized.as_posix(), args.val_ratio) else train_root
                out_path = split_root / rel_out

                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                if args.dry_run:
                    print(f"[DRY-RUN] {info.filename} -> {out_path}")
                    written += 1
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                raw = zf.read(info)
                tmp_path = out_path.parent / f".{out_path.name}.tmp.{os.getpid()}"
                try:
                    with gzip.open(tmp_path, "wb") as f:
                        f.write(raw)
                    os.replace(tmp_path, out_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
                written += 1

        total_seen += seen
        total_written += written
        total_skipped += skipped
        total_invalid += invalid
        print(
            f"{zip_path}: seen={seen}, written={written}, skipped={skipped}, invalid={invalid}"
        )

    print(
        "Done. "
        f"total_seen={total_seen}, total_written={total_written}, "
        f"total_skipped={total_skipped}, total_invalid={total_invalid}"
    )


if __name__ == "__main__":
    main()
