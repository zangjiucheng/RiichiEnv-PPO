"""Minimal uploader for Hugging Face Hub model repo.

This script intentionally keeps a very small interface and follows:
    login()
    upload_folder(folder_path=..., repo_id="zangjiucheng/riichippo", repo_type="model")
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from huggingface_hub import login, upload_folder
except ModuleNotFoundError as e:
    raise SystemExit(
        "huggingface_hub is required. Install with: pip install huggingface_hub"
    ) from e


REPO_ID = "zangjiucheng/riichippo"
REPO_TYPE = "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a folder to HF repo zangjiucheng/riichippo")
    parser.add_argument(
        "--folder",
        default=".",
        help="Folder to upload (default: current directory).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Target subfolder in the HF repo (default: repo root).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Sync mode: delete remote files under target path that are not in local --folder. "
            "Uses delete_patterns='**'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")
    path_in_repo = args.path_in_repo.strip("/") if args.path_in_repo else None
    delete_patterns = "**" if args.sync else None

    login()
    commit_info = upload_folder(
        folder_path=str(folder),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo=path_in_repo,
        delete_patterns=delete_patterns,
    )
    print(f"Upload complete: {commit_info}")


if __name__ == "__main__":
    main()
