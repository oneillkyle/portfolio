from __future__ import annotations
import os
import shutil
from typing import Tuple

from .utils import load_config, parse_args, ensure_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw_src = cfg["raw_data_path"]
    raw_dir = cfg["raw_dir"]
    ensure_dir(raw_dir)

    # Copy source into pipeline raw dir (symlink if large)
    dst = os.path.join(raw_dir, os.path.basename(raw_src))
    if not os.path.exists(dst):
        try:
            os.symlink(os.path.abspath(raw_src), dst)
        except OSError:
            shutil.copy2(raw_src, dst)
    print(f"Ingested raw file -> {dst}")


if __name__ == "__main__":
    main()
