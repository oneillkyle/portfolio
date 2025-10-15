from __future__ import annotations
import os
import shutil
from typing import Tuple

from .utils import load_config, parse_args, ensure_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Read source path and raw_dir; allow backwards compatibility if source_data_path is missing
    raw_src = cfg.get("source_data_path", cfg.get("raw_data_path"))
    if not raw_src or not isinstance(raw_src, str):
        raise ValueError("source_data_path/raw_data_path is missing or invalid in config")
    raw_dir = cfg["raw_dir"]
    ensure_dir(raw_dir)

    # Copy source into pipeline raw dir (symlink if large)
    dst = os.path.join(raw_dir, os.path.basename(raw_src))
    if not os.path.exists(dst):
        try:
            os.symlink(os.path.abspath(raw_src), dst)
        except OSError:
            shutil.copy2(raw_src, dst)
    # If config has raw_data_path pointing elsewhere, rewrite it to canonical location
    cfg_path = args.config
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["raw_data_path"] = dst
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"Ingested raw file -> {dst} (updated raw_data_path in {cfg_path})")
    except Exception as e:
        # Non-fatal; continue
        print(f"Ingested raw file -> {dst} (could not update config: {e})")


if __name__ == "__main__":
    main()
