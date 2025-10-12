from __future__ import annotations
import os
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

from .utils import load_config, parse_args, ensure_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = os.path.join(cfg["output_dir"], "exports")
    ensure_dir(out_dir)

    # Prefer latest trained checkpoint if pointer exists
    exp = cfg.get("experiment_name", "default")
    latest_ptr = os.path.join(cfg["log_dir"], exp, "latest.txt")
    model_src = open(latest_ptr).read().strip() if os.path.exists(latest_ptr) else cfg.get("ckpt_dir", cfg["model_name"])
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_src)
    tok = AutoTokenizer.from_pretrained(model_src if os.path.exists(model_src) else cfg["model_name"])

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Exported HF model+tokenizer -> {out_dir}")


if __name__ == "__main__":
    main()
