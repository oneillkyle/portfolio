from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import yaml
from .utils import load_config, parse_args, ensure_dir

def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = os.path.join(cfg["output_dir"], "exports")
    ensure_dir(out_dir)

    # Use latest trained checkpoint if available
    model_dir = cfg.get("log_dir", "./logs")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Exported PyTorch model+tokenizer -> {out_dir}")

if __name__ == "__main__":
    main()
