from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
from typing import cast, Optional
import math
import numpy as np
import os
import yaml
from .utils import load_config, parse_args, ensure_dir
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

import transformers
print(transformers.__file__)
from transformers import TrainingArguments
print(TrainingArguments.__init__.__doc__)

import numpy as np
def _random_spans_noise_mask(length: int, noise_density: float, mean_span_length: float) -> np.ndarray:
    num_noise_tokens = int(np.round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_spans = int(np.round(num_noise_tokens / mean_span_length))
    num_spans = max(num_spans, 1)
    span_lengths = np.random.multinomial(num_noise_tokens, [1/num_spans]*num_spans)
    num_nonnoise = length - num_noise_tokens
    gap_lengths = np.random.multinomial(num_nonnoise, [1/(num_spans+1)]*(num_spans+1))
    mask = np.array([], dtype=bool)
    for gap, span in zip(gap_lengths, np.append(span_lengths, 0)):
        mask = np.concatenate([mask, np.zeros(gap, dtype=bool), np.ones(span, dtype=bool)])
    mask = np.concatenate([mask, np.zeros(gap_lengths[-1], dtype=bool)])
    return mask[:length]

def _sentinel_id(tokenizer, n: int) -> int:
    tok = tokenizer.convert_tokens_to_ids(f"<extra_id_{n}>")
    if isinstance(tok, list):
        return tok[0]
    return tok

def span_corrupt_example(example, tokenizer, noise_density=0.15, mean_span_length=3, block_size=256):
    ids = tokenizer.encode(example["text"], truncation=True, max_length=block_size)
    mask = _random_spans_noise_mask(len(ids), noise_density, mean_span_length)
    input_ids = []
    target_ids = []
    sentinel_count = 0
    i = 0
    while i < len(ids):
        if mask[i]:
            input_ids.append(_sentinel_id(tokenizer, sentinel_count))
            target_ids.append(_sentinel_id(tokenizer, sentinel_count))
            sentinel_count += 1
            while i < len(ids) and mask[i]:
                target_ids.append(ids[i])
                i += 1
        else:
            input_ids.append(ids[i])
            i += 1
    target_ids.append(tokenizer.eos_token_id)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = np.pad(input_ids, (0, max(0, block_size - len(input_ids))), constant_values=pad_id)[:block_size]
    target_ids = np.pad(target_ids, (0, max(0, block_size - len(target_ids))), constant_values=-100)[:block_size]
    return {"input_ids": input_ids.tolist(), "labels": target_ids.tolist()}

def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Optional TF32 for speed on Ampere+; disabled when debug_numerics is True
    allow_tf32_cfg = bool(cfg.get("allow_tf32", True))
    debug_numerics = bool(cfg.get("debug_numerics", True))
    if allow_tf32_cfg and not debug_numerics and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # PyTorch 2.x API to pick precision policy
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            print("[DEBUG] TF32 enabled for matmul and cuDNN")
        except Exception as e:
            print(f"[DEBUG] Could not enable TF32: {e}")

    print("[DEBUG] Starting data load...")
    train_ds = cast(Dataset, load_dataset("EdinburghNLP/xsum", split="train", revision="main"))
    val_ds = cast(Dataset, load_dataset("EdinburghNLP/xsum", split="validation", revision="main"))
    print(f"[DEBUG] Train split loaded: {len(train_ds)} samples")
    print(f"[DEBUG] Validation split loaded: {len(val_ds)} samples")
    print("[DEBUG] Dataset loaded.")

    def preprocess(example):
        input_text = "summarize: " + example["document"]
        target_text = example["summary"]
        model_inputs = tokenizer(input_text, max_length=cfg["block_size"], truncation=True)
        labels = tokenizer(target_text, max_length=cfg["block_size"], truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("[DEBUG] Applying preprocessing to train set...")
    train_ds = train_ds.map(preprocess)
    print("[DEBUG] Train preprocessing complete.")
    print("[DEBUG] Applying preprocessing to validation set...")
    val_ds = val_ds.map(preprocess)
    print("[DEBUG] Validation preprocessing complete.")

    print("[DEBUG] Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("[DEBUG] Model loaded.")

    # For streaming datasets, set max_steps instead of num_train_epochs
    # Use diagnostic_mode to disable eval/save/logging for fast iteration
    diagnostic_mode = bool(cfg.get("diagnostic_mode", False))
    max_steps = int(cfg.get("max_steps", 10000))
    
    # Higher default intervals to avoid frequent eval/save stalls
    step_interval = int(cfg.get("save_interval", cfg.get("eval_steps", 5000)))
    eval_steps = int(cfg.get("eval_steps", step_interval))
    save_steps = int(cfg.get("save_steps", step_interval))
    logging_steps = int(cfg.get("logging_steps", 100))

    # If debug_numerics is enabled, force full precision
    # Build TrainingArguments via kwargs so we can conditionally include fields
    num_workers = int(cfg.get("num_workers", max(1, (os.cpu_count() or 2)//2)))
    pin_memory = bool(cfg.get("pin_memory", True)) and torch.cuda.is_available()
    persistent_workers = bool(cfg.get("persistent_workers", True)) if num_workers > 0 else False
    disable_wandb = bool(cfg.get("disable_wandb", False))
    report_targets = ["none"] if diagnostic_mode else (["tensorboard"] if disable_wandb else (["tensorboard", "wandb"] if cfg.get("use_wandb", False) else ["tensorboard"]))

    args_kwargs = dict(
        output_dir=cfg["log_dir"],
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        max_steps=max_steps,
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        eval_strategy="no" if diagnostic_mode else "steps",
        save_strategy="no" if diagnostic_mode else "steps",
        eval_steps=None if diagnostic_mode else eval_steps,
        save_steps=None if diagnostic_mode else save_steps,
        logging_steps=logging_steps,
        logging_dir=os.path.join(cfg["log_dir"], "tensorboard"),
        report_to=report_targets,
        fp16=False if debug_numerics else cfg.get("mixed_precision", False),
        bf16=False if debug_numerics else bool(cfg.get("bf16", False)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        gradient_accumulation_steps=int(cfg.get("grad_accum_steps", 1)),
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_persistent_workers=persistent_workers,
        run_name=cfg.get("run_name", "t5-xsum-training"),
        seed=int(cfg.get("random_seed", 42)),
        data_seed=int(cfg.get("random_seed", 42)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
    )
    if num_workers > 0:
        args_kwargs["dataloader_prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    training_args = TrainingArguments(**args_kwargs)

    # Use DataCollatorForSeq2Seq to properly pad batches
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Optional anomaly detection (very slow). Enable by setting detect_anomaly: true in config.
    if cfg.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    class NaNSafeTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Prepare inputs first (handles device move, etc.)
            inputs = self._prepare_inputs(inputs)

            # Debug every N steps or on anomaly
            step = int(getattr(self.state, "global_step", 0))
            debug_every = int(self.args.logging_steps) if cfg.get("debug_every_log", True) else int(cfg.get("debug_every", 0) or 0)

            def tensor_stats(name, t: torch.Tensor):
                try:
                    mn = t.min().item(); mx = t.max().item(); mean = t.float().mean().item()
                    n_nan = int(torch.isnan(t).sum().item()); n_inf = int(torch.isinf(t).sum().item())
                    print(f"[DEBUG] Step {step} {name}: shape={tuple(t.shape)} min={mn} max={mx} mean={mean} n_nan={n_nan} n_inf={n_inf}")
                except Exception as e:
                    print(f"[DEBUG] Step {step} {name}: stats error: {e}")

            # Print inputs stats conditionally
            should_log = debug_every > 0 and (step % debug_every == 0)
            if should_log:
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        tensor_stats(k, v)

            # Validate labels presence and content before forward
            labels = inputs.get("labels")
            if labels is None:
                print(f"[ERROR] No 'labels' in inputs at step {step}.")
            elif isinstance(labels, torch.Tensor):
                valid_mask = (labels != -100)
                valid = int(valid_mask.sum().item())
                total = int(labels.numel())
                if valid == 0 and total > 0:
                    print(f"[ERROR] All labels are -100 (ignored) at step {step}. total={total}")
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and k in ("input_ids", "attention_mask", "labels"):
                            tensor_stats(k, v)
                    raise RuntimeError("All labels ignored (-100). Aborting to debug data pipeline.")

            # Forward pass
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # Check loss for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                # Always print detailed stats on anomaly
                print(f"[ERROR] NaN/Inf loss detected at step {step}.")
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        tensor_stats(k, v)
                # Inspect labels valid token ratio if available
                labels = inputs.get("labels")
                if isinstance(labels, torch.Tensor):
                    valid = int((labels != -100).sum().item()); total = int(labels.numel())
                    print(f"[DEBUG] Labels valid/total: {valid}/{total} ({(valid/total*100) if total>0 else 0:.2f}%)")
                # Stop training immediately so we can inspect
                raise RuntimeError("NaN/Inf loss encountered; aborting to preserve state")

            # Explicit warning on zero loss patterns
            try:
                loss_val = float(loss.detach().item())
            except Exception:
                loss_val = None
            if loss_val is not None and (loss_val == 0.0 or not math.isfinite(loss_val)):
                print(f"[WARN] Zero loss at step {step}. Investigating batch...")
                labels = inputs.get("labels")
                if isinstance(labels, torch.Tensor):
                    valid = int((labels != -100).sum().item()); total = int(labels.numel())
                    print(f"[DEBUG] Labels valid/total: {valid}/{total} ({(valid/total*100) if total>0 else 0:.2f}%)")
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and k in ("input_ids", "attention_mask", "labels"):
                        tensor_stats(k, v)
                if loss_val is None or not math.isfinite(loss_val):
                    raise RuntimeError("Non-finite loss value encountered.")

            if return_outputs:
                return (loss, outputs)
            return loss

    print("[DEBUG] Initializing Trainer...")
    from transformers import TrainerCallback

    class NumericsGuardCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            model = kwargs.get("model")
            if model is None:
                return
            for n, p in model.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    print(f"[ERROR] Non-finite gradients detected in param {n} at step {state.global_step}")
                    control.should_training_stop = True
                    print("[ERROR] Stopping training due to non-finite gradients.")
                    break

    class LogWatchCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss")
            grad_norm = logs.get("grad_norm")
            bad = False
            if loss is not None:
                try:
                    lv = float(loss)
                    if lv == 0.0 or not math.isfinite(lv):
                        print(f"[WARN] Log shows suspicious loss={lv} at step {state.global_step}")
                        bad = True
                except Exception:
                    pass
            if grad_norm is not None:
                try:
                    gn = float(grad_norm)
                    if not math.isfinite(gn):
                        print(f"[WARN] Log shows non-finite grad_norm={grad_norm} at step {state.global_step}")
                        bad = True
                except Exception:
                    pass
            if bad:
                control.should_training_stop = True
                print("[ERROR] Stopping training due to suspicious logs (zero/NaN).")

    class PerfMonitorCallback(TrainerCallback):
        def __init__(self):
            self._last_step = None
            self._last_time = None
            self.stop_reason = None

        def on_step_begin(self, args, state, control, **kwargs):
            import time
            self._last_step = state.global_step
            self._last_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            import time
            now = time.time()
            if self._last_time is not None:
                dt = now - self._last_time
                print(f"[PERF] Step {state.global_step} took {dt:.3f}s")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"[PERF] CUDA max_memory_allocated ~ {mem:.2f} GB")

        def on_train_end(self, args, state, control, **kwargs):
            reason = getattr(control, 'should_training_stop', False)
            print(f"[PERF] Training ended. should_training_stop={reason}")

    class LongOpWatchdog(TrainerCallback):
        def __init__(self, step_timeout: Optional[float] = None):
            self.step_timeout = step_timeout or float(cfg.get("step_timeout_secs", 60))
            self._step_start_time = None
            self._last_dump_time = None
        def on_step_begin(self, args, state, control, **kwargs):
            import time
            self._step_start_time = time.time()
            self._last_dump_time = self._step_start_time
        def on_step_end(self, args, state, control, **kwargs):
            self._step_start_time = None
            self._last_dump_time = None
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Evaluate potential stall mid-step
            if self._step_start_time is None:
                return
            import time, threading, traceback, sys
            elapsed = time.time() - self._step_start_time
            if elapsed > self.step_timeout:
                now = time.time()
                if self._last_dump_time is None or (now - self._last_dump_time) >= 10:
                    self._last_dump_time = now
                    print(f"[WATCHDOG] Step {state.global_step} exceeding {self.step_timeout:.1f}s (elapsed {elapsed:.1f}s); dumping stack traces.")
                    for thread_id, frame in sys._current_frames().items():
                        print(f"[WATCHDOG] Thread {thread_id} stack:")
                        traceback.print_stack(frame)
                    # Stop after first full dump to allow analysis
                    control.should_training_stop = True
                    print("[WATCHDOG] Requesting training stop due to stall.")
    trainer = NaNSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if diagnostic_mode else val_ds,
        data_collator=data_collator,
    )
    trainer.add_callback(NumericsGuardCallback())
    trainer.add_callback(LogWatchCallback())
    trainer.add_callback(PerfMonitorCallback())
    trainer.add_callback(LongOpWatchdog())
    print("[DEBUG] Starting training...")
    trainer.train()
    print("[DEBUG] Training complete. Saving model...")
    trainer.save_model(cfg["log_dir"])
    print("[DEBUG] Model saved.")

if __name__ == "__main__":
    main()
