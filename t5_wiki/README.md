# TensorFlow T5 Pipeline

End-to-end pipeline to pre-train a T5-style seq2seq model on a raw text corpus using TensorFlow + Hugging Face Transformers.

## layout

t5_wiki/
- data/
	- raw/               # raw source files (symlinked/copied during ingest)
	- processed/         # TFRecord shards written by transform
- src/
	- ingest.py          # bring raw text into data/raw
	- transform.py       # tokenize, chunk, create labels (-100 mask), write TFRecords
	- model.py           # HF TF model wrapped for Keras (returns logits)
	- train.py           # tf.data input, masked loss, TensorBoard, checkpoints
	- evaluate.py        # loss/perplexity on val or train
	- export.py          # export latest trained checkpoint
- configs/
	- default.yaml       # default hyperparameters and paths
- scripts/
	- run_pipeline.sh    # ingest -> transform -> train -> evaluate -> export
	- smoke_test.py      # tiny E2E sanity check
- logs/                # {experiment}/{timestamp}/{tensorboard,checkpoints}

## prerequisites

- Python 3.10+
- GPU optional (recommended)
- Install dependencies:

```
python -m venv .venv
. .venv/bin/activate
pip install -r t5_wiki/requirements.txt
```

## configuration

`t5_wiki/configs/default.yaml` keys:

- data
	- `raw_data_path`: path to your raw text (one document per line)
	- `raw_dir`, `processed_dir`, `output_dir`
- model/tokenization
	- `model_name`: e.g. `t5-small`
	- `max_length`: per-line tokenizer max length
	- `block_size`: final sequence length in TFRecords
- training
	- `batch_size`, `epochs`, `learning_rate`, `mixed_precision`
- logging
	- `experiment_name`, `log_dir`

## run the pipeline

Run all steps:

```
bash t5_wiki/scripts/run_pipeline.sh --config t5_wiki/configs/default.yaml
// OR

bash python3 -m t5_wiki.scripts.run_all
```

Or step-by-step:

```
python -m t5_wiki.src.ingest --config t5_wiki/configs/default.yaml
python -m t5_wiki.src.transform --config t5_wiki/configs/default.yaml
python -m t5_wiki.src.train --config t5_wiki/configs/default.yaml
python -m t5_wiki.src.evaluate --config t5_wiki/configs/default.yaml
python -m t5_wiki.src.export --config t5_wiki/configs/default.yaml
```

python3 -m t5_wiki.scripts.make_test_split --num_lines 1000

View TensorBoard:

```
tensorboard --logdir t5_wiki/logs
```

## TensorBoard integration

All training, evaluation, test, and tuning metrics are logged to TensorBoard for easy visualization and comparison.

- **Training:**
  - Logs are written to `t5_wiki/logs/{experiment}/{timestamp}/tensorboard/`
- **Evaluation:**
  - Logs are written to `t5_wiki/logs/{experiment}/eval_tensorboard/`
- **Test:**
  - Logs are written to `t5_wiki/logs/{experiment}/test_tensorboard/`
- **Tuning:**
  - Logs are written to `t5_wiki/logs/tuning/tuning_tensorboard/`

To launch TensorBoard and view all experiment metrics:

```bash
. .venv/bin/activate
# From the project root:
tensorboard --logdir t5_wiki/logs
```

Open the displayed URL in your browser to explore training curves, validation/test metrics, and compare tuning runs.

You can filter, group, and compare runs by experiment, timestamp, or hyperparameters.

## notes

- Transform creates labels by shifting input blocks and masking pad positions with -100; training uses `sample_weight` to ignore those.
- To mimic T5 span corruption, we can add a span-masking step in `transform.py` and adjust labels accordingly.