# TensorFlow T5 Pipeline

End-to-end pipeline to pre-train a T5-style seq2seq model on a raw text corpus using TensorFlow + Hugging Face Transformers.

## layout

my_pipeline/
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
pip install -r my_pipeline/requirements.txt
```

## configuration

`my_pipeline/configs/default.yaml` keys:

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
bash my_pipeline/scripts/run_pipeline.sh --config my_pipeline/configs/default.yaml
```

Or step-by-step:

```
python -m my_pipeline.src.ingest --config my_pipeline/configs/default.yaml
python -m my_pipeline.src.transform --config my_pipeline/configs/default.yaml
python -m my_pipeline.src.train --config my_pipeline/configs/default.yaml
python -m my_pipeline.src.evaluate --config my_pipeline/configs/default.yaml
python -m my_pipeline.src.export --config my_pipeline/configs/default.yaml
```

View TensorBoard:

```
tensorboard --logdir my_pipeline/logs
```

## notes

- Transform creates labels by shifting input blocks and masking pad positions with -100; training uses `sample_weight` to ignore those.
- To mimic T5 span corruption, we can add a span-masking step in `transform.py` and adjust labels accordingly.