# 📚 QA-on-Wikipedia: Retrieval-Augmented T5 Pipeline

An end-to-end repository for building a factual question-answering system over Wikipedia:

- **Domain-adaptive pre-training** on Wikipedia text (T5)  
- **Semantic cleaning** of Natural Questions data  
- **BM25 retrieval** via Elasticsearch  
- **Fine-tuning** a T5 model on cleaned QA pairs  
- **Inference** with context-augmented prompts  
- **Batch evaluation** (BLEU & ROUGE)  
- **Containerized deployment** (Docker & Docker Compose)  
- **CI/CD** (GitHub Actions & GitLab CI)

---

## 🗂️ Repository Structure

```
.
├── data/
│   ├── wiki_json/                  ← WikiExtractor JSON output
│   └── cleaned_nq.jsonl            ← QA pairs after semantic cleaning
│
├── index_wiki_passages.py          ← Bulk-indexes wiki paragraphs into ES
├── semantic_cleaner_advanced.py    ← Filters & dedups NQ examples
├── split_jsonl.py                  ← Train/val split for JSONL
│
├── train_with_t5.py                ← Fine-tunes T5 on QA data
├── evaluate_bleu_rouge.py          ← Computes BLEU & ROUGE scores
│
├── predict_t5.py                   ← Interactive CLI with beam search
├── batch_predict_t5.py             ← Batch prediction (JSONL & CSV)
├── fastapi_t5.py                   ← FastAPI endpoints: /predict & /batch_predict
│
├── Dockerfile                      ← Production container for API & indexer
├── docker-compose.yml              ← Elasticsearch, indexer & API services
│
├── .github/workflows/              ← GitHub Actions CI/CD pipelines
└── .gitlab-ci.yml                  ← GitLab CI example
```

---

## ⚙️ Setup & Dependencies

1. **Python 3.10+**  
2. **Elasticsearch 7.x**  
3. **CUDA & GPU** (optional, for faster training/inference)

Install core Python deps:

```bash
pip install   fastapi uvicorn transformers torch   sentence-transformers elasticsearch   datasets nltk rouge-score tqdm
```

---

## 📖 Data Preparation

1. **Extract Wikipedia** (using WikiExtractor) into `data/wiki_json/`  
2. **Semantic cleaning** of NQ:
   ```bash
   python semantic_cleaner_advanced.py      --input simplified-nq-train.jsonl      --output data/cleaned_nq.jsonl      --limit 100000
   ```
3. **Split** into train/val:
   ```bash
   python split_jsonl.py      --input data/cleaned_nq.jsonl      --train_out data/cleaned_nq.train.jsonl      --val_out data/cleaned_nq.val.jsonl      --train_ratio 0.9
   ```

---

## 🗄️ Indexing with Elasticsearch

Build and populate a BM25 index of Wikipedia paragraphs:

```bash
docker-compose up -d elasticsearch
python index_wiki_passages.py   --wiki_dir data/wiki_json/   --es_host http://localhost:9200   --index qa_passages
```

---

## 🤖 Model Training

### 1. **Pre-training (optional)**
Continue pre-training T5 on Wikipedia corpus:

Optionally take a subsample of the corpus for training.
```bash
shuf -n 1000000 ai/datasets/wiki_corpus.txt > ai/datasets/wiki_corpus.subsample.txt
```

```bash
python pretrain_t5_wiki.py
# outputs → t5_wiki_pretrain/
```

### 2. **Fine-tuning on QA**

```bash
python train_with_t5.py
# loads from t5-small (or t5_wiki_pretrain/)
# outputs → t5_trained_nq/
```

---

## 📊 Evaluation

Compute BLEU & ROUGE on validation set:

```bash
python evaluate_bleu_rouge.py   --model_dir t5_trained_nq   --data data/cleaned_nq.val.jsonl   --max_len 64 --beams 4
```

Use `results_visualizer.py` or React dashboard for plots.

---

## 🚀 Inference

### Interactive CLI

```bash
python predict_t5.py   --model_dir t5_trained_nq   --device cuda   --num_beams 5
```

### Batch

```bash
python batch_predict_t5.py   --model_dir t5_trained_nq   --input example_eval.jsonl   --output predictions.csv
```

### FastAPI Server

```bash
uvicorn fastapi_t5:app --host 0.0.0.0 --port 8000 --reload
```

- **POST** `/predict` → `{ "question": "..." }`  
- **POST** `/batch_predict` → `[{"question":"..."}, …]`

---

## 📦 Deployment

1. Place your `t5_trained_nq/` under `./models/`  
2. Ensure `./wiki_json/` and `docker-compose.yml` are present  
3. Launch all services:

   ```bash
   docker-compose up --build
   ```

Access the API at **http://localhost:8000/docs**.

---

## 🔄 CI / CD

- **GitHub Actions**: see `.github/workflows/ci.yml` & `cd.yml`  
- **GitLab CI**: see `.gitlab-ci.yml`

On every push to **main**, code is linted, tested, Docker images built—and on deploy, containers are updated via SSH.

---

## 📚 Further Reading

- “Retrieval-Augmented Generation” (RAG): https://arxiv.org/abs/2005.11401  
- Hugging Face T5 tutorial: https://huggingface.co/docs/transformers/model_doc/t5  
- Elasticsearch BM25 & vector search docs

---

> **“A model is only as good as its data and grounding.”**  
> By combining Wikipedia pre-training with explicit retrieval, you’ll get both broad knowledge and up-to-date facts. Enjoy building!
