from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

from ai.transformers.t2_mlm_collator import DataCollatorForT5MLM

MODEL_OUT = "ai/saved_models/t5_wiki_pretrain"

# 2a) Stream the text file as an iterable dataset
ds = load_dataset(
    "text",
    data_files={"train": "ai/datasets/wiki_corpus.txt"},
    split="train",
    # streaming=True,
)

# 2b) Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 3) Tokenize every paragraph to fixed-length input_ids
def tokenize_fn(examples):
    # returns dict of lists, each list has len == batch_size
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized = ds.map(
    tokenize_fn,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
)

# 4) Now group those tokenized sequences into blocks of 512 **contiguously**.
#    This WILL reduce the number of examples, but is allowed.
block_size = 512
def group_texts(examples):
    # concatenate all input_ids together
    all_ids = sum(examples["input_ids"], [])
    # drop the tail so it divides evenly
    total_length = (len(all_ids) // block_size) * block_size
    result = {
        "input_ids": [
            all_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
    }
    return result

lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    batch_size=1000,
)
    
model = T5ForConditionalGeneration.from_pretrained("t5-small")

data_collator = DataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3,
    input_length=512,
    target_length=512,
)

training_args = TrainingArguments(
    output_dir=MODEL_OUT,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=5000,
    save_total_limit=2,
    fp16=True,                      # if you have a GPU
    logging_steps=1000,
    learning_rate=1e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)
