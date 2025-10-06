# finetune_gpt2_agnews.py

# ---------------------
# imports
# ---------------------
import os
import random
import numpy as np
import torch
from datasets import load_dataset, ClassLabel , concatenate_datasets , DatasetDict
from transformers import (
    GPT2TokenizerFast,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import accuracy_score , f1_score , precision_recall_fscore_support , confusion_matrix

# ---------------------
# Config / Hyperparams
# ---------------------
MODEL_NAME = "gpt2"
OUTPUT_DIR = "./gpt2_large_agnews_baseline"
BATCH_SIZE = 16
EPOCHS = 10
LR = 5e-5
MAX_LENGTH = 256
SEED = 42
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED)
# ---------------------
# Load dataset
# ---------------------
dataset = load_dataset("ag_news")  # Hugging Face dataset
# dataset: splits "train" (120k) and "test" (7.6k). labels 0..3 
# World (0), Sports (1), Business (2), Sci/Tech (3)

# Merge train and test into one
full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

# Shuffle for randomness
full_dataset = full_dataset.shuffle(seed=SEED)

# First split: train (80%) and temp (20%)
train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=SEED)

# Now split the 20% temp into validation (10%) and test (10%)
valid_test = train_testvalid["test"].train_test_split(test_size=0.5, seed=SEED)

# Final splits
dataset = DatasetDict({
    "train": train_testvalid["train"], # 102080
    "validation": valid_test["train"], # 12760
    "test": valid_test["test"]         # 12760
})


# ---------------------
# Tokenizer and model
# ---------------------
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
# GPT-2 tokenizer has no pad token by default: set pad token to eos token to allow batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load model for sequence classification (num_labels = 4)
num_labels = 4
model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME , num_labels = num_labels)

# ensure model knows padding id (prevents warnings)
model.config.pad_token_id = tokenizer.pad_token_id

# move to device
# model.to(DEVICE)

def preprocess_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )

# Apply tokenizer to dataset:
tokenized_dataset = dataset.map(preprocess_fn, batched=True)

# Collator that pads to longest in batch (uses tokenizer.pad_token)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

# ---------------------
# Metrics
# ---------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------------
# Training Arguments & Trainer
# ---------------------
training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    overwrite_output_dir = True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    
    # --- EVALUATION AND LOGGING MODIFICATIONS ---
    
    # 1. EVALUATION: Keep evaluation running at the end of every epoch
    eval_strategy="epoch",  
    
    # 2. LOGGING STRATEGY: CHANGE from 'steps' (default) to 'epoch' 
    # This ensures train loss is logged at the same frequency as eval loss.
    logging_strategy="epoch",     
    
    # 3. TENSORBOARD: CHANGE 'report_to' from "none" to "tensorboard"
    report_to=["tensorboard"],   
    
    # 4. LOGGING STEPS: This is now ignored because logging_strategy is 'epoch'
    logging_steps=100,            
    
    # ---------------------------------------------
    
    save_strategy="epoch",
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir="./logs",
    save_total_limit=12,
    fp16=False,               
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---------------------
# Train
# ---------------------
trainer.train()

# ---------------------
# Evaluate on test set
# ---------------------
test_tok = tokenized_dataset["test"]
metrics = trainer.evaluate(test_tok)
print("Test metrics:", metrics)

# additional: get confusion matrix and per-class metrics
preds_output = trainer.predict(test_tok)
preds = np.argmax(preds_output.predictions, axis=1)
labels = preds_output.label_ids
print("Accuracy:", accuracy_score(labels, preds))
print("Weighted F1:", f1_score(labels, preds, average="weighted"))
print("Confusion matrix:\n", confusion_matrix(labels, preds))

# ---------------------
# Save model & tokenizer
# ---------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved baseline model to", OUTPUT_DIR)

# ---------------------
# Save model size (bytes)
# ---------------------
def folder_size_bytes(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            tot += os.path.getsize(fp)
    return tot

print("Model folder size (MB):", folder_size_bytes(OUTPUT_DIR) / 1e6)
