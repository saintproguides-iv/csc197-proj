import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# Detect best available compute device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using execution device: {device}")

# 1. Load Data
CSV_PATH = "tasks_dataset_expanded.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing {CSV_PATH} in project root!")

df = pd.read_csv(CSV_PATH)
df["log_hours"] = np.log1p(df["base_hours"]).astype(np.float32)

# 2. Train/Validation Split (80/20)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Standardize targets using TRAIN statistics only
LABEL_MEAN = float(train_df["log_hours"].mean())
LABEL_STD = float(train_df["log_hours"].std() + 1e-8)
train_df = train_df.assign(
    labels=((train_df["log_hours"] - LABEL_MEAN) / LABEL_STD).astype(np.float32)
)
val_df = val_df.assign(
    labels=((val_df["log_hours"] - LABEL_MEAN) / LABEL_STD).astype(np.float32)
)

train_dataset = Dataset.from_pandas(
    train_df[["task_description", "labels"]], preserve_index=False
)
val_dataset = Dataset.from_pandas(
    val_df[["task_description", "labels"]], preserve_index=False
)

# 3. Tokenization
MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["task_description"], truncation=True, max_length=64)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# 4. Evaluation Metrics (undo standardization, then undo log1p)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred_hours = np.expm1(np.squeeze(predictions) * LABEL_STD + LABEL_MEAN)
    true_hours = np.expm1(np.squeeze(labels) * LABEL_STD + LABEL_MEAN)
    mae = np.mean(np.abs(pred_hours - true_hours))
    rmse = np.sqrt(np.mean((pred_hours - true_hours) ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}

# 5. Initialize Model (full fine-tune, forced to float32)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
    dtype=torch.float32,   # if this errors, use torch_dtype=torch.float32
)
model = model.float()      # belt and suspenders: guarantees every param is float32
print("Model dtype:", next(model.parameters()).dtype)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    warmup_steps=8,
    max_grad_norm=1.0,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="mae",
    greater_is_better=False,
    logging_steps=5,
    save_total_limit=1,
    fp16=False,
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# 7. Run Training & Save Outputs
if __name__ == "__main__":
    print("Starting DeBERTa-v3 fine-tuning...")
    trainer.train()

    SAVE_DIR = "./saved_model"
    trainer.save_model(SAVE_DIR)          # saves the best model
    tokenizer.save_pretrained(SAVE_DIR)
    with open(os.path.join(SAVE_DIR, "label_stats.json"), "w") as f:
        json.dump({"mean": LABEL_MEAN, "std": LABEL_STD}, f)
    print(f"\nTraining complete. Saved model to: {SAVE_DIR}")