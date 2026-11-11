import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from text_utils import clean_text

# ----------------------------------------------------------------------
# Command-line options (all optional; defaults shown)
#   python train.py
#   python train.py --train data/sip_train_aug.csv --save-dir ./saved_model_v2
#   python train.py --no-clean            (reproduces the old behaviour: no text cleaning)
# NOTE: data/sip_test.csv is intentionally NOT loaded here.
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fine-tune DeBERTa to predict task hours.")
parser.add_argument("--train", default="data/sip_train.csv")
parser.add_argument("--val", default="data/sip_val.csv")
parser.add_argument("--save-dir", default="./saved_model_v2")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
parser.add_argument("--no-clean", action="store_true",
                    help="do not clean the text (collapse repeated letters, spaces, lowercase)")
args = parser.parse_args()

TRAIN_PATH = args.train
VAL_PATH = args.val
NUM_EPOCHS = args.epochs
MODEL_NAME = args.model_name
SAVE_DIR = args.save_dir
CLEAN = not args.no_clean
LOWERCASE = True

# Detect best available compute device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using execution device: {device}")

# 1. Load pre-split data (made by prepare_sip.py)
for p in (TRAIN_PATH, VAL_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing {p}. Put the CSVs in the data/ folder or run prepare_sip.py first."
        )

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

if CLEAN:
    # the SAME cleaning is applied in inference.py to whatever the user types
    for d in (train_df, val_df):
        d["task_description"] = d["task_description"].map(lambda t: clean_text(t, LOWERCASE))
    print("Text cleaning ON: repeated letters collapsed, spaces fixed, lowercased")
else:
    print("Text cleaning OFF")

for d in (train_df, val_df):
    d["log_hours"] = np.log1p(d["base_hours"]).astype(np.float32)
print(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)}")

# 2. Standardize targets using TRAIN statistics only
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
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="mae",
    greater_is_better=False,
    logging_steps=50,
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

# 7. Run Training, print results, save outputs
if __name__ == "__main__":
    print("Starting DeBERTa-v3 fine-tuning...")
    trainer.train()

    # --- validation results (best epoch, since load_best_model_at_end=True) ---
    metrics = trainer.evaluate()
    print("\n=== Validation results (best epoch) ===")
    print(f"MAE : {metrics['eval_mae']:.2f} hours")
    print(f"RMSE: {metrics['eval_rmse']:.2f} hours")

    # baseline: always guess the training median
    base = float(train_df["base_hours"].median())
    base_mae = float((val_df["base_hours"] - base).abs().mean())
    print(f"Baseline (always guess {base:.2f} h) MAE: {base_mae:.2f} hours")

    print("\nEpoch-by-epoch:")
    for log in trainer.state.log_history:
        if "eval_mae" in log:
            print(f"  epoch {log['epoch']:.0f}: MAE {log['eval_mae']:.2f}  RMSE {log['eval_rmse']:.2f}")

    # --- save the best model + what inference needs to know ---
    trainer.save_model(SAVE_DIR)          # saves the best model
    tokenizer.save_pretrained(SAVE_DIR)
    with open(os.path.join(SAVE_DIR, "label_stats.json"), "w") as f:
        json.dump({"mean": LABEL_MEAN, "std": LABEL_STD,
                   "clean": CLEAN, "lowercase": LOWERCASE if CLEAN else False}, f)
    print(f"\nTraining complete. Saved model to: {SAVE_DIR}")
