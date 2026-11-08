import json, math, os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./saved_model"

# Automatically detect available compute hardware
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Loading fine-tuned model on {device}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    with open(os.path.join(MODEL_DIR, "label_stats.json")) as f:
        stats = json.load(f)
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"\n[ERROR] Failed to load model from '{MODEL_DIR}': {e}")
    print("Ensure you have run 'python train.py' to generate model weights first.")
    sys.exit(1)

with open(os.path.join(MODEL_DIR, "label_stats.json")) as f:
    stats = json.load(f)

def predict_hours(task_description: str) -> float:
    inputs = tokenizer(task_description, return_tensors="pt",
                       truncation=True, max_length=64).to(device)
    with torch.no_grad():
        raw = model(**inputs).logits.squeeze().item()

    if not math.isfinite(raw):
        raise RuntimeError("Model produced NaN/inf. Retrain; training diverged.")

    log_pred = raw * stats["std"] + stats["mean"]   # undo standardization
    hours = float(np.expm1(log_pred))               # undo log1p
    return round(max(0.1, hours), 2)

def main():
    print("SAFE-PROCRASTINATION PREDICTOR — INTERACTIVE CLI")
    print("Enter any task description to predict base effort hours.")
    print("Type 'exit' or 'quit' to terminate the session.\n")

    while True:
        try:
            user_input = input("Enter task description > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting session. Good luck with your tasks!")
                break

            estimated_hours = predict_hours(user_input)
            print(f" Estimated Base Effort: {estimated_hours} hours\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated. Goodbye!")
            break

if __name__ == "__main__":
    main()