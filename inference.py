"""
inference.py - one place that loads a trained model and turns text into predicted hours.

predict.py, evaluate_test.py and robustness_check.py all use this, so the cleaning applied
to the input is always the same one the model was trained with.

Which cleaning is used?
  train.py writes "clean" and "lowercase" into saved_model*/label_stats.json.
  - Models trained with cleaning (v2):  the same cleaning is applied automatically.
  - Older models (v1) have no such flag: no cleaning, unless force_clean=True, which
    collapses repeated letters and spaces (but does not lowercase, since v1 was trained on
    cased text). That lets you test the repeated-letter fix on v1 without retraining.
"""
import json
import os

import numpy as np

from text_utils import clean_text


class Predictor:
    def __init__(self, model_dir="./saved_model", force_clean=False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model folder '{model_dir}' not found. Run train.py first.")
        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        with open(os.path.join(model_dir, "label_stats.json")) as f:
            self.stats = json.load(f)

        self.clean = bool(self.stats.get("clean", False))
        self.lowercase = bool(self.stats.get("lowercase", False))
        if force_clean and not self.clean:
            self.clean, self.lowercase = True, False

    def describe(self):
        if not self.clean:
            return "no input cleaning"
        return "input cleaning: collapse repeated letters + spaces" + (" + lowercase" if self.lowercase else "")

    def prepare(self, text):
        return clean_text(text, self.lowercase) if self.clean else str(text)

    def predict(self, texts, batch_size=32):
        texts = [self.prepare(t) for t in texts]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i + batch_size], return_tensors="pt",
                                     truncation=True, max_length=64, padding=True).to(self.device)
                raw = self.model(**enc).logits.squeeze(-1).float().cpu().numpy()
                raw = np.atleast_1d(raw)
                if not np.all(np.isfinite(raw)):
                    raise RuntimeError("Model produced NaN/inf. Retrain; training diverged.")
                log_pred = raw * self.stats["std"] + self.stats["mean"]   # undo standardization
                out.append(np.maximum(0.1, np.expm1(log_pred)))           # undo log1p
        return np.concatenate(out)

    def predict_one(self, text):
        return round(float(self.predict([text])[0]), 2)
