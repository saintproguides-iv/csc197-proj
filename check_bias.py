"""
check_bias.py - do a model's predictions run too high or too low?

Compares one or more trained models on
  1. the VALIDATION titles: average prediction vs average true hours (never a test file), and
  2. a few coursework-style prompts, side by side, with the ratio between models.

Usage
  python check_bias.py                                   # compares ./saved_model and ./saved_model_v2
  python check_bias.py --model-dirs ./saved_model ./saved_model_v2
  python check_bias.py --force-clean        # older models get the repeated-letter/space cleaning too

If a model's average prediction on validation titles is close to the average true hours, it is
fine "in-domain", and any inflation on the probe prompts is specific to unfamiliar (coursework)
text. If it is well above, the model as a whole runs high.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROBES = [
    "Build a command-line calculator in Java",
    "Develop a Flask web app with a SQLite database",
    "Complete a SQL practice worksheet",
    "Write a short reflection on a programming lecture",
    "Study for the final exam in computer networks",
    "Develop a 2D game in Unity for a course project",
]


def summarize(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return {
        "mean_pred_h": round(float(pred.mean()), 2),
        "median_pred_h": round(float(np.median(pred)), 2),
        "mean_true_h": round(float(true.mean()), 2),
        "median_true_h": round(float(np.median(true)), 2),
        "mean_signed_error_h": round(float((pred - true).mean()), 2),
        "MAE_h": round(float(np.abs(pred - true).mean()), 2),
        "share_pred_over_8h": round(float((pred > 8).mean()), 3),
        "max_pred_h": round(float(pred.max()), 2),
    }


def main():
    p = argparse.ArgumentParser(description="Check whether predictions run high or low.")
    p.add_argument("--model-dirs", nargs="+", default=["./saved_model", "./saved_model_v2"])
    p.add_argument("--csv", default="data/sip_val.csv")
    p.add_argument("--force-clean", action="store_true",
                   help="for older models without a cleaning flag: collapse repeated letters and spaces first")
    args = p.parse_args()

    if "test" in os.path.basename(args.csv).lower():
        sys.exit("[ERROR] Use the validation file, not a test file.")
    if not os.path.exists(args.csv):
        sys.exit(f"[ERROR] Cannot find {args.csv}")
    df = pd.read_csv(args.csv)
    titles, true = df["task_description"].astype(str).tolist(), df["base_hours"].values

    from inference import Predictor
    rows, probe_preds = [], {}
    for d in args.model_dirs:
        pr = Predictor(d, force_clean=args.force_clean)
        row = {"model": os.path.basename(os.path.normpath(d))}
        row.update(summarize(pr.predict(titles), true))
        rows.append(row)
        probe_preds[row["model"]] = pr.predict(PROBES)
        print(f"Done: {d} ({pr.describe()})")

    print(f"\n=== Validation titles ({len(df)} rows) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    names = list(probe_preds)
    table = pd.DataFrame({"prompt": PROBES})
    for n in names:
        table[n] = np.round(probe_preds[n], 2)
    if len(names) >= 2:
        table[f"{names[-1]} / {names[0]}"] = np.round(probe_preds[names[-1]] / probe_preds[names[0]], 2)
    print("\n=== Coursework-style probe prompts (hours) ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
