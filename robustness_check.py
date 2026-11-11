"""
robustness_check.py - how much do predictions move when the text is damaged?

For N validation titles it predicts hours for the original text, then for versions with
different case, extra spaces, a repeated letter, one typo, or two typos, and reports how
much the prediction changed. Smaller numbers = more robust.

Usage (run once per model, then compare the tables)
  python robustness_check.py --model-dir ./saved_model                 # v1, as it was
  python robustness_check.py --model-dir ./saved_model --force-clean   # v1 + repeated-letter fix, no retraining
  python robustness_check.py --model-dir ./saved_model_v2              # retrained with cleaning + typo data

Uses validation data only (never a test file). Changes are measured against the model's OWN
prediction on the original text, so this measures stability, not accuracy.
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

from text_utils import add_typo, repeat_letter

TYPO_OPS = ("swap", "delete", "substitute")


def _changed(fn, text, tries=8):
    for _ in range(tries):
        r = fn(text)
        if r != text:
            return r
    return text


def build_variants(texts, rng):
    one_typo = lambda t: add_typo(t, rng, ops=TYPO_OPS)
    return {
        "lowercase": [t.lower() for t in texts],
        "UPPERCASE": [t.upper() for t in texts],
        "Title Case": [t.title() for t in texts],
        "extra spaces": [t.replace(" ", "  ") for t in texts],
        "repeated letter (x3-5)": [_changed(lambda x: repeat_letter(x, rng), t) for t in texts],
        "one typo": [_changed(one_typo, t) for t in texts],
        "two typos": [_changed(lambda x: one_typo(one_typo(x)), t) for t in texts],
    }


def run_robustness(predict_fn, texts, seed=42):
    """predict_fn: list[str] -> np.array of hours. Returns a DataFrame, one row per perturbation."""
    rng = random.Random(seed)
    base = np.asarray(predict_fn(texts), dtype=float)
    rows = []
    for name, variant in build_variants(texts, rng).items():
        n_changed = sum(a != b for a, b in zip(variant, texts))
        p = np.asarray(predict_fn(variant), dtype=float)
        diff = np.abs(p - base)
        rel = (p - base) / np.maximum(base, 0.1)
        rows.append({
            "perturbation": name,
            "texts_changed": n_changed,
            "mean_abs_change_h": round(float(diff.mean()), 2),
            "median_abs_change_h": round(float(np.median(diff)), 2),
            "mean_abs_change_pct": round(float(np.abs(rel).mean() * 100), 1),
            "share_changed_over_25pct": round(float((np.abs(rel) > 0.25).mean()), 3),
            "mean_signed_change_pct": round(float(rel.mean() * 100), 1),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Measure prediction stability under text damage.")
    p.add_argument("--model-dir", default="./saved_model")
    p.add_argument("--csv", default="data/sip_val.csv")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-clean", action="store_true",
                   help="for older models: collapse repeated letters and spaces before predicting")
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    if "test" in os.path.basename(args.csv).lower():
        sys.exit("[ERROR] Do not run this on a test file. Use the validation file.")
    if not os.path.exists(args.csv):
        sys.exit(f"[ERROR] Cannot find {args.csv}")

    texts = (pd.read_csv(args.csv)["task_description"].astype(str)
             .drop_duplicates().sample(n=args.n, random_state=args.seed).tolist())

    from inference import Predictor
    predictor = Predictor(args.model_dir, force_clean=args.force_clean)
    print(f"Model: {args.model_dir} on {predictor.device} ({predictor.describe()})")
    print(f"Titles tested: {len(texts)} (from {args.csv})\n")

    table = run_robustness(predictor.predict, texts, args.seed)
    print(table.to_string(index=False))
    print(f"\nOverall mean change across perturbations: {table['mean_abs_change_pct'].mean():.1f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.model_dir)) + ("_forceclean" if args.force_clean else "")
    path = os.path.join(args.out_dir, f"robustness_{tag}.csv")
    table.to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
