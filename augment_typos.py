"""
augment_typos.py - add misspelled / re-cased copies of TRAINING titles (same hours).

Why: the model has only seen clean titles, so a typo makes a known task look unfamiliar.
Showing it damaged copies with the same label teaches it that a typo does not change the task.

Usage
  python augment_typos.py                          # data/sip_train.csv -> data/sip_train_aug.csv
  python augment_typos.py --copies 2 --seed 42

Only augment the TRAINING file. Never augment validation or test files.
Training on the output takes (1 + copies) times longer.
"""
import argparse
import os
import random
import sys

import pandas as pd

from text_utils import add_typo, change_case, repeat_letter


def main():
    p = argparse.ArgumentParser(description="Create a typo-augmented training file.")
    p.add_argument("--input", default="data/sip_train.csv")
    p.add_argument("--out", default="data/sip_train_aug.csv")
    p.add_argument("--copies", type=int, default=2, help="damaged copies per title")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if any(x in os.path.basename(args.input).lower() for x in ("val", "test")):
        sys.exit(f"[ERROR] {args.input} looks like a validation or test file. Augment training data only.")
    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] Cannot find {args.input}")

    df = pd.read_csv(args.input)
    rng = random.Random(args.seed)
    rows = []
    for text, hours in zip(df["task_description"].astype(str), df["base_hours"]):
        for _ in range(args.copies):
            t = text
            for _ in range(1 if rng.random() < 0.75 else 2):        # 1 typo (75%) or 2 typos (25%)
                t = add_typo(t, rng)
            if rng.random() < 0.20:                                  # sometimes a repeated letter too
                t = repeat_letter(t, rng)
            t = change_case(t, rng.choice(["keep", "keep", "lower", "upper", "title"]))
            if t != text:
                rows.append({"task_description": t, "base_hours": hours})

    aug = pd.DataFrame(rows)
    out = pd.concat([df[["task_description", "base_hours"]], aug], ignore_index=True)
    out = out.drop_duplicates(subset="task_description", keep="first").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"Original rows: {len(df)} | damaged copies added: {len(out) - len(df)} | total: {len(out)}")
    print("\nExamples (original -> damaged copy):")
    for _, r in aug.sample(n=min(8, len(aug)), random_state=args.seed).iterrows():
        print(f"  {r['task_description'][:100]}")
    print(f"\nWrote {args.out}. Train on it with:  python train.py --train {args.out}")


if __name__ == "__main__":
    main()
