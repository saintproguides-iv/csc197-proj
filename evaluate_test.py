"""
evaluate_test.py - ONE-TIME final evaluation on a held-out test CSV.

Use it only when the model is final. It refuses to run without --final, so the
test set is not touched by accident.

Examples
  python evaluate_test.py --final                                          # SiP test, ./saved_model
  python evaluate_test.py --final --model-dir ./saved_model_v2
  python evaluate_test.py --final --csv data/survey_test.csv --name survey

Input CSV needs two columns: task_description, base_hours
Needs: a trained model folder (from train.py) and data/sip_train.csv (for the baseline).
Writes: data/predictions_<name>.csv   (inside data/, which is git-ignored)
The model's own input cleaning (if it was trained with it) is applied automatically.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Final evaluation on a held-out test CSV.")
    p.add_argument("--final", action="store_true",
                   help="required: confirms this is the one-time final evaluation")
    p.add_argument("--csv", default="data/sip_test.csv", help="test CSV to evaluate")
    p.add_argument("--name", default="sip_test", help="label used in the output file name")
    p.add_argument("--model-dir", default="./saved_model")
    p.add_argument("--force-clean", action="store_true",
                   help="for older models: collapse repeated letters and spaces before predicting")
    p.add_argument("--train-csv", default="data/sip_train.csv",
                   help="used only to compute the 'always guess the median' baseline")
    p.add_argument("--top", type=int, default=15, help="how many largest errors to print")
    return p.parse_args()


def summarize(df, pred, baseline_hours, name, top=15):
    """df has task_description + base_hours; pred is a numpy array of predicted hours."""
    out = df.copy()
    out["pred_hours"] = np.round(pred, 2)
    out["abs_error"] = (out["pred_hours"] - out["base_hours"]).abs()
    mae = float(out["abs_error"].mean())
    rmse = float(np.sqrt(((out["pred_hours"] - out["base_hours"]) ** 2).mean()))
    base_mae = float((out["base_hours"] - baseline_hours).abs().mean())
    ratio = out["pred_hours"] / out["base_hours"].clip(lower=0.05)
    within2x = float(((ratio >= 0.5) & (ratio <= 2.0)).mean())

    print(f"\n=== {name}: results ({len(out)} rows) ===")
    print(f"MAE : {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"Baseline (always guess {baseline_hours:.2f} h) MAE: {base_mae:.2f} hours")
    print(f"Predictions within 2x of the true hours: {within2x * 100:.0f}%")
    print(f"\nLargest {top} errors:")
    cols = ["task_description", "base_hours", "pred_hours", "abs_error"]
    print(out.sort_values("abs_error", ascending=False)[cols].head(top).to_string(index=False))
    return out, {"mae": mae, "rmse": rmse, "baseline_mae": base_mae, "within_2x": within2x}


def main():
    args = parse_args()
    if not args.final:
        sys.exit(
            "Refusing to run: the test set is for ONE final evaluation.\n"
            "If the model is final, rerun with --final."
        )
    for path in (args.csv, args.train_csv):
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Cannot find {path}")
    if not os.path.isdir(args.model_dir):
        sys.exit(f"[ERROR] Cannot find {args.model_dir}. Run train.py first.")

    df = pd.read_csv(args.csv)
    if not {"task_description", "base_hours"}.issubset(df.columns):
        sys.exit("[ERROR] CSV needs columns: task_description, base_hours")
    baseline = float(pd.read_csv(args.train_csv)["base_hours"].median())

    from inference import Predictor
    predictor = Predictor(args.model_dir, force_clean=args.force_clean)
    print(f"Loaded {args.model_dir} on {predictor.device} ({predictor.describe()})")
    pred = predictor.predict(df["task_description"].astype(str).tolist())
    out, _ = summarize(df, pred, baseline, args.name, args.top)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", f"predictions_{args.name}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path} (keep it out of Git).")


if __name__ == "__main__":
    main()
