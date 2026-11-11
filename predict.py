import argparse
import sys

from inference import Predictor
from quantity import estimate_with_quantity


def main():
    parser = argparse.ArgumentParser(description="Interactive task-hours predictor.")
    parser.add_argument("--model-dir", default="./saved_model",
                        help="folder made by train.py (for example ./saved_model_v2b)")
    parser.add_argument("--force-clean", action="store_true",
                        help="for older models: collapse repeated letters and spaces before predicting")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="quantity exponent: total = one-unit hours x n^alpha (an assumption, default 0.5; 1.0 = every unit takes equally long)")
    parser.add_argument("--no-quantity", action="store_true",
                        help="turn the quantity rule off (the model alone, as in the test results)")
    args = parser.parse_args()

    print(f"Loading fine-tuned model from {args.model_dir}...")
    try:
        predictor = Predictor(args.model_dir, force_clean=args.force_clean)
    except Exception as e:
        print(f"\n[ERROR] Failed to load model from '{args.model_dir}': {e}")
        print("Ensure you have run 'python train.py' to generate model weights first.")
        sys.exit(1)
    rule = "off" if args.no_quantity else f"on, exponent {args.alpha}"
    print(f"Model loaded successfully on {predictor.device} ({predictor.describe()}; quantity rule {rule})\n")

    print("SAFE-PROCRASTINATION PREDICTOR - INTERACTIVE CLI")
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

            if args.no_quantity:
                hours = predictor.predict_one(user_input)
                seen = predictor.prepare(user_input)
                if seen != user_input:
                    print(f" (read as: \"{seen}\")")
                print(f" Estimated Base Effort: {hours} hours\n")
                continue

            r = estimate_with_quantity(predictor.predict_one, user_input, args.alpha)
            if r["used_rule"]:
                print(f" Quantity found: {r['n']} x \"{r['noun']}\"")
                print(f" One unit (\"{r['unit_text']}\"): {r['unit_hours']} hours")
                print(f" Total = {r['unit_hours']} x {r['n']}^{r['alpha']} = {r['total']} hours")
                print(f" (the exponent {r['alpha']} is an assumption, not fitted to data)")
                print(f" Estimated Base Effort: {r['total']} hours\n")
            else:
                seen = predictor.prepare(user_input)
                if seen != user_input:
                    print(f" (read as: \"{seen}\")")
                print(f" Estimated Base Effort: {r['total']} hours\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated. Goodbye!")
            break


if __name__ == "__main__":
    main()
