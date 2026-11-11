import argparse
import sys

from inference import Predictor
from quantity import estimate_with_quantity

def prompt_survey_score(prompt_text: str) -> float:
    while True:
        try:
            val = float(input(prompt_text).strip())
            if 1 <= val <= 5:
                return val
            print("  [!] Please enter a number between 1 and 5.")
        except ValueError:
            print("  [!] Invalid input. Please enter a numerical score (1-5).")

def calculate_personalized_hours(base_hours: float, speed_score: float, friction_score: float, resilience_score: float) -> tuple[float, dict]:
    
    #Adjust base estimated hours using personal behavioral parameters (SFR 1-5) and 3 is neutral average baseline.
    #Constants used are assumed and can be changed once research backed values appear(0.20, 0.10, 0.075)

    #Execution Speed Factor (S_factor): Higher score -> faster execution -> lower hours
    #Range: S=1 -> 1.40x (+40%), S=3 -> 1.00x, S=5 -> 0.60x (-40%)
    s_factor = 1.0 - 0.20 * (speed_score - 3.0)

    #Task Friction / Avoidance Factor (F_factor): Higher score -> high friction -> higher total hours
    #Range: F=1 -> 0.80x (-20%), F=3 -> 1.00x, F=5 -> 1.20x (+20%)
    f_factor = 1.0 + 0.10 * (friction_score - 3.0)

    #Resilience / Stress Under Deadline Factor (R_factor): Higher score -> stays composed -> lower hours
    #Range: R=1 -> 1.15x (+15%), R=3 -> 1.00x, R=5 -> 0.85x (-15%)
    r_factor = 1.0 - 0.075 * (resilience_score - 3.0)

    #Combined Multiplier
    #Didn't used additive cause low level task will suffer
    combined_multiplier = s_factor * f_factor * r_factor
    personalized_hours = base_hours * combined_multiplier

    factors_info = {
        "s_factor": s_factor,
        "f_factor": f_factor,
        "r_factor": r_factor,
        "combined_multiplier": combined_multiplier
    }
    return max(0.1, round(personalized_hours, 2)), factors_info

def prompt_deadline_hours() -> float:
    while True:
        try:
            val = float(input("Enter hours remaining until deadline (e.g., 24 or 4.5) > ").strip())
            if val > 0:
                return val
            print("  [!] Deadline hours must be greater than 0.")
        except ValueError:
            print("  [!] Invalid input. Please enter a valid number of hours.")

def determine_task_urgency(personalized_hours: float, time_remaining: float) -> tuple[str, float]:
    
    ratio = personalized_hours / time_remaining

    if ratio >= 0.70 or personalized_hours > time_remaining:
        status = "URGENT"
    elif ratio >= 0.30:
        status = "PACE YOURSELF"
    else:
        status = "CAN BE PUSHED BACK"

    return status

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

    print("Personal Profile Assessment (1 = Lowest, 5 = Highest)")
    speed_score = prompt_survey_score("1. How fast do you complete tasks compared to your classmates? (1-5): ")
    friction_score = prompt_survey_score("2. I tend to delay important tasks when they feel overwhelming/difficult (1-5): ")
    resilience_score = prompt_survey_score("3. I maintain my work quality and avoid freezing when approaching deadlines (1-5): ")

    print("Proceed to enter your task")
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

            time_remaining = prompt_deadline_hours()

            if args.no_quantity:
                hours = predictor.predict_one(user_input)
                seen = predictor.prepare(user_input)
                if seen != user_input:
                    print(f" (read as: \"{seen}\")")
            else: 
                r = estimate_with_quantity(predictor.predict_one, user_input, args.alpha)
                hours = float(r['total'])
                if r["used_rule"]:
                    print(f" Quantity found: {r['n']} x \"{r['noun']}\"")
                    print(f" One unit (\"{r['unit_text']}\"): {r['unit_hours']} hours")
                    print(f" Total = {r['unit_hours']} x {r['n']}^{r['alpha']} = {r['total']} hours")
                    print(f" (the exponent {r['alpha']} is an assumption, not fitted to data)")
                else:
                    seen = predictor.prepare(user_input)
                    if seen != user_input:
                        print(f" (read as: \"{seen}\")")

            # Adjust baseline effort using personal assessment scores
            adjusted_hours, factors = calculate_personalized_hours(hours, speed_score, friction_score, resilience_score)
            status = determine_task_urgency(adjusted_hours, time_remaining)

            #For demo purposes
            print(f"S = {factors["s_factor"]} = 1.0 - 0.20 * ({speed_score} - 3.0)")
            print(f"F = {factors["f_factor"]} = 1.0 + 0.10 * ({friction_score} - 3.0)")
            print(f"R = {factors["r_factor"]} = 1.0 - 0.075 * ({resilience_score} - 3.0)")
            print(f"Personalised = {hours:.2f} * {factors["s_factor"]} * {factors["f_factor"]} * {factors["r_factor"]}")

            print(f" Personalized Estimated Time : {adjusted_hours:.2f} hours")
            print(f" Estimated Base Effort       : {hours:.2f} hours")
            print(f" Priority Status             : {status}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated. Goodbye!")
            break


if __name__ == "__main__":
    main()
