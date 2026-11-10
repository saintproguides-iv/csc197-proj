# Safe-Procrastination Predictor (CSC173 / CSC197) - task effort model

Predicts the **base effort hours** of a task from its text (DeBERTa-v3 fine-tuned for regression).
This is stage 1 of the project: the personalization layer (S, F, R and the red/yellow/green zone)
is built on top of this estimate.

## Where to start (5 steps)

1. **Clone and open the project**
   ```powershell
   git clone <repo-url>
   cd <repo-folder>
   git checkout sip-testing        # if the work is not merged into main yet
   ```
2. **Create and activate a virtual environment** (Python 3.10+; 3.12 or 3.13 recommended)
   ```powershell
   python -m venv venv
   venv\Scripts\activate           # the prompt should now start with (venv)
   ```
   If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
3. **Install packages**
   - NVIDIA GPU:
     ```powershell
     pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
     ```
   - No NVIDIA GPU / Mac: delete the three lines starting with `torch==`, `torchaudio==`,
     `torchvision==` from `requirements.txt`, then:
     ```powershell
     pip install -r requirements.txt
     pip install torch
     ```
4. **Train the model yourself** (the trained weights are not in the repo: they are too big for GitHub)
   ```powershell
   python train.py
   ```
   About 3-5 minutes on an NVIDIA GPU, much longer on CPU (set `NUM_EPOCHS = 2` in `train.py` for a quick test).
   The first run downloads DeBERTa (~700 MB). If the download times out, run the command again.
   When it finishes you will have a `saved_model/` folder, which stays on your own computer.
5. **Try it**
   ```powershell
   python predict.py
   ```
   Type a task (e.g. `Build a command-line calculator in Java`), press Enter, type `exit` to quit.

## What is in the repo

| File | Purpose |
|---|---|
| `train.py` | Fine-tunes DeBERTa on `data/sip_train.csv`, checks `data/sip_val.csv`, saves `saved_model/` |
| `predict.py` | Interactive predictor (loads `saved_model/`) |
| `evaluate_test.py` | One-time final evaluation on a held-out test CSV (needs `--final`) |
| `prepare_sip.py` | Rebuilds the train/val/test CSVs from the original SiP file |
| `prepare_survey.py` | Turns the Google Forms responses export into a survey test CSV |
| `data/sip_train.csv`, `sip_val.csv`, `sip_test.csv` | Cleaned SiP tasks split by task text (task_description, base_hours) |
| `data/survey_test.csv` | 20 CCS activities with survey hours (final test set) |
| `tasks_dataset_expanded.csv` | Old CCS dataset (no longer used for training) |

## Rules for testing

- **Tune only on the validation set** (`sip_val.csv`). Do not train on or look at the test files to change the model.
- `evaluate_test.py` refuses to run without `--final`, so each test set is scored once:
  ```powershell
  python evaluate_test.py --final
  python evaluate_test.py --final --csv data/survey_test.csv --name survey
  ```
- Do not commit `saved_model/`, `results/`, `venv/`, or `data/predictions_*.csv` (already in `.gitignore`).
