# Safe-Procrastination Predictor (CSC173 / CSC197) - task effort model

Predicts the **base effort hours** of a task from its text (DeBERTa-v3 fine-tuned for regression).
This is stage 1 of the project: the personalization layer (S, F, R and the red/yellow/green zone)
is built on top of this estimate.

The final model is "version 2b": the task text is cleaned (lowercased, runs of 3 or more repeated
letters shrunk to one, spaces fixed) before the model reads it, and the model is trained on the
cleaned SiP training titles.

## Where to start (6 steps)

1. **Clone and open the project**
   ```powershell
   git clone <repo-url>
   cd <repo-folder>
   git checkout sip-testing        # only if the work is not merged into main yet
   ```
2. **Create and activate a virtual environment** (Python 3.11 or newer; 3.12 or 3.13 recommended)
   ```powershell
   python -m venv venv
   venv\Scripts\activate           # the prompt should now start with (venv)
   ```
   If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
3. **Install PyTorch first** (the command depends on your computer), **then the other packages**
   - NVIDIA GPU:
     ```powershell
     pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
     ```
   - No NVIDIA GPU (or Mac):
     ```powershell
     pip install torch==2.11.0
     ```
   Then, on every machine:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Check the data is there.** `data/` should contain `sip_train.csv`, `sip_val.csv`, `sip_test.csv`
   and `survey_test.csv`.
5. **Train the model yourself** (the trained weights are not in the repo: they are too big for GitHub)
   ```powershell
   python train.py --save-dir ./saved_model
   ```
   About 3 to 15 minutes on an NVIDIA GPU; much longer on CPU (add `--epochs 1` for a quick test).
   The first run downloads DeBERTa (about 700 MB). If the download times out, run the command again.
   At the end it prints the validation results and saves the model to `saved_model/`, which stays on
   your own computer.
6. **Try it**
   ```powershell
   python predict.py
   ```
   Type a task (for example `Build a command-line calculator in Java`), press Enter, type `exit` to quit.

   `predict.py` also reads counts such as "5 problems": it asks the model about one problem and scales
   the answer by n^0.5. The exponent 0.5 is an assumption, not fitted to data. Use `--alpha 0.8` to
   change it, or `--no-quantity` to turn the rule off.

## What is in the repo

| File                                                          | Purpose                                                                     |
| ------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `train.py`                                                    | Fine-tunes DeBERTa on `data/sip_train.csv`, checks `data/sip_val.csv`        |
| `predict.py`                                                  | Interactive predictor (loads `saved_model/`)                                |
| `inference.py`                                                | Loads a trained model and turns text into predicted hours                   |
| `text_utils.py`                                               | Text cleaning (used in training and prediction) and typo helpers            |
| `quantity.py`                                                 | Reads counts like "5 problems" and scales the estimate                      |
| `evaluate_test.py`                                            | One-time final evaluation on a held-out test CSV (needs `--final`)          |
| `robustness_check.py`                                         | Measures how much predictions move when the text is damaged                 |
| `check_bias.py`                                               | Checks whether predictions run high or low                                  |
| `augment_typos.py`                                            | Experiment: adds misspelled copies of training titles (it did not help)     |
| `data/sip_train.csv`, `data/sip_val.csv`, `data/sip_test.csv` | Cleaned SiP tasks split by task text (`task_description`, `base_hours`)     |
| `data/survey_test.csv`                                        | 20 CCS activities with survey hours (final test set)                        |
| `tasks_dataset_expanded.csv`                                  | Old CCS dataset (no longer used for training)                               |

## Optional checks (validation data only)

```powershell
python robustness_check.py --model-dir ./saved_model
python check_bias.py --model-dirs ./saved_model
```

## Rules for testing

- **Tune only on the validation set** (`sip_val.csv`). Do not train on, or look at, the test files to change the model.
- `evaluate_test.py` refuses to run without `--final`, so each test set is scored once:
  ```powershell
  python evaluate_test.py --final
  python evaluate_test.py --final --csv data/survey_test.csv --name survey
  ```
- Do not commit `saved_model*/`, `results/`, `venv/`, or generated files in `data/` such as
  `predictions_*.csv`, `robustness_*.csv` and `sip_train_aug.csv` (all listed in `.gitignore`).
