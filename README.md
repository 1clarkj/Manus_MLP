# Manus Glove Gesture ML + Pi Maze Control

## Purpose

This repository contains the full pipeline to:

1. Convert raw Manus glove finger-joint data into ML-ready features.
2. Train and tune an MLP classifier for hand-gesture recognition.
3. Map predicted gestures to control commands.
4. Send target positions to a Raspberry Pi and display predicted vs. actual motion in a GUI.

Current gesture classes:
- `pointing`
- `thumbs_up`
- `okay_sign` 

`neutral_position.csv` is intentionally not used in the training pipeline.

## Repository Layout

```text
ML/
  artifacts/                 # generated models + metadata
  data/
    raw/                     # source CSV recordings
    processed/               # generated train/val/test splits
  src/
    preprocess_data.py       # build splits + scaler from raw CSVs
    tune_and_train.py        # hyperparameter tuning + final training
    model_evaluate.py        # evaluate saved model on val/test split
    model_evaluate_real_time.py # UDP live inference only (no GUI)
    run_gui_real_time.py     # runtime GUI + UDP communication to Pi
    udp_debug_receiver_pi.py # optional UDP debug helper
  .gitignore
  LICENSE
  README.md
  requirements.txt
```

## Data Inputs

Place raw gesture CSVs in `data/raw/`.

The preprocessing script resolves these classes by filename/keywords:

- Pointing: e.g. `position1_pointing.csv`
- Thumbs up: e.g. `position2_thumbs_up.csv`
- Okay sign: e.g. `position3_okay_sign.csv` (preferred), or `okay_sign`/`reset` variants

## Reproducible Pipeline

Run these commands from repo root.

### 1) Environment setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Build processed datasets and scaler

```powershell
python src/preprocess_data.py
```

Creates:
- `data/processed/train_data.csv`
- `data/processed/val_data.csv`
- `data/processed/test_data.csv`
- `artifacts/minmax_scaler.pkl`
- `artifacts/class_mapping.json`

### 3) Hyperparameter tuning and final model training

```powershell
python src/tune_and_train.py
```

What this does:
1. Tunes MLP hyperparameters on the validation set.
2. Selects best config by validation macro-F1 (accuracy tie-break).
3. Retrains best model on `train + val`.
4. Evaluates once on unseen test split.

Creates:
- `artifacts/mlp_model_final.pkl` (primary trained model)
- `artifacts/mlp_model.pkl` (compatibility copy for runtime script)
- `artifacts/best_params.json`

### 4) Optional standalone evaluation

```powershell
python src/model_evaluate.py --split test
python src/model_evaluate.py --split val
```

### 5) Run the real-time GUI + Pi communication

```powershell
python src/run_gui_real_time.py
```

## Real-Time Model-Only Test (No GUI)

This mode uses the same UDP hand-data stream as the GUI (`127.0.0.1:5005`) but only runs model inference in real time.
It prints:

- predicted left-hand gesture + confidence
- predicted right-hand gesture + confidence
- mapped command `(x, y)` used by runtime logic

Run:

```powershell
python src/model_evaluate_real_time.py
```

Optional arguments:

```powershell
python src/model_evaluate_real_time.py --ip 127.0.0.1 --port 5005 --threshold 0.93
```


## License

Licensed under Apache License 2.0. See `LICENSE`.
