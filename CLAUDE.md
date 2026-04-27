# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is an **AutoResearch** loop for a Northwestern STAT 390 Capstone project. The goal is to autonomously iterate on ML experiments to predict the **Shannon Diversity Index (SDI)** of US National Parks using ecological, climate, visitation, and human-impact features.

**Success criteria:**
- `val_rmse` ≤ 0.10
- `val_r2` ≥ 0.85

## Python Environment

```bash
conda activate vscode_env
# interpreter: /Users/isabellawoods/anaconda3/envs/vscode_env/bin/python
```

## Key Commands

```bash
# Run model and capture output
python train.py > run.log 2>&1

# Extract metrics from run
grep "^val_rmse:\|^val_r2:\|^best_params:\|^model_type:\|^total_seconds:" run.log

# Read last 50 lines on crash
tail -n 50 run.log

# Commit an experiment
git add train.py
git commit -m "experiment: <short description>"

# Log result to results.tsv
echo -e "$(git rev-parse --short HEAD)\t<val_rmse>\t<val_r2>\t<status>\t<model_type>\t<description>" >> results.tsv

# Discard a failed experiment
git reset --hard HEAD~1
```

## Architecture

The codebase has three active files the agent touches:

| File | Role |
|------|------|
| `train.py` | **Only file to edit.** Contains model definition, fit, and evaluation. |
| `prepare.py` | Read-only data prep — loads `full_data_clean.csv`, returns `X_train`, `X_val`, `y_train`, `y_val`. |
| `full_data_clean.csv` | Read-only. ~170 parks, one row per park. Target column: `SDI`. Drop `n_observations`, `n_species`. |

Supporting files (do not commit):
- `results.tsv` — TSV experiment log (tab-separated, never commas)
- `experiment_log.md` — human-readable notes per experiment
- `run.log` — stdout/stderr from most recent run

## AutoResearch Loop Rules

**HARD RULES — never violate:**
1. Only `train.py` may be edited.
2. Do NOT modify `full_data_clean.csv`, `prepare.py`, or the evaluation block in `train.py` (section 4 and 5).
3. Do NOT change how `val_rmse` or `val_r2` are calculated.
4. Do NOT install new packages.
5. Do NOT stop to ask if you should continue — loop indefinitely until Ctrl+C.
6. Do NOT commit `results.tsv` or `experiment_log.md`.
7. Each run must complete within 5 minutes (300s). Kill and treat as crash if it exceeds 10 minutes.

**Keep vs. discard logic:**
- `val_rmse` improved **AND** `val_r2 ≥ 0.85` → keep commit, advance
- `val_rmse` worse **OR** `val_r2 < 0.85` → `git reset --hard HEAD~1`, log as discard

## Model Search Order

Edit only the **model definition section** (section 2) in `train.py`:

1. `LinearRegression` — baseline
2. `Ridge` / `Lasso` / `ElasticNet` — alpha in [0.01, 0.1, 1, 10, 100]
3. `PolynomialFeatures(degree=2)` + `Ridge`
4. `RandomForestRegressor` — n_estimators in [50, 100, 200]
5. `HistGradientBoostingRegressor` — handles NaN natively, max_iter in [100, 200]
6. `TransformedTargetRegressor` with log transform on SDI
7. `SelectKBest` or drop low-importance features
8. Hyperparameter tuning on the best model so far

## results.tsv Format

```
commit	val_rmse	val_r2	status	model_type	description
```

Status values: `keep`, `discard`, `crash`
