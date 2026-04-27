# AutoResearch — NPS Biodiversity SDI Prediction

## Overview
Autonomous ML experimentation loop. Goal: minimize `val_rmse` while keeping `val_r2 >= 0.85`.
The agent runs experiments, logs results, and iterates — without human intervention.

---

## Setup

1. Ensure `full_data_clean.csv` is in the working directory
2. Ensure `train.py`, `results.tsv`, and `experiment_log.md` are in the working directory
3. Install dependencies: `pip install scikit-learn pandas numpy`
4. Initialize git branch: `git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')`
5. Run baseline: `python train.py > run.log 2>&1`
6. Record baseline in `results.tsv`

---

## Success Criteria

| Metric | Target |
|--------|--------|
| `val_rmse` | ≤ 0.10 |
| `val_r2` | ≥ 0.85 |

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | **Only file the agent edits** |
| `full_data_clean.csv` | Input data — read only, never modify |
| `prepare.py` | Data prep — read only, never modify |
| `results.tsv` | Experiment log (TSV, not committed to git) |
| `experiment_log.md` | Detailed experiment notes |
| `run.log` | Output from last run |

---

## Running an Experiment

```bash
python train.py > run.log 2>&1
grep "^val_rmse:\|^val_r2:" run.log
```

Extract results:
```bash
grep "^val_rmse:" run.log
grep "^val_r2:"   run.log
grep "^best_params:" run.log
grep "^model_type:"  run.log
```

---

## results.tsv Format

Tab-separated. Do NOT use commas. Do NOT commit this file.

```
commit	val_rmse	val_r2	status	model_type	description
```

Example:
```
a1b2c3d	0.142300	0.7812	keep	LinearRegression	baseline
b2c3d4e	0.118000	0.8234	keep	Ridge alpha=10	ridge regularization
c3d4e5f	0.099000	0.8901	keep	RandomForest	random forest 100 trees
d4e5f6g	0.000000	0.0000	crash	HistGradientBoosting	OOM on large grid
```

Status options: `keep`, `discard`, `crash`

---

## Experiment Loop

**LOOP FOREVER until manually stopped:**

### Step 1 — Read state
```bash
git log --oneline -5
cat results.tsv
```

### Step 2 — Pick an experiment
Follow this search order:
1. `LinearRegression` — baseline
2. `Ridge` / `Lasso` / `ElasticNet` — regularization, try `alpha` in [0.01, 0.1, 1, 10, 100]
3. `PolynomialFeatures(degree=2)` + `Ridge` — interaction terms
4. `RandomForestRegressor` — n_estimators in [50, 100, 200]
5. `HistGradientBoostingRegressor` — handles NaN natively, try max_iter in [100, 200]
6. `TransformedTargetRegressor` with log transform on SDI
7. Feature selection: `SelectKBest` or drop low-importance features
8. Hyperparameter tuning on best model so far

### Step 3 — Edit train.py
Modify only the **Model definition** section (between the model definition comments).

### Step 4 — Commit
```bash
git add train.py
git commit -m "experiment: <short description>"
```

### Step 5 — Run
```bash
python train.py > run.log 2>&1
```

**Timeout: 5 minutes (300 seconds).** If run exceeds 10 minutes, kill it:
```bash
kill %1
```
Treat as crash, revert, move on.

### Step 6 — Read results
```bash
grep "^val_rmse:\|^val_r2:\|^best_params:\|^model_type:\|^total_seconds:" run.log
```

If grep returns nothing → crash. Read the error:
```bash
tail -n 50 run.log
```

### Step 7 — Log to results.tsv
```bash
echo -e "$(git rev-parse --short HEAD)\t<val_rmse>\t<val_r2>\t<status>\t<model_type>\t<description>" >> results.tsv
```

Also append to `experiment_log.md` with full details (params, notes, what you tried).

### Step 8 — Keep or discard
- **val_rmse improved AND val_r2 ≥ 0.85** → keep commit, advance branch
- **val_rmse worse OR val_r2 < 0.85** → discard:
```bash
git reset --hard HEAD~1
```
- **Crash** → log as crash, fix if trivial, otherwise skip and move on

---

## Hard Rules — NEVER VIOLATE

1. **DO NOT modify** `full_data_clean.csv`, `prepare.py`, or the evaluation block in `train.py`
2. **DO NOT change** how `val_rmse` or `val_r2` are calculated
3. **DO NOT install** new packages not already available
4. **DO NOT** stop and ask the human if you should continue
5. **DO NOT** run for more than 10 minutes per experiment — kill and treat as crash
6. **DO NOT** commit `results.tsv` or `experiment_log.md` to git
7. **val_rmse is the ONLY success metric** — do not substitute a different metric

---

## Crashes

- Typo or missing import → fix and re-run
- OOM → log crash, revert, try smaller model
- Fundamental idea broken → log crash, move on

---

## NEVER STOP

Once the loop starts, run indefinitely until manually interrupted.
If stuck, try: combining previous near-misses, more aggressive feature selection,
different CV strategies, or log-transforming the target.

The loop ends only when the human types Ctrl+C.
