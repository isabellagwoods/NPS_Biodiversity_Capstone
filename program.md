Here's the AutoResearch instructions adapted for your capstone:

---

**AutoResearch — NPS Biodiversity Prediction**

**Setup**

To set up a new experiment, work with the user to:

1. Agree on a run tag: propose a tag based on today's date (e.g. `apr20`). The branch `autoresearch/<tag>` must not already exist.
2. Create the branch: `git checkout -b autoresearch/<tag>` from current master.
3. Read the in-scope files:
   - `README.md` — repository context
   - `prepare.py` — fixed constants, data loading, feature engineering, train/val split, evaluation. **Do not modify.**
   - `model.py` — the file you modify. Model architecture, hyperparameters, feature selection, preprocessing pipeline.
4. Verify data exists: Check that `data/processed/` contains the model-ready dataset. If not, tell the human to run `python prepare.py`.
5. Initialize `results.tsv`: Create with just the header row. Baseline will be recorded after the first run.
6. Confirm and go.

---

**Experimentation**

Run experiments simply as: `python run.py "description of experiment" > run.log 2>&1`

**What you CAN do:**

- Modify `model.py` — this is the only file you edit. Everything is fair game: model type, hyperparameters, feature selection, preprocessing, cross-validation strategy, ensembling.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It contains the fixed evaluation metric, data loading, train/val split, and feature definitions.
- Install new packages. Only use what's already available (sklearn, pandas, numpy, scipy).
- Modify the evaluation harness. The validation RMSE and R² computed in `prepare.py` are the ground truth metrics.

**The goal:** Get the lowest `val_rmse` while keeping `val_r2 >= 0.85`. Since the dataset is small (~200 parks), runs are fast — everything is fair game. Try different model families, feature subsets, transformations, and hyperparameters.

**VRAM / memory** is not a concern at this scale.

**Simplicity criterion:** All else being equal, simpler is better. A LinearRegression that hits the target beats a messy ensemble that barely improves it. When evaluating whether to keep a change, weigh complexity cost against improvement magnitude.

---

**The first run:** Always establish the baseline first — run `model.py` as-is (LinearRegression).

---

**Output format**

Once the script finishes it prints a summary like this:

```
---
val_rmse:    0.1050
val_r2:      0.8712
num_features: 12
model_type:  LinearRegression
num_parks:   187
```

Extract the key metrics:
```bash
grep "^val_rmse:\|^val_r2:" run.log
```

---

**Logging results**

Log to `results.tsv` (tab-separated, NOT comma-separated).

Header and columns:
```
commit	val_rmse	val_r2	status	description
```

- `commit`: git commit hash (short, 7 chars)
- `val_rmse`: achieved RMSE (e.g. `0.105000`) — use `0.000000` for crashes
- `val_r2`: achieved R² (e.g. `0.871200`) — use `0.000000` for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short text description of what this experiment tried

Example:
```
commit	val_rmse	val_r2	status	description
a1b2c3d	0.105000	0.8712	keep	baseline LinearRegression
b2c3d4e	0.098000	0.8891	keep	Ridge alpha=10 log-transform target
c3d4e5f	0.110000	0.8500	discard	Lasso removed too many features
d4e5f6g	0.000000	0.0000	crash	RandomForest OOM on CV grid search
```

---

**The experiment loop**

Suggested model search order:
1. `LinearRegression` — baseline
2. `Ridge` / `Lasso` / `ElasticNet` — regularization
3. `PolynomialFeatures` — interaction terms (visitation × biome)
4. `RandomForestRegressor`
5. `HistGradientBoostingRegressor` — handles NaN natively
6. `TransformedTargetRegressor` with log transform on SDI
7. Hyperparameter tuning on best model

**LOOP FOREVER:**

1. Look at git state: current branch/commit
2. Tune `model.py` with an experimental idea
3. `git commit`
4. Run: `python run.py "description" > run.log 2>&1`
5. Read results: `grep "^val_rmse:\|^val_r2:" run.log`
6. If grep is empty, run crashed — `tail -n 50 run.log` for stack trace
7. Log results to `results.tsv`
8. If `val_rmse` improved (lower) AND `val_r2 >= 0.85`: **keep**, advance the branch
9. If worse: `git reset` back to previous commit

**Timeout:** Each experiment should take well under 60 seconds. If a run exceeds 5 minutes, kill it and treat as failure.

**Crashes:** Fix trivial bugs and re-run. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP:** Once the loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, try combining previous near-misses, adding spatial features, trying different CV strategies, or more radical feature engineering. The loop runs until the human interrupts you, period.

---

*Success criterion: `val_rmse ≤ 0.10` and `val_r2 ≥ 0.85`*
