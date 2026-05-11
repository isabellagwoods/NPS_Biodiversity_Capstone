# AutoResearch — Two-Stage NPS Biodiversity Model
### Northwestern STAT 390 Capstone

---

## Overview

Autonomous ML experimentation loop for a **two-stage biodiversity prediction model**:

- **Stage 1** — Predict Shannon Diversity Index (SDI) from ecological features (park-level, static)
- **Stage 2** — Predict the SDI residual from human impact features (monthly, longitudinal)

The agent runs experiments, logs results, generates deliverables, and iterates without human intervention.

---

## Success Criteria & Stop Condition

| Stage | Metric | Target |
|-------|--------|--------|
| Stage 1 — Ecological Baseline | `s1_val_rmse` | ≤ 0.20 |
| Stage 2 — Human Impact Residual | `s2_val_rmse` | ≤ 0.20 |

**Stop when ANY of these is true:**
- Both `s1_val_rmse ≤ 0.20` AND `s2_val_rmse ≤ 0.20`
- 50 experiments completed
- 5 hours total elapsed

Check stop condition after every run:
```bash
python -c "
import pandas as pd
r = pd.read_csv('results.tsv', sep='\t')
s1 = r['s1_val_rmse'].min()
s2 = r['s2_val_rmse'].dropna().min() if r['s2_val_rmse'].notna().any() else 999
print(f'Best s1={s1:.4f}  s2={s2:.4f}  n_exp={len(r)}')
if s1 <= 0.20 and s2 <= 0.20: print('TARGETS MET — STOP')
elif len(r) >= 50: print('50 EXPERIMENTS — STOP')
else: print('Continue')
"
```

---

## Files

| File | Role | Editable? |
|------|------|-----------|
| `train.py` | Model definitions + full pipeline | ✅ **Section 2 only** |
| `full_data_clean.csv` | Park-level features + SDI | ❌ Read-only |
| `monthly_sdi.csv` | Monthly SDI (auto-built first run) | ❌ Read-only after creation |
| `EDA.ipynb` | Source notebook | ❌ **Never touch** |
| `results.tsv` | Experiment log | ❌ Append only |
| `experiment_log.md` | Detailed notes | Append only |
| `outputs/` | Auto-generated plots and tables | Auto |

---

## Setup

```bash
conda activate vscode_env
git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')
python train.py > run.log 2>&1
grep "^s1_val_rmse:\|^s2_val_rmse:\|^total_seconds:" run.log
```

Initialize `results.tsv` header if missing:
```bash
echo -e "commit\ts1_val_rmse\ts1_val_r2\ts1_cv_rmse\ts2_val_rmse\ts2_val_r2\ts1_model\ts2_model\ttotal_seconds\tstatus\tdescription" > results.tsv
```

---

## Running an Experiment

```bash
python train.py > run.log 2>&1
grep "^s1_val_rmse:\|^s2_val_rmse:\|^s1_val_r2:\|^s2_val_r2:\|^s1_cv_rmse:\|^total_seconds:\|^s1_model:\|^s2_model:" run.log
```

On crash:
```bash
tail -n 60 run.log
```

Kill a hung run (>12 min):
```bash
kill %1
```

---

## Logging

### Append to results.tsv after every run
```bash
echo -e "$(git rev-parse --short HEAD)\t<s1_rmse>\t<s1_r2>\t<s1_cv>\t<s2_rmse>\t<s2_r2>\t<s1_model>\t<s2_model>\t<seconds>\t<status>\t<description>" >> results.tsv
```

Status: `keep`, `discard`, `crash`

### Append to experiment_log.md after every run
```markdown
## Experiment NNN — YYYY-MM-DD
**Commit:** abc1234
**Stage 1:** Ridge(alpha=10)   **Stage 2:** HistGBM(max_iter=100)
**s1_val_rmse:** 0.2341   **s1_r2:** 0.61   **s1_cv_rmse:** 0.2518
**s2_val_rmse:** 0.3812   **s2_r2:** 0.40
**Status:** keep
**What changed:** Increased Stage 1 alpha from 1 to 10
**Notes:** CV gap narrowed. Stage 2 still high — monthly data sparse.
```

---

## Experiment Loop

**LOOP UNTIL stop condition:**

### Step 1 — Read state
```bash
git log --oneline -5
tail -5 results.tsv
cat outputs/park_residuals.csv | head -10  # worst-performing parks
```

### Step 2 — Pick one thing to change
Change **one thing at a time** — model type OR hyperparameter OR feature set, never all three.

**Stage 1 search order:**
1. `Ridge(alpha=...)` — alpha in [0.01, 0.1, 1, 10, 100]
2. `Lasso(alpha=...)` — sparse selection
3. `ElasticNet(alpha=..., l1_ratio=...)`
4. `RandomForestRegressor(n_estimators=100, max_depth=5)`
5. `HistGradientBoostingRegressor(max_iter=200)` — handles NaN
6. Log-transform: `from sklearn.compose import TransformedTargetRegressor` with `func=np.log1p`
7. Feature trimming: reduce `stage1_features` to top 10 by permutation importance
8. `PolynomialFeatures(degree=2, interaction_only=True)` + Ridge

**Stage 2 search order:**
1. `Ridge(alpha=...)` with time features (month_sin/cos, year_norm)
2. `HistGradientBoostingRegressor` — best for sparse monthly data
3. Trim `stage2_features` to only traffic + top 5 pollution vars
4. Add latitude/longitude from poi as features
5. Predict `SDI_monthly` directly (remove residual framing)
6. Add park-level ecological features as Stage 2 controls

**Cross-stage ideas:**
- Use Stage 1 prediction as a feature in Stage 2
- Single joint model on merged park × month data
- Separate models by park size (bbox_area from poi)

### Step 3 — Edit train.py Section 2 only

### Step 4 — Commit
```bash
git add train.py
git commit -m "exp: <one-line description of single change>"
```

### Step 5 — Run (5 min limit per stage, 10 min total)
```bash
python train.py > run.log 2>&1
```

### Step 6 — Parse results
```bash
grep "^s1_val_rmse:\|^s2_val_rmse:\|^s1_model:\|^s2_model:\|^total_seconds:" run.log
```

### Step 7 — Keep or discard
- **Either stage improved** → keep, advance
- **Both stages worse** → `git reset --hard HEAD~1`, log discard
- **Crash** → fix trivial bugs + re-run; if broken idea, log crash + move on

### Step 8 — Generate deliverables (every 5 experiments)
```bash
# experiment-result matrix
python -c "
import pandas as pd
r = pd.read_csv('results.tsv', sep='\t')
r.to_csv('outputs/experiment_result_matrix.csv', index=False)
print(r[['commit','s1_val_rmse','s2_val_rmse','s1_model','s2_model','status','description']].to_string())
"
```

Metric-over-time plot and residual diagnostics are auto-generated by train.py.

### Step 9 — Check stop condition (see top of this file)

---

## Hard Rules — NEVER VIOLATE

1. **Only Section 2 of `train.py` may be edited.** Sections 0, 1, 4, 5 are frozen.
2. **Never touch `EDA.ipynb`.** This is the source notebook.
3. **Never modify `full_data_clean.csv` or `monthly_sdi.csv`.**
4. **Never change how `s1_val_rmse` or `s2_val_rmse` are calculated** (Section 4 of train.py).
5. **Never install new packages.**
6. **Never modify past rows in `results.tsv`.** Append only.
7. **Never stop to ask the human if you should continue.**
8. **Kill any run exceeding 12 minutes.** Treat as crash.
9. **Do not commit `results.tsv`, `experiment_log.md`, or `run.log`.**

---

## Required Deliverables

### 1. Controlled Experiment Set
**File:** `experiment_log.md`
Change one variable per experiment. Document in commit message + log.

Example controlled sequence:
```
Exp 01: Ridge alpha=1.0   (baseline)
Exp 02: Ridge alpha=10    (alpha only)
Exp 03: Ridge alpha=100   (alpha only)
Exp 04: RandomForest n=50 (model type only)
Exp 05: RandomForest n=100 (n_estimators only)
```

### 2. Experiment-Result Matrix
**File:** `outputs/experiment_result_matrix.csv`
Auto-generated — run the command in Step 8 every 5 experiments.

### 3. Metric-Over-Time Plot
**File:** `outputs/metric_over_time.png`
Auto-generated by train.py after every run (requires ≥2 rows in results.tsv).

### 4. Error Taxonomy
**File:** `outputs/error_taxonomy.md`

Write manually after reviewing experiment_log.md. Use this template:

```markdown
# Error Taxonomy

| Error Type | Description | How to Detect | Fix |
|------------|-------------|---------------|-----|
| Underfitting | High train+val RMSE | s1_val_rmse > 0.35 | Try RandomForest, add features |
| Overfitting | val_rmse >> cv_rmse | cv_rmse - val_rmse > 0.05 | Regularize, reduce features |
| Data sparsity (Stage 2) | Too few monthly rows | Stage 2 skipped | Lower n_obs threshold, use HistGBM |
| Fill value contamination | -9999/-4999 in features | Huge negative predictions | Verify NaN replacement |
| Feature leakage | ID cols in model | R²=0.99 | Check drop_cols |
| Runtime crash | OOM or timeout | run.log empty or truncated | Reduce grid, use HistGBM |
| Stage 2 unavailable | monthly_sdi.csv missing | s2_val_rmse=nan | Check iNaturalist file paths |
```

### 5. Failure Analysis Memo
**File:** `outputs/failure_analysis_memo.md`

Write manually. One page. Template:

```markdown
# Failure Analysis Memo
**Project:** NPS Biodiversity SDI Prediction
**Date:** <date>
**Author:** Isabella Woods

## Dominant Failure Mode
<One sentence: the main reason targets are not met>

## Evidence
- Best s1_val_rmse: <value>  (target 0.20)
- Best s2_val_rmse: <value>  (target 0.20)
- Experiments run: <N>
- Models tried: <list>

## Root Cause
<2-3 sentences>

## What I Tried
| Approach | s1_rmse | s2_rmse | Why It Didn't Work |
|----------|---------|---------|-------------------|
| Ridge alpha=1 | 0.31 | nan | Too simple |
| ... | ... | ... | ... |

## Next Steps
1. <specific fix>
2. <specific fix>
```

---

## Submission Checklist

```
outputs/
  experiment_result_matrix.csv   ← deliverable 2
  metric_over_time.png           ← deliverable 3
  error_taxonomy.md              ← deliverable 4
  failure_analysis_memo.md       ← deliverable 5
  residual_diagnostics.png       ← supplementary
  park_residuals.csv             ← supplementary
experiment_log.md                ← deliverable 1 evidence
```

**Short description for submission:**
> "Experiments were controlled by changing one variable per commit — model type, a single hyperparameter, or the feature set, never simultaneously. The error taxonomy identifies seven failure modes: underfitting from small n (~170 parks), fill value contamination from NASA satellite data (−9999/−4999 not yet replaced), Stage 2 data sparsity from iNaturalist rate-limiting gaps, overfitting when feature count exceeds park count, feature leakage from identifier columns, runtime crashes from oversized hyperparameter grids, and Stage 2 unavailability when monthly_sdi.csv is not built."
