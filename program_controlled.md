# Controlled Experiment — Monthly Traffic Features in Stage 2
### Northwestern STAT 390 Capstone

---

## Purpose

Test one variable in 5 controlled steps: whether adding **monthly-resolution traffic/visitor data** (`monthly_traffic`, `annual_visitors` from `monthly_sdi.csv`) to Stage 2 improves prediction over the park-level annual averages currently used.

**Baseline model** (from autoresearch run, commit `c4a542d`):
- Stage 1: `SimpleImputer(median) → StandardScaler → SelectKBest(k=8) → Ridge(alpha=100)`
- Stage 2: `HistGradientBoostingRegressor(max_iter=500)` on top-8 park-level human features + time features
- Best: `s1_val_rmse=0.6441`, `s2_val_rmse=0.4736`

**Controlled variable:** `stage2_extra` — the monthly-level columns added from `monthly_sdi.csv`.  
Everything else is frozen.

---

## Files

| File | Role | Editable? |
|------|------|-----------|
| `train.py` | Model pipeline | ✅ Section 2 only |
| `full_data_clean.csv` | Park-level features | ❌ |
| `monthly_sdi.csv` | Monthly SDI + monthly_traffic + annual_visitors | ❌ |
| `results_controlled.tsv` | This experiment's log | Append only |
| `experiment_log.md` | Shared detailed log | Append only |
| `outputs/` | Plots and tables | Auto |

---

## Setup

```bash
conda activate vscode_env
# Initialize results_controlled.tsv if missing
echo -e "exp\tcommit\ts1_val_rmse\ts1_val_r2\ts1_cv_rmse\ts2_val_rmse\ts2_val_r2\ts1_model\ts2_model\ttotal_seconds\tstatus\tstage2_extra\tdescription" > results_controlled.tsv
```

Verify baseline runs cleanly before changing anything:
```bash
python train.py > run.log 2>&1
grep "^s1_val_rmse:\|^s2_val_rmse:\|^commit:" run.log
# Expected: s1_val_rmse ≈ 0.6441  s2_val_rmse ≈ 0.4736
```

---

## The 5 Controlled Experiments

All experiments keep Stage 1 frozen at `SelectKBest(k=8) + Ridge(alpha=100)`.  
Only `stage2_extra` in Section 2 changes.

| Exp | `stage2_extra` | What's being tested |
|-----|---------------|---------------------|
| CE-01 | `[]` | Baseline — no monthly cols (verify reproducibility) |
| CE-02 | `['annual_visitors']` | Monthly visitor counts (r=0.28, NaN=4%) |
| CE-03 | `['monthly_traffic']` | Monthly traffic counts (r=0.19, NaN=27%) |
| CE-04 | `['annual_visitors', 'monthly_traffic']` | Both monthly cols together |
| CE-05 | `['annual_visitors', 'monthly_traffic']` + remove `avg_annual_traffic` from `stage2_human_features` | Monthly cols replacing park-level traffic |

---

## Running Each Experiment

### Step 1 — Edit Section 2
Change only `stage2_extra` (and optionally `stage2_human_features` for CE-05):

```python
# CE-01 baseline
stage2_extra = []

# CE-02
stage2_extra = ['annual_visitors']

# CE-03
stage2_extra = ['monthly_traffic']

# CE-04
stage2_extra = ['annual_visitors', 'monthly_traffic']

# CE-05 (also remove avg_annual_traffic from stage2_human_features)
stage2_extra = ['annual_visitors', 'monthly_traffic']
stage2_human_features = [f for f in [
    'avg_recreation_visitors', 'max_recreation_visitors', 'visit_slope',
    'pop_density', 'traffic_cv', 'n_facilities', 'hours_per_visitor'
] if f in ALL_HUMAN_FEATURES]
```

### Step 2 — Commit
```bash
git add train.py
git commit -m "ctrl-exp CE-0N: stage2_extra=[...] — <one-line description>"
```

### Step 3 — Run
```bash
python train.py > run.log 2>&1
grep "^s1_val_rmse:\|^s2_val_rmse:\|^s1_val_r2:\|^s2_val_r2:\|^s1_cv_rmse:\|^total_seconds:\|^s1_model:\|^s2_model:\|^commit:" run.log
```

On crash:
```bash
tail -60 run.log
```

### Step 4 — Log to results_controlled.tsv
```bash
echo -e "CE-0N\t$(git rev-parse --short HEAD)\t<s1_rmse>\t<s1_r2>\t<s1_cv>\t<s2_rmse>\t<s2_r2>\tRidge\tHistGradientBoostingRegressor\t<seconds>\t<keep|discard>\t\"<stage2_extra>\"\t<description>" >> results_controlled.tsv
```

### Step 5 — Log to experiment_log.md
```markdown
## [CONTROLLED] CE-0N — YYYY-MM-DD
**Controlled variable:** stage2_extra
**stage2_extra:** [...]
**Commit:** abc1234
**Stage 1:** SelectKBest(k=8)+Ridge(alpha=100)   **Stage 2:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.XXXX   **s1_val_r2:** 0.XX   **s1_cv_rmse:** 0.XX
**s2_val_rmse:** 0.XXXX   **s2_val_r2:** 0.XX
**Status:** keep / discard
**Notes:** <observation>
```

### Step 6 — Keep or Discard
- **s2_val_rmse improved** → keep, continue
- **s2_val_rmse same or worse** → log discard, revert train.py to last kept state (`git reset --hard HEAD~1`)
- At end: if best experiment is better than baseline (`c4a542d`), commit the winning train.py

---

## RMSE-Over-Time Plot

After all 5 experiments, generate the plot:

```python
import pandas as pd, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')

r = pd.read_csv('results_controlled.tsv', sep='\t')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Controlled Experiment — Stage 2 Monthly Traffic Features\n(Stage 1 frozen at SelectKBest(k=8)+Ridge(α=100))', fontsize=12)

for ax, col, label, color in [
    (axes[0], 's1_val_rmse', 'Stage 1 — Ecological', 'steelblue'),
    (axes[1], 's2_val_rmse', 'Stage 2 — Monthly SDI', 'darkorange'),
]:
    ax.plot(range(len(r)), r[col], 'o-', color=color, label=label, markersize=8)
    for i, (x, y, lbl) in enumerate(zip(range(len(r)), r[col], r['exp'])):
        ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    ax.axhline(0.20, color='red', linestyle=':', linewidth=1.5, label='Target (0.20)')
    ax.axhline(r.iloc[0][col], color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Baseline')
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels(r['exp'], rotation=20)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('val_rmse')
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/controlled_rmse_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/controlled_rmse_over_time.png")
```

---

## Commit Decision

After 5 experiments:

```bash
# Check if best CE beats baseline c4a542d (s2_val_rmse=0.4736)
python -c "
import pandas as pd
r = pd.read_csv('results_controlled.tsv', sep='\t')
best = r.loc[r['s2_val_rmse'].idxmin()]
baseline_s2 = 0.4736
print(f'Best CE: {best[\"exp\"]}  s2={best[\"s2_val_rmse\"]:.4f}  (baseline={baseline_s2})')
if best['s2_val_rmse'] < baseline_s2:
    print('IMPROVEMENT — commit winning train.py')
else:
    print('No improvement — revert to baseline c4a542d')
"
```

If improved:
```bash
git add train.py
git commit -m "ctrl-exp: best config from controlled experiment — stage2_extra=[...] (s2 improved X.XXX→X.XXX)"
```

If not improved:
```bash
git checkout c4a542d -- "Split Model Autoresearch/train.py"
git add train.py
git commit -m "ctrl-exp: revert to baseline c4a542d — no monthly feature improvement found"
```

---

## Token Budget Rule

**Check token usage before starting each experiment.** If you estimate you have used ≥ 90% of your context budget:

1. Skip remaining experiments
2. Generate the RMSE-over-time plot with whatever data exists in `results_controlled.tsv`
3. Write a brief note in `experiment_log.md` under the heading `## [CONTROLLED] Wrap-Up — Token Budget`
4. Run the commit decision check and commit/revert accordingly

---

## Hard Rules (inherited from program.md)

1. Only Section 2 of `train.py` may be edited.
2. Never touch `EDA.ipynb`, `full_data_clean.csv`, or `monthly_sdi.csv`.
3. Never modify past rows in any TSV. Append only.
4. Never stop to ask the human if you should continue.
5. Do not commit `results_controlled.tsv`, `experiment_log.md`, or `run.log`.
6. Kill any run exceeding 12 minutes. Treat as crash.
7. All log entries for this experiment must include `[CONTROLLED]` in the heading.

---

## Expected Deliverables

```
outputs/
  controlled_rmse_over_time.png   ← required
  experiment_result_matrix.csv    ← auto-regenerated by train.py
results_controlled.tsv            ← 5 rows (do not commit)
experiment_log.md                 ← 5 [CONTROLLED] entries appended
```
