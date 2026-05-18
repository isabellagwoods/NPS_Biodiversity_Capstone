# AutoResearch — NPS Biodiversity SDI Prediction
### Northwestern STAT 390 Capstone

---

## Current Model (as of Exp 070)

**Architecture:** Single-stage ElasticNet regression  
**Target:** `SDI_rarefied` — Shannon Diversity Index computed on 50-observation rarefied subsample per (park × month × taxon_group)  
**Data:** 2021–2025 iNaturalist observations, ~8,824 rows, 149 parks, 13 taxon groups  
**Split:** Park-based — all months × taxa from one park stay in the same split (108 train parks / 27 val parks)

**Current best:** `val_rmse = 0.2435`, `val_r2 = 0.768`, `cv_rmse = 0.211`  
**Target:** `val_rmse ≤ 0.20`  
**Experiments run:** 71 (032–071 on rarefied SDI model; 001–031 on prior two-stage model)

---

## Model Architecture

```python
ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000)
```

Wrapped in a `ColumnTransformer` pipeline:
- **Numeric features (76):** `SimpleImputer(median)` → `StandardScaler`
- **Categorical (taxon_group, 13 groups):** `SimpleImputer(constant)` → `OneHotEncoder`

---

## Feature Set (76 numeric + 1 categorical)

| Group | Features | Count |
|-------|----------|-------|
| Ecological (ECO) | ET_range, GPP_range, avg_FPAR, max_FPAR, FPAR_range, avg_SNOW, max_SNOW, SNOW_range, n_burn_observations, pct_burned | 10 |
| Human impact (HUM) | avg_recreation_visitors, max_recreation_visitors, visit_slope, pop_density, avg_annual_traffic, traffic_cv, n_facilities, hours_per_visitor | 8 |
| Geographic (GEO) | latitude, longitude | 2 |
| Temporal (TIME) | month_sin, month_cos, year_norm | 3 |
| Traffic (TRAF) | monthly_traffic, annual_visitors | 2 |
| Observation | log_n_obs | 1 |
| AR lags | SDI_lag1, SDI_lag2, SDI_lag3, SDI_lag6, SDI_lag9, SDI_lag12 | 6 |
| Baselines | SDI_cumean (all-time), SDI_month_cumean (same-month prior years) | 2 |
| Deviations | SDI_dev (=lag1−month_cumean), SDI_dev12 (=lag12−month_cumean) | 2 |
| Reliability | n_prior_months (count of prior observations for this park×taxon) | 1 |
| Taxon×season | tg_{taxon}_sin, tg_{taxon}_cos for 13 taxon groups | 26 |
| Taxon×trend | tg_{taxon}_year for 13 taxon groups | 13 |
| **Categorical** | taxon_group (OHE, 13 groups) | — |

---

## Success Criterion & Stop Condition

**Target:** `val_rmse ≤ 0.20`

**Stop when ANY is true:**
- `val_rmse ≤ 0.20`
- 100 experiments completed
- 5 hours total elapsed

Check after every run:
```bash
python -c "
import pandas as pd
r = pd.read_csv('results.tsv', sep='\t')
best = r['s1_val_rmse'].min()
n = len(r)
print(f'Best val_rmse={best:.4f}  n_exp={n}')
if best <= 0.20: print('TARGET MET — STOP')
else: print(f'Gap to target: {best-0.20:.4f}  Continue')
"
```

---

## Files

| File | Role | Editable? |
|------|------|-----------|
| `train.py` | Model + full pipeline | **Section 2 only** |
| `full_data_clean.csv` | Park-level features | Read-only |
| `monthly_sdi.csv` | Monthly SDI | Read-only after creation |
| `outputs/cache/taxon_monthly_sdi_rarefied_n50.csv` | Rarefied SDI cache | Read-only after creation |
| `results.tsv` | Experiment log | Append only |
| `experiment_log.md` | Detailed notes | Append only |
| `outputs/` | Auto-generated plots and tables | Auto |

---

## Running an Experiment

```bash
conda run -n vscode_env python train.py 2>&1 | tee run.log
grep "^s1_val_rmse:\|^s1_val_r2:\|^s1_cv_rmse:\|^total_seconds:\|^s1_model:" run.log
```

---

## Experiment Protocol

### Per-experiment steps

1. **Read state:** `git log --oneline -5` + `tail -3 results.tsv`
2. **Pick ONE change:** model type, hyperparameter, OR feature set — never more than one
3. **Edit Section 2 of train.py**
4. **Run:** `conda run -n vscode_env python train.py 2>&1 | grep -E "^s1_val_rmse:|^s1_val_r2:|^s1_cv_rmse:|^total_seconds:"`
5. **Decide:**
   - `val_rmse` improved → **keep**: `git add train.py && git commit -m "expNNN: <description>"`
   - `val_rmse` same or worse → **discard**: `git checkout -- train.py`
6. **Log:** append row to `results.tsv`, append entry to `experiment_log.md`

### results.tsv row format
```bash
echo -e "<commit>\t<val_rmse>\t<val_r2>\t<cv_rmse>\tnan\tnan\t<model>\tsingle-stage\t<seconds>\t<status>\t<expNNN description>" >> results.tsv
```

---

## Remaining Experiment Ideas (in priority order)

### Tier 1 — Likely to help
- **Taxon×ECO interactions**: tg_{taxon} × avg_FPAR, × pop_density (per-taxon ecological response)
- **ElasticNet alpha tuning**: try alpha=0.0005 or 0.003 (current 0.001 may not be optimal for 76 features)
- **SDI_park_cumean**: mean SDI_cumean across ALL taxa for this park (park-level baseline independent of taxon)

### Tier 2 — Worth trying
- **Taxon×GEO interactions**: tg_{taxon} × latitude (latitudinal diversity gradients differ by taxon)
- **SDI rolling variance**: variance of last 3–6 SDI observations (volatility signal)
- **Separate models per taxon**: 13 taxon-specific ElasticNet models (likely better fit, may overfit small taxa)

### Tier 3 — Structural changes (bigger risk)
- **Drop ECO/HUM features**: test if temporal features alone generalize better cross-park
- **Stronger alpha** (0.01–0.1): force more regularization to close the val-CV gap (0.244 vs 0.211)
- **Park target encoding**: compute mean SDI per park from training set, fill val parks with nearest-park mean

---

## Hard Rules — NEVER VIOLATE

1. **Only Section 2 of `train.py` may be edited.** Sections 0, 1, 4, 5 are frozen.
2. **Never touch `EDA.ipynb`.**
3. **Never modify `full_data_clean.csv`, `monthly_sdi.csv`, or the rarefied cache.**
4. **Never change how `val_rmse` or `val_r2` are calculated** (Section 4 of train.py).
5. **Never install new packages.** Use only what's in `vscode_env`.
6. **Never modify past rows in `results.tsv`.** Append only.
7. **Never use `python3`** — use `conda run -n vscode_env python` (python3 lacks sklearn in this env).
8. **Kill any run exceeding 10 minutes.**
9. **Do not commit `results.tsv`, `experiment_log.md`, or `run.log`.**
10. **Change ONE thing per experiment.**

---

## Progress Summary

| Phase | Experiments | Key Finding | Best val_rmse |
|-------|-------------|-------------|---------------|
| Original two-stage | 001–031 | Ridge baseline; SDI prediction feasible | 0.326 |
| Rarefied SDI baseline | 032–041 | PCA(n=10)+Ridge best; rarefaction removes effort bias | 0.326 |
| AR lag expansion | 042–059 | SDI_lag1→lag12 dominant improvement | 0.253 |
| Regularization tuning | 060–062 | ElasticNet(l1_ratio=0.2) marginal gain | 0.252 |
| Baseline features | 063–064 | SDI_cumean + SDI_month_cumean | 0.246 |
| Interaction features | 066–070 | SDI_dev, n_prior_months, taxon×season/trend | 0.243 |

**Total improvement:** 0.326 → 0.243 (−25%)  
**Remaining gap:** 0.243 − 0.200 = 0.043  
**Assessment:** Model in diminishing-returns territory; ~0.0002 improvement per experiment. Reaching 0.20 may require a structural change (mixed-effects model, richer temporal data, or taxon-specific submodels).
