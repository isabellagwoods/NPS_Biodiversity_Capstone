# Experiment Log — NPS Biodiversity SDI Prediction

**Goal:** Predict Shannon Diversity Index (SDI) from ecological, climate, geographical, and visitation data  
**Success:** val_rmse ≤ 0.10 AND val_r2 ≥ 0.85  
**Metric:** val_rmse (lower = better). DO NOT change.

---

## Experiment 001 — Baseline (LinearRegression + median imputation)

**Date:** 2026-04-27  
**Commit:** 0893195  
**Model:** LinearRegression  
**Params:** fit_intercept=True, positive=False  
**val_rmse:** 1170.517705  
**val_r2:** -2941740.956822  
**fit_seconds:** 2.1  
**Status:** keep  
**Notes:** Baseline. Original train.py crashed immediately — data has NaN values in many columns
(notably `visit_covid_impact` and `backcountry_covid_impact` are all-NaN). Fixed by adding
`SimpleImputer(strategy='median')` to the pipeline. Metrics are catastrophically bad because
LinearRegression with 92 features and ~120 training samples massively overfits. This sets the
floor; Ridge regularization should improve dramatically.

---

## Experiment 002 — Ridge Regression

**Date:** 2026-04-27  
**Commit:** c55f6b7  
**Model:** Ridge  
**Params:** alpha=100 (best from grid [0.01, 0.1, 1, 10, 100])  
**val_rmse:** 0.576944  
**val_r2:** 0.285312  
**fit_seconds:** 2.1  
**Status:** discard  
**Notes:** Massive improvement in val_rmse vs baseline (0.577 vs 1170) — L2 regularization tames
the 92-feature overfit. However val_r2 = 0.285 < 0.85 threshold → discarded per rules.
Best alpha was 100, suggesting we need much stronger regularization or a better model family.
Next: try HistGradientBoostingRegressor (handles NaN natively, ensemble method).  

---

## Experiment 003 — HistGradientBoostingRegressor

**Date:** 2026-04-27  
**Commit:** afaed1c  
**Model:** HistGradientBoostingRegressor  
**Params:** learning_rate=0.05, max_depth=3, max_iter=100 (best from grid)  
**val_rmse:** 0.566529  
**val_r2:** 0.310884  
**fit_seconds:** 4.1  
**Status:** discard  
**Notes:** Handles NaN natively; removed imputer from pipeline. Marginally better than Ridge
(val_rmse 0.567 vs 0.577). Both models show similar r2 (~0.30), suggesting the 92 features may
contain a lot of noise relative to signal with only ~120 training samples.
val_r2 = 0.311 < 0.85 threshold → discarded per rules. Next steps: aggressive feature selection
(drop near-zero importance columns), or log-transform of SDI target, or ensemble stacking.  

---

<!-- copy the block above for each new experiment -->
