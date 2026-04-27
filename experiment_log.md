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

## Experiment 004 — Lasso

**Date:** 2026-04-27  
**Commit:** 9e9e6d9  
**Model:** Lasso  
**Params:** alpha=0.1 (best from grid [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500]), max_iter=10000  
**val_rmse:** 0.654973  
**val_r2:** 0.078923  
**fit_seconds:** 2.1  
**Status:** discard  
**Notes:** L1 penalty performs worse than Ridge (rmse 0.655 vs 0.577). Best alpha=0.1 is relatively
small, meaning it isn't zeroing out many features. Low val_r2=0.079 suggests the sparse solution
isn't capturing signal well. val_r2 < 0.85 → discarded. Next: ElasticNet to blend L1+L2.

---

## Experiment 005 — ElasticNet

**Date:** 2026-04-27  
**Commit:** 1138160  
**Model:** ElasticNet  
**Params:** alpha=1, l1_ratio=0.1 (best from 8×5=40 grid combos), max_iter=10000  
**val_rmse:** 0.640322  
**val_r2:** 0.119669  
**fit_seconds:** 2.5  
**Status:** discard  
**Notes:** Between Lasso (0.655) and Ridge (0.577) in rmse. Best l1_ratio=0.1 means the model
gravitates toward pure Ridge — confirms L2 is more useful than L1 for this dataset. The wide
alpha grid confirms alpha=1 is a reasonable sweet spot for the L1+L2 blend. val_r2=0.120 < 0.85
→ discarded. Linear models are hitting a ceiling around val_r2~0.3 with these 92 features.
Next: ensemble/tree methods or aggressive feature selection needed to break through.  

---

## Experiment 006 — PolynomialFeatures(degree=2) + Ridge

**Date:** 2026-04-27  
**Commit:** 939754d  
**Model:** PolynomialFeatures(degree=2, interaction_only=True) + Ridge  
**Params:** alpha=500 (best from [0.1, 1, 10, 100, 500])  
**val_rmse:** 0.719819  
**val_r2:** -0.112489  
**fit_seconds:** 2.2  
**Status:** discard  
**Notes:** Interaction terms hurt rather than help. With 92 features, degree-2 interactions create
~4000+ pairwise columns, most of which are noise. Even with high alpha=500 Ridge can't regularize
them enough given ~120 training samples. Negative val_r2 means worse than predicting the mean.
val_r2 < 0.85 → discarded.

---

## Experiment 007 — RandomForestRegressor

**Date:** 2026-04-27  
**Commit:** 4da78d9  
**Model:** RandomForestRegressor  
**Params:** n_estimators=200, max_depth=None, max_features=0.5 (best from grid)  
**val_rmse:** 0.633819  
**val_r2:** 0.137461  
**fit_seconds:** 4.1  
**Status:** discard  
**Notes:** Best val_r2 among tree models tried so far (0.137). Unlimited depth (None) chosen,
suggesting the forest needs to grow deep to capture signal. max_features=0.5 (50% of features
per split) beats sqrt, indicating broader feature coverage per tree helps. Still far from 0.85
target. val_r2 < 0.85 → discarded.

---

## Experiment 008 — TransformedTargetRegressor (log1p) + Ridge

**Date:** 2026-04-27  
**Commit:** f130887  
**Model:** TransformedTargetRegressor(func=log1p, inverse=expm1) wrapping Ridge  
**Params:** regressor__alpha=100 (best from [0.01, 0.1, 1, 10, 100])  
**val_rmse:** 0.583874  
**val_r2:** 0.268041  
**fit_seconds:** 1.9  
**Status:** discard  
**Notes:** Log-transforming SDI provides no meaningful benefit over plain Ridge (0.584 vs 0.577
rmse). SDI distribution is likely already near-normal, so log transform doesn't reduce skew.
Same best alpha=100 as plain Ridge. val_r2 = 0.268 < 0.85 → discarded.

---

## Experiment 009 — SelectKBest(f_regression, k=30) + Ridge

**Date:** 2026-04-27  
**Commit:** 92e1827  
**Model:** SelectKBest(f_regression) + Ridge  
**Params:** select__k=30, model__alpha=100 (best from 4×4=16 grid combos)  
**val_rmse:** 0.569386  
**val_r2:** 0.303914  
**fit_seconds:** 2.1  
**Status:** discard  
**Notes:** Narrowly ties HGB (0.567) as best val_rmse overall. Cutting 92 → 30 features via
univariate F-test improves on plain Ridge (0.577), confirming ~62 features add mostly noise.
k=30 is the sweet spot — k=50 likely reintroduces noise, k=10/20 underfit. alpha=100 consistent
with all previous Ridge-based runs. val_r2 = 0.304 < 0.85 → discarded. Combining SelectKBest
with an ensemble (HGB/RF) may be a productive next direction.

---

<!-- copy the block above for each new experiment -->
