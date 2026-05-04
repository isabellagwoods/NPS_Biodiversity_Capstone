# Experiment Log — NPS Biodiversity Two-Stage Model

**Goal:** Predict SDI (stage 1) and human impact residual (stage 2)
**Targets:** s1_val_rmse ≤ 0.20 AND s2_val_rmse ≤ 0.20
**Stop condition:** Both targets met, OR 50 experiments, OR 5 hours elapsed

---

## Experiment 001 — Baseline

**Date:** 2026-05-04
**Commit:** dae3c49
**Stage 1 model:** Ridge(alpha=1.0)
**Stage 2 model:** Ridge(alpha=1.0)
**s1_val_rmse:** 0.696955   **s1_val_r2:** -0.042937   **s1_cv_rmse:** 0.760973
**s2_val_rmse:** 0.831244   **s2_val_r2:** 0.329346
**total_seconds:** 0.1
**Status:** keep
**What changed:** Baseline — added SimpleImputer(median) for NaN handling; excluded lc_type_code (string) from stage1_features
**Notes:** Severe underfitting in Stage 1 (R²=-0.04, model worse than mean). Stage 2 R²=0.33 is at least positive. Both RMSE values far from 0.20 target. Next: try larger alpha for Ridge, then HistGBM which handles NaN natively.

---

## Experiment 002 — Stage 1 HistGradientBoostingRegressor

**Date:** 2026-05-04
**Commit:** 3b54bbd
**Stage 1 model:** HistGradientBoostingRegressor(max_iter=200)
**Stage 2 model:** Ridge(alpha=1.0)
**s1_val_rmse:** 0.833555   **s1_val_r2:** -0.491823   **s1_cv_rmse:** 0.808825
**s2_val_rmse:** 0.831244   **s2_val_r2:** 0.329346
**total_seconds:** 0.9
**Status:** discard
**What changed:** Stage 1 model swapped from Ridge to HistGBM(max_iter=200)
**Notes:** Stage 1 got worse (R²=-0.49). HistGBM is overfitting with only 121 training parks. Reverted. Root cause of poor Stage 1 is likely fill-value contamination: avg_GPP/max_GPP top at 32766, soil_moisture_range tops at 9999, avg_ET/max_ET top at 3276 — all MODIS fill values not caught by the <-100 filter. Next: trim features.

---

## Experiment 003 — Trim fill-value-contaminated Stage 1 features

**Date:** 2026-05-04
**Commit:** 4d5bbdd
**Stage 1 model:** Ridge(alpha=1.0) + 11 clean features
**Stage 2 model:** Ridge(alpha=1.0)
**s1_val_rmse:** 0.661187   **s1_val_r2:** 0.061363   **s1_cv_rmse:** 0.739144
**s2_val_rmse:** 0.831244   **s2_val_r2:** 0.329346
**total_seconds:** 0.1
**Status:** keep
**What changed:** Removed 6 fill-value-contaminated features from stage1_features (avg_GPP, max_GPP, avg_ET, max_ET, soil_moisture_range, avg_soil_moisture)
**Notes:** Stage 1 R² flipped positive (0.06). RMSE dropped 0.697→0.661. Root cause confirmed: fill values were corrupting features. 11 clean features remain. Still far from 0.20 — next: tune Ridge alpha.

---

## Experiment 004 — Ridge(alpha=0.1) Stage 1

**Date:** 2026-05-04  **Commit:** b9026b4
**s1_val_rmse:** 0.662007   **s1_val_r2:** 0.059033   **s1_cv_rmse:** 0.740925
**s2_val_rmse:** 0.831244   **s2_val_r2:** 0.329346   **total_seconds:** 0.1
**Status:** discard  **What changed:** Ridge alpha=1→0.1
**Notes:** Essentially identical to alpha=1. Alpha tuning gives tiny returns.

---

## Experiment 005 — Ridge(alpha=10) Stage 1

**Date:** 2026-05-04  **Commit:** 2e8fc7d
**s1_val_rmse:** 0.655062   **s1_val_r2:** 0.078673   **s1_cv_rmse:** 0.732195
**s2_val_rmse:** 0.831244   **s2_val_r2:** 0.329346   **total_seconds:** 0.1
**Status:** keep  **What changed:** Ridge alpha=1→10
**Notes:** Small gain. More regularization is better with small n.

---

## Experiment 006 — Ridge(alpha=100) Stage 1

**Date:** 2026-05-04  **Commit:** 731aea2
**s1_val_rmse:** 0.648115   **s1_val_r2:** 0.098111   **s1_cv_rmse:** 0.721184
**s2_val_rmse:** 0.831244   **s2_val_r2:** 0.329346   **total_seconds:** 0.1
**Status:** keep  **What changed:** Ridge alpha=10→100
**Notes:** Monotone improvement continues. Best Stage 1 so far.

---

## Experiment 007 — Ridge(alpha=1000) Stage 1

**Date:** 2026-05-04  **Commit:** 006204a
**s1_val_rmse:** 0.670116   **s1_val_r2:** 0.035841   **s1_cv_rmse:** 0.723252
**s2_val_rmse:** 0.831244   **s2_val_r2:** 0.329346   **total_seconds:** 0.1
**Status:** discard  **What changed:** Ridge alpha=100→1000
**Notes:** Worse — too much regularization toward mean. Optimal alpha ≈ 100.

---

## Experiment 008 — RandomForest(n=100, max_depth=5) Stage 1

**Date:** 2026-05-04  **Commit:** e90fbd6
**s1_val_rmse:** 0.718993   **s1_val_r2:** -0.109935   **s1_cv_rmse:** 0.773802
**s2_val_rmse:** 0.831244   **s2_val_r2:** 0.329346   **total_seconds:** 0.1
**Status:** discard  **What changed:** Stage 1 Ridge→RandomForest
**Notes:** Worse than Ridge. Overfitting with 121 training parks.

---

## Experiment 009 — Stage 2 HistGradientBoostingRegressor

**Date:** 2026-05-04  **Commit:** 3afcb79
**Stage 1 model:** Ridge(alpha=100)   **Stage 2 model:** HistGBM(max_iter=200)
**s1_val_rmse:** 0.648115   **s1_val_r2:** 0.098111   **s1_cv_rmse:** 0.721184
**s2_val_rmse:** 0.477557   **s2_val_r2:** 0.778644   **total_seconds:** 1.4
**Status:** keep  **What changed:** Stage 2 Ridge→HistGBM(max_iter=200)
**Notes:** Massive Stage 2 improvement: 0.831→0.477 RMSE, R² 0.33→0.78. HistGBM handles the 10k-row monthly data much better. Next: tune HistGBM, then improve Stage 1.

---

## Experiment 010 — Stage 2 HistGBM(max_iter=500)

**Date:** 2026-05-04  **Commit:** 5d6f43d
**Stage 1 model:** Ridge(alpha=100) + SelectKBest   **Stage 2 model:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.648115   **s1_val_r2:** 0.098111   **s1_cv_rmse:** 0.721184
**s2_val_rmse:** 0.475134   **s2_val_r2:** 0.780885   **total_seconds:** 2.3
**Status:** keep  **What changed:** Stage 2 max_iter 200→500
**Notes:** Tiny gain (0.477→0.475). More iterations marginally better.

---

## Experiment 011 — Stage 1 log1p target transform

**Date:** 2026-05-04  **Commit:** 26af516
**s1_val_rmse:** 0.655808   **s1_val_r2:** 0.076573   **s1_cv_rmse:** 0.722577
**s2_val_rmse:** 0.475134   **s2_val_r2:** 0.780885   **total_seconds:** 0.1
**Status:** discard  **What changed:** Stage 1 target log-transformed via custom wrapper
**Notes:** Slightly worse (0.648→0.656). SDI not skewed enough to benefit from log transform.

---

## Experiment 012 — Stage 1 SelectKBest(k=8)

**Date:** 2026-05-04  **Commit:** 985c25f
**s1_val_rmse:** 0.644141   **s1_val_r2:** 0.109138   **s1_cv_rmse:** 0.720135
**s2_val_rmse:** 0.475134   **s2_val_r2:** 0.780885   **total_seconds:** 0.1
**Status:** keep  **What changed:** Added SelectKBest(k=8) step before Ridge in Stage 1
**Notes:** Small gain (0.648→0.644). Dropping 3 weakest eco features helps marginally.

---

## Experiment 013 — Stage 1 SelectKBest(k=6)

**Date:** 2026-05-04  **Commit:** 7d48cf3
**s1_val_rmse:** 0.648750   **s1_val_r2:** 0.096344   **s1_cv_rmse:** 0.720771
**s2_val_rmse:** 0.475134   **s2_val_r2:** 0.780885   **total_seconds:** 0.1
**Status:** discard  **What changed:** k=8→k=6
**Notes:** Worse than k=8. Sweet spot is k=8.

---

## Experiment 014 — Stage 2 trim to top-8 human features

**Date:** 2026-05-04  **Commit:** c4a542d
**Stage 1 model:** Ridge+SelectKBest(k=8)   **Stage 2 model:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.644141   **s1_val_r2:** 0.109138   **s1_cv_rmse:** 0.720135
**s2_val_rmse:** 0.473580   **s2_val_r2:** 0.782315   **total_seconds:** 2.6
**Status:** keep  **What changed:** Stage 2 human features trimmed to top 8 by |correlation|
**Notes:** Best Stage 2 result (0.474). Dropping NaN covid cols and weak pollution features marginal improvement. **Final best configuration.**

---

## Experiment 015 — Stage 2 HistGBM learning_rate=0.05

**Date:** 2026-05-04  **Commit:** c808cdf
**s1_val_rmse:** 0.644141   **s2_val_rmse:** 0.477581   **total_seconds:** 2.8
**Status:** discard  **What changed:** HistGBM learning_rate 0.1→0.05
**Notes:** Slightly worse. Default lr=0.1 optimal for this dataset size.

---

<!-- Copy the block above for each new experiment -->
