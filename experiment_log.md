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

---

## Experiment 016 — 2021+ Filter + Park-Based Split Baseline

**Date:** 2026-05-11
**Commit:** b54de58
**Stage 1 model:** Ridge(alpha=100) + SelectKBest(k=8)
**Stage 2 model:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.653716   **s1_val_r2:** 0.107614   **s1_cv_rmse:** 0.720135
**s2_val_rmse:** 0.936023   **s2_val_r2:** -0.033197
**total_seconds:** 3.3
**Status:** keep
**What changed:** (1) Monthly data filtered to year ≥ 2021 (~5552 rows vs 10393). (2) Park-based train/test split: all months from a park go entirely to train or test. (3) Added PCA/KernelPCA imports for dim-reduction study.
**Notes:** Stage 1 RMSE unchanged (0.6537 vs 0.6441 prior best — slight variation from new split randomness). Stage 2 jumps 0.474→0.936: the park-based split forces true park-level generalization instead of month-level, which is a harder but more honest evaluation. s2_r2 is negative — model predicts worse than mean for unseen parks.

---

## Experiment 017 — PCA(n=5)+Ridge Stage 1

**Date:** 2026-05-11
**Commit:** cd170a2
**Stage 1 model:** PCA(n=5)+Ridge(alpha=100)
**Stage 2 model:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.654275   **s1_val_r2:** 0.106087   **s1_cv_rmse:** 0.720247
**s2_val_rmse:** 0.936023   **s2_val_r2:** -0.033197
**total_seconds:** 2.6
**Status:** discard
**What changed:** Stage 1 — replaced SelectKBest(k=8) with PCA(n_components=5)
**Notes:** Essentially identical to baseline (0.6543 vs 0.6537). PCA retains variance but loses discriminative feature alignment that SelectKBest preserves. Not an improvement.

---

## Experiment 018 — PCA(n=8)+Ridge Stage 1

**Date:** 2026-05-11
**Commit:** 2ce6948
**Stage 1 model:** PCA(n=8)+Ridge(alpha=100)
**s1_val_rmse:** 0.658365   **s1_val_r2:** 0.094876   **s1_cv_rmse:** 0.721328
**s2_val_rmse:** 0.936023   **s2_val_r2:** -0.033197
**total_seconds:** 2.5
**Status:** discard
**What changed:** PCA n_components 5→8
**Notes:** Worse than n=5. More PCA components re-introduce noise that SelectKBest would have filtered. PCA(n=8) approaches the full 11-feature space with correlated axes.

---

## Experiment 019 — PCA(n=3)+Ridge Stage 1

**Date:** 2026-05-11
**Commit:** f861282
**Stage 1 model:** PCA(n=3)+Ridge(alpha=100)
**s1_val_rmse:** 0.654761   **s1_val_r2:** 0.104758   **s1_cv_rmse:** 0.718283
**s2_val_rmse:** 0.936023   **s2_val_r2:** -0.033197
**total_seconds:** 2.6
**Status:** discard
**What changed:** PCA n_components 5→3
**Notes:** Val RMSE slightly worse than baseline but best CV RMSE of all PCA variants (0.7183 < 0.7201). Narrower PCA generalizes slightly better in CV but not in val split. Overall: no PCA variant beats SelectKBest.

---

## Experiment 020 — PCA(n=5) Both Stage 1 and Stage 2

**Date:** 2026-05-11
**Commit:** 0996b94
**Stage 1 model:** PCA(n=5)+Ridge(alpha=100)
**Stage 2 model:** Imputer+Scaler+PCA(n=5)+HistGBM(max_iter=500)
**s1_val_rmse:** 0.654275   **s1_val_r2:** 0.106087   **s1_cv_rmse:** 0.720247
**s2_val_rmse:** 1.029077   **s2_val_r2:** -0.248836
**total_seconds:** 2.9
**Status:** discard
**What changed:** Added PCA(n=5) to Stage 2 pipeline (with imputer+scaler)
**Notes:** Stage 2 gets significantly worse (0.936→1.029). PCA on the human/traffic/time feature space destroys the critical dimensions that let HistGBM generalize across parks. The original feature space carries semantically meaningful axes (traffic, pollution, visitors) that PCA rotates away.

---

## Experiment 021 — KernelPCA(n=5, rbf)+Ridge Stage 1

**Date:** 2026-05-11
**Commit:** d65b62e
**Stage 1 model:** KernelPCA(n=5, kernel=rbf)+Ridge(alpha=100)
**Stage 2 model:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.692817   **s1_val_r2:** -0.002333   **s1_cv_rmse:** 0.728246
**s2_val_rmse:** 0.936023   **s2_val_r2:** -0.033197
**total_seconds:** 2.3
**Status:** discard
**What changed:** Stage 1 — replaced PCA with KernelPCA(n_components=5, kernel='rbf')
**Notes:** Much worse (0.693 vs 0.654). Nonlinear kernel PCA needs many more samples to find stable nonlinear manifold structure. With only ~122 training parks, kernel trick finds noise. R² near zero — model barely better than mean.

---

## Experiment 022 — SelectKBest(k=10)+PCA(n=5)+Ridge Stage 1

**Date:** 2026-05-11
**Commit:** 236e82d
**Stage 1 model:** SelectKBest(k=10)+PCA(n=5)+Ridge(alpha=100)
**Stage 2 model:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.659575   **s1_val_r2:** 0.091545   **s1_cv_rmse:** 0.721757
**s2_val_rmse:** 0.936023   **s2_val_r2:** -0.033197
**total_seconds:** 2.5
**Status:** discard
**What changed:** Stage 1 — chain SelectKBest(k=10) then PCA(n=5) (filter then reduce)
**Notes:** Worse than SelectKBest alone (0.660 vs 0.654). The PCA step after filtering discards discriminative variance that Ridge would otherwise exploit. The two-step pipeline creates too much compression.

---

## Experiment 023 — PCA(0.95 variance explained)+Ridge Stage 1

**Date:** 2026-05-11
**Commit:** cf4cca2
**Stage 1 model:** PCA(n_components=0.95, svd_solver=full)+Ridge(alpha=100)
**Stage 2 model:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.659225   **s1_val_r2:** 0.092508   **s1_cv_rmse:** 0.721549
**s2_val_rmse:** 0.936023   **s2_val_r2:** -0.033197
**total_seconds:** 2.9
**Status:** discard
**What changed:** Stage 1 — PCA auto-selects n_components to explain 95% variance
**Notes:** Worse than baseline (0.659 vs 0.654). Auto-threshold doesn't help — selecting 95% variance from 11 features keeps most components, reintroducing correlated noise. Feature selection by target correlation (SelectKBest) outperforms variance-based selection (PCA) for this dataset.

---

## Experiment 024 — Best Config Confirmed (Dim-Reduction Study Complete)

**Date:** 2026-05-11
**Commit:** b70e695
**Stage 1 model:** SelectKBest(k=8)+Ridge(alpha=100)
**Stage 2 model:** HistGBM(max_iter=500)
**s1_val_rmse:** 0.653716   **s1_val_r2:** 0.107614   **s1_cv_rmse:** 0.720135
**s2_val_rmse:** 0.936023   **s2_val_r2:** -0.033197
**total_seconds:** 2.7
**Status:** keep
**What changed:** Reverted to best Stage 1 config — end of dimensionality reduction study
**Notes:** Confirms SelectKBest(k=8)+Ridge(alpha=100) is the best Stage 1 configuration. Dimensionality reduction study conclusion: for n=152 parks with 11 clean ecological features, target-aligned feature selection (SelectKBest) consistently outperforms variance-based reduction (PCA/KernelPCA). Stage 2 cross-park generalization remains the harder problem (RMSE=0.936 with park-based split).

---

## Dimension-Reduction Study Summary (Exp 016–024)

**Controlled variable:** Dimensionality reduction method in Stage 1 (one change per experiment)
**Fixed:** Stage 2 = HistGBM(max_iter=500), data = 2021+ monthly, park-based split

| Exp | Method | s1_val_rmse | s1_cv_rmse | vs baseline |
|-----|--------|-------------|------------|-------------|
| 016 | SelectKBest(k=8) — baseline | 0.6537 | 0.7201 | — |
| 017 | PCA(n=5) | 0.6543 | 0.7202 | +0.0006 |
| 018 | PCA(n=8) | 0.6584 | 0.7213 | +0.0047 |
| 019 | PCA(n=3) | 0.6548 | **0.7183** | +0.0011 |
| 021 | KernelPCA(rbf,n=5) | 0.6928 | 0.7282 | +0.0391 |
| 022 | SelectKBest(k=10)+PCA(n=5) | 0.6596 | 0.7218 | +0.0059 |
| 023 | PCA(0.95 var) | 0.6592 | 0.7215 | +0.0055 |

**Finding:** SelectKBest beats all PCA variants for Stage 1. Feature selection by correlation with target is more effective than variance-based reduction when n_samples (~122) is comparable to n_features (11). The SDI signal is carried in specific ecological features (FPAR, SNOW, burn), not in variance directions.

<!-- Copy the block above for each new experiment -->

---

## ═══ NEW MODEL: Single-Stage Rarefied Taxon SDI ═══

**Architecture change (2026-05-11):** Replaced two-stage park-level model with a single-stage
model predicting rarefied SDI per (park × month × taxon_group). Rarefaction (N=50 subsample per cell)
removes the iNaturalist observer-effort confound (r: 0.88 → 0.35). Data: 8,824 rows from 149 parks,
2021+. Park-based split maintained. See `train_datastructures.py` Structure G for prior validation.

---

## Experiment 032 — Baseline: Single-Stage Ridge(alpha=100) on Rarefied Taxon SDI

**Date:** 2026-05-11
**Commit:** 04434f5
**Model:** Ridge(alpha=100)
**Target:** SDI_rarefied (N=50 rarefaction per park × month × taxon_group cell)
**Data:** 8,824 rows, 149 parks, 2021+
**Features:** ALL_NUM (26 numeric: ECO+HUM+GEO+TIME+TRAF+log_n_obs) + taxon_group (one-hot)
**s1_val_rmse:** 0.334240   **s1_val_r2:** 0.611188   **s1_cv_rmse:** 0.304210
**total_seconds:** 0.1
**Status:** keep
**What changed:** New train.py — single-stage rarefied taxon model replacing two-stage architecture
**Notes:** Matches Structure G result from train_datastructures.py (val_rmse=0.334, R²=0.611).
  Taxon group one-hot is the most powerful feature set; ecological features (FPAR, SNOW) modulate
  group-level baselines. CV RMSE (0.304) better than val RMSE (0.334) — val parks are slightly
  harder than average (park-based split variance). Goal: improve val_r2 above 0.70.

---

## Experiment 033 — Round 1a: Ridge(alpha=10)

**Date:** 2026-05-11
**Commit:** 163b115
**Model:** Ridge(alpha=10)
**s1_val_rmse:** 0.334572   **s1_val_r2:** 0.610417   **s1_cv_rmse:** 0.288099
**total_seconds:** 0.2
**Status:** discard
**What changed:** alpha 100 → 10 (less regularization)
**Notes:** Marginally worse than alpha=100 (val_rmse 0.3346 vs 0.3342). Less regularization
  does not help — the correlated feature groups (multiple FPAR columns, multiple visitor
  columns) benefit from the shrinkage that alpha=100 provides. CV RMSE improved slightly
  (0.288 vs 0.304), indicating less bias, but val generalization suffers.

---

## Experiment 034 — Round 1b: Ridge(alpha=500)

**Date:** 2026-05-11
**Commit:** 51babb3
**Model:** Ridge(alpha=500)
**s1_val_rmse:** 0.375614   **s1_val_r2:** 0.508974   **s1_cv_rmse:** 0.363112
**total_seconds:** 0.1
**Status:** discard
**What changed:** alpha 100 → 500 (more regularization)
**Notes:** Substantially worse (RMSE 0.334→0.376, R² 0.611→0.509). Over-regularization
  shrinks all coefficients toward zero — the taxon_group one-hot coefficients (which are
  large and carry the dominant signal) get over-shrunk. Monotone degradation as alpha increases.

---

## Experiment 035 — Round 1c: Ridge(alpha=1000)

**Date:** 2026-05-11
**Commit:** 4015334
**Model:** Ridge(alpha=1000)
**s1_val_rmse:** 0.405512   **s1_val_r2:** 0.427693   **s1_cv_rmse:** 0.398100
**total_seconds:** 0.1
**Status:** discard
**What changed:** alpha 100 → 1000 (further regularization)
**Notes:** Worse still (0.406, R²=0.428). Confirms monotone degradation above alpha=100.
  Alpha=100 is the optimal regularization strength for this dataset.

---

## Round 1 Summary — Alpha Search

| Exp | alpha | val_rmse | val_r2 | cv_rmse | Status |
|-----|-------|----------|--------|---------|--------|
| 032 | 100 | **0.334** | **0.611** | 0.304 | baseline |
| 033 | 10  | 0.335 | 0.610 | 0.288 | discard |
| 034 | 500 | 0.376 | 0.509 | 0.363 | discard |
| 035 | 1000 | 0.406 | 0.428 | 0.398 | discard |

**Finding:** alpha=100 is optimal. Both lower (α=10) and higher (α=500+) regularization degrade
val performance. The model is well-calibrated at α=100. Round 2 uses alpha=100.

---

## Experiment 036 — Round 2a: PCA(n=15)+Ridge(alpha=100)

**Date:** 2026-05-11
**Commit:** 5f21ce3
**Model:** PCA(n_components=15) inside numeric sub-pipeline + Ridge(alpha=100)
**s1_val_rmse:** 0.334289   **s1_val_r2:** 0.611075   **s1_cv_rmse:** 0.305437
**total_seconds:** 0.1
**Status:** discard
**What changed:** Added PCA(n=15) step after StandardScaler in numeric sub-pipeline.
  Categorical (taxon_group) branch untouched.
**Notes:** Effectively identical to baseline (0.334289 vs 0.334240). Retaining 15/26
  principal components keeps nearly all variance — PCA at this level is redundant.

---

## Experiment 037 — Round 2b: PCA(n=10)+Ridge(alpha=100)

**Date:** 2026-05-11
**Commit:** ac27209
**Model:** PCA(n_components=10) inside numeric sub-pipeline + Ridge(alpha=100)
**s1_val_rmse:** 0.326281   **s1_val_r2:** 0.629484   **s1_cv_rmse:** 0.308746
**total_seconds:** 0.1
**Status:** keep  ← NEW BEST
**What changed:** PCA reduced to n=10 components (from n=15 in Exp 036)
**Notes:** New best result. val_rmse 0.334→0.326 (−2.4%), R² 0.611→0.629. Reducing to 10
  components filters out low-signal variance directions among the 26 correlated numeric features
  (MODIS eco-features are highly inter-correlated; HUM features overlap). PCA regularizes by
  discarding noise components. The taxon_group one-hot is unaffected — its signal is preserved.
  Why 10 works: the 26 features span ~10 meaningful ecological/human dimensions; the remaining
  16 components capture noise + multicollinearity. Key finding: PCA provides implicit regularization
  for correlated eco-feature groups.

---

## Experiment 038 — Round 2c: PCA(n=20)+Ridge(alpha=100)

**Date:** 2026-05-11
**Commit:** f4df94c
**Model:** PCA(n_components=20) inside numeric sub-pipeline + Ridge(alpha=100)
**s1_val_rmse:** 0.334101   **s1_val_r2:** 0.611513   **s1_cv_rmse:** 0.305197
**total_seconds:** 0.1
**Status:** discard
**What changed:** PCA at n=20 (between 15 and 26 — keeping most variance)
**Notes:** Essentially identical to baseline and n=15 result. The benefit of PCA only
  appears when enough components are discarded to remove multicollinearity (n=10).
  20 components retain too much correlated structure.

---

## Round 2 Summary — PCA n_components Search

| Exp | n_components | val_rmse | val_r2 | cv_rmse | Status |
|-----|--------------|----------|--------|---------|--------|
| 032 | none (baseline) | 0.334 | 0.611 | 0.304 | — |
| 036 | 15 | 0.334 | 0.611 | 0.305 | discard |
| **037** | **10** | **0.326** | **0.629** | 0.309 | **keep** |
| 038 | 20 | 0.334 | 0.612 | 0.305 | discard |

**Finding:** PCA(n=10) is the sweet spot. The 26 numeric features collapse into ~10 meaningful
dimensions (ecological satellite bands and human-impact axes are heavily inter-correlated).
Discarding 16 low-signal components regularizes the linear model without losing signal.

---

## Experiment 039 — Round 3a: ElasticNet(alpha=0.01, l1_ratio=0.5)

**Date:** 2026-05-11
**Commit:** 0ad72cf
**Model:** ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
**s1_val_rmse:** 0.337580   **s1_val_r2:** 0.603378   **s1_cv_rmse:** 0.308464
**total_seconds:** 0.2
**Status:** discard
**What changed:** Replaced Ridge with ElasticNet (equal L1+L2 mix, weak regularization)
**Notes:** Marginally worse than Ridge(α=100). L1 penalty zeros out a few features but the
  resulting sparse solution hurts generalization. Ridge's L2 uniform shrinkage fits this
  problem better — all features carry some signal (taxon, eco, human); none should be zeroed.

---

## Experiment 040 — Round 3b: Lasso(alpha=0.001)

**Date:** 2026-05-11
**Commit:** 65be0a2
**Model:** Lasso(alpha=0.001, max_iter=10000)
**s1_val_rmse:** 0.330861   **s1_val_r2:** 0.619011   **s1_cv_rmse:** 0.289594
**total_seconds:** 0.4
**Status:** keep
**What changed:** Replaced Ridge with Lasso (pure L1, alpha=0.001)
**Notes:** Better than Ridge baseline (0.331 vs 0.334). Lasso at very low alpha (soft L1)
  selects a sparse set of the most predictive features; the one-hot taxon coefficients dominate
  and are kept, while some redundant eco-feature coefficients are zeroed. CV RMSE (0.290) is
  notably better than baseline (0.304). However, PCA(n=10)+Ridge (Exp 037) still outperforms
  on val_rmse (0.326 < 0.331). Both are improvements over the baseline.

---

## Experiment 041 — Round 3c: PolynomialFeatures(degree=2, interaction_only)+Ridge

**Date:** 2026-05-11
**Commit:** bbe02cb
**Model:** PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) after
  ColumnTransformer, then Ridge(alpha=100)
**s1_val_rmse:** 0.388727   **s1_val_r2:** 0.474089   **s1_cv_rmse:** 0.262424
**total_seconds:** 0.5
**Status:** discard
**What changed:** Added PolynomialFeatures step after ColumnTransformer to test taxon×numeric
  interactions (e.g., traffic × Insecta, FPAR × Plantae). Creates ~630 interaction features.
**Notes:** Strong train/val divergence — CV RMSE excellent (0.262, best of all experiments)
  but val RMSE 0.389 is much worse than baseline. Textbook overfitting pattern: the 630
  interaction features learn park-specific interaction patterns that don't transfer to the 27
  held-out val parks. Ridge(α=100) cannot shrink 630 features adequately with n=6,979 training
  points when generalization target is 27 unseen parks (the bottleneck is park diversity, not row N).
  Key finding: taxon×feature interactions are not globally generalizable across unseen parks.
  Within-park dynamics dominate interaction structure.

---

## Round 3 Summary — Model Type Search

| Exp | Model | val_rmse | val_r2 | cv_rmse | Status |
|-----|-------|----------|--------|---------|--------|
| 032 | Ridge(α=100) — baseline | 0.334 | 0.611 | 0.304 | — |
| 039 | ElasticNet(α=0.01, l1=0.5) | 0.338 | 0.603 | 0.308 | discard |
| **040** | **Lasso(α=0.001)** | **0.331** | **0.619** | **0.290** | **keep** |
| 041 | PolynomialFeatures+Ridge | 0.389 | 0.474 | 0.262 | discard |

**Finding:** Ridge and Lasso are comparable; Lasso's soft sparsity improves over baseline marginally.
Interaction features overfit badly — the park-level generalization bottleneck (27 val parks) limits
the number of parameters that can be estimated without overfitting.

---

## Three-Round Controlled Experiment Summary (Exp 032–041)

**Study:** Single-stage rarefied taxon SDI model — optimizing the linear predictor
**Controlled structure:** Rarefied taxon×month×park rows (8,824), park-based split, 2021+

| Exp | Config | val_rmse | val_r2 | cv_rmse | Status |
|-----|--------|----------|--------|---------|--------|
| 032 | Ridge(α=100) — baseline | 0.334 | 0.611 | 0.304 | keep |
| 033 | Ridge(α=10) | 0.335 | 0.610 | 0.288 | discard |
| 034 | Ridge(α=500) | 0.376 | 0.509 | 0.363 | discard |
| 035 | Ridge(α=1000) | 0.406 | 0.428 | 0.398 | discard |
| 036 | PCA(n=15)+Ridge | 0.334 | 0.611 | 0.305 | discard |
| **037** | **PCA(n=10)+Ridge** | **0.326** | **0.629** | 0.309 | **BEST** |
| 038 | PCA(n=20)+Ridge | 0.334 | 0.612 | 0.305 | discard |
| 039 | ElasticNet(α=0.01) | 0.338 | 0.603 | 0.308 | discard |
| 040 | Lasso(α=0.001) | 0.331 | 0.619 | 0.290 | keep |
| 041 | PolyFeatures+Ridge | 0.389 | 0.474 | 0.262 | discard |

**Best model:** PCA(n=10)+Ridge(alpha=100) → val_rmse=0.326, val_r2=0.629
**Canonical `train.py`** updated to this configuration (commit 7104b47).

### Key Findings

1. **alpha=100 is optimal** for Ridge on this dataset. Monotone degradation above α=100.
2. **PCA(n=10) provides meaningful regularization** for the 26 correlated numeric features.
   Eco-satellite bands and human-impact features collapse to ~10 orthogonal signal dimensions.
3. **Lasso is a viable alternative** (0.331, R²=0.619) but PCA+Ridge is still best on val.
4. **Interaction features overfit** — park generalization (27 val parks) is the bottleneck,
   not row count. 630 interaction features cannot generalize across park ecology.
5. **Current best R²=0.629** — approaching but not yet at the 0.70 target. Next step:
   feature engineering (e.g., SDI_lag1 from Structure E) or richer data sources.

<!-- New experiments appended below -->

## Experiment 042 — SDI_lag1 Temporal Lag Feature

**Date:** 2026-05-17
**Commit:** bfd87cf
**Model:** Ridge(alpha=100) + SDI_lag1
**Structure:** Added SDI_lag1 (prior month's rarefied SDI for same park×taxon_group) as numeric feature.
  Lag computed via sort-by-(park,taxon,year,month) then groupby shift(1). Rows with NaN lag dropped (~1 row per park-group).
**s1_val_rmse:** 0.291723   **s1_val_r2:** 0.693769   **s1_cv_rmse:** 0.269599
**total_seconds:** 4.3
**Status:** keep
**What changed:** Added SDI_lag1 temporal autocorrelation feature; PCA(n=10) still in pipeline
**Notes:** Biggest single improvement of the session. R² jumped from 0.629→0.694. Temporal autocorrelation
  dominates — prior month's biodiversity is the strongest predictor of current month's. PCA(n=10) may be
  diluting the lag signal by mixing it with 27 other features. Next: try removing PCA.

---

## Experiment 043 — HistGBM + Lag (No PCA)

**Date:** 2026-05-17
**Commit:** 060a5d1 (reverted)
**Model:** HistGradientBoostingRegressor(max_iter=500) + SDI_lag1, no PCA
**s1_val_rmse:** 0.303075   **s1_val_r2:** 0.669472   **s1_cv_rmse:** 0.235978
**total_seconds:** 19.7
**Status:** discard
**What changed:** Swapped Ridge for HistGBM; removed PCA; added lag
**Notes:** CV=0.236 is excellent (best so far) but val=0.303 is worse than exp042 (0.292).
  Persistent val-CV gap indicates park-level overfitting. HistGBM fits within-park patterns that don't
  generalize to unseen parks. Ridge with lag is more conservative. Reverted.

---

## Experiment 044 — Drop PCA, Ridge Direct on All Features + Lag

**Date:** 2026-05-17
**Commit:** bf02766
**Model:** Ridge(alpha=100) + SDI_lag1, no PCA
**s1_val_rmse:** 0.281012   **s1_val_r2:** 0.715844   **s1_cv_rmse:** 0.254197
**total_seconds:** 3.8
**Status:** keep
**What changed:** Removed PCA from numeric pipeline; Ridge gets direct coefficient for each feature
**Notes:** Confirmed hypothesis — PCA was diluting the SDI_lag1 signal. Removing PCA lets Ridge assign
  a direct coefficient to lag. 0.292→0.281. R² 0.694→0.716. Next: try Lasso for sparse selection.

---

## Experiment 045 — Lasso(alpha=0.001) + Lag, No PCA

**Date:** 2026-05-17
**Commit:** 44a0c4a
**Model:** Lasso(alpha=0.001, max_iter=10000) + SDI_lag1, no PCA
**s1_val_rmse:** 0.278130   **s1_val_r2:** 0.721642   **s1_cv_rmse:** 0.250152
**total_seconds:** 0.3
**Status:** keep
**What changed:** Swapped Ridge(alpha=100) for Lasso(alpha=0.001) — sparse linear with L1 penalty
**Notes:** Marginal gain over Ridge. 0.281→0.278. Lasso's sparse selection handles collinear eco/human
  features slightly better. CV also improves (0.254→0.250). New canonical base: Lasso+lag, no PCA.

---

## Experiment 046 — HistGBM Conservative (lr=0.05, min_samples_leaf=50) + Lag

**Date:** 2026-05-17
**Commit:** ea27373 (reverted)
**Model:** HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, min_samples_leaf=50) + SDI_lag1
**s1_val_rmse:** 0.297982   **s1_val_r2:** 0.680488   **s1_cv_rmse:** 0.233969
**total_seconds:** 42.6
**Status:** discard
**What changed:** HistGBM with conservative hyperparameters to reduce park overfitting; no PCA
**Notes:** CV=0.234 (best CV of all experiments) but val=0.298 is still worse than Lasso (0.278).
  Conservative settings reduce val-CV gap somewhat (exp043 val=0.303 → 0.298) but can't close it.
  The fundamental issue is that tree models fit park-specific interaction patterns that don't transfer.
  Linear models generalize better here. Reverted.

---

## Experiment 047 — Nearest-Park Residual Correction (ECO+HUM+GEO, K=5)

**Date:** 2026-05-18
**Commit:** ff3de74
**Model:** Lasso(alpha=0.001) + SDI_lag1 + post-hoc nearest-park correction
**Correction:** Per-park mean training residual, averaged across K=5 nearest neighbors by
  ecological+human+geographic distance (StandardScaled, Euclidean). Uniform weighting.
**s1_val_rmse:** 0.275341   **s1_val_r2:** 0.727197   **s1_cv_rmse:** 0.250152
**total_seconds:** 0.4
**Status:** keep
**What changed:** Added post-hoc residual correction using nearest training parks
**Notes:** Grid search over K∈{3,5,10,15} and feature sets {ECO, ECO+HUM, ECO+HUM+GEO}.
  Best: K=5, ECO+HUM+GEO. 0.278→0.275. Marginal gain; residuals are weakly ecology-correlated.
  Next: try inverse-distance weighting (closer parks contribute more).

---

## Experiment 048 — IDW Correction (ECO+HUM+GEO, K=5)

**Date:** 2026-05-18
**Commit:** d5554ec  ← HEAD (canonical)
**Model:** Lasso(alpha=0.001) + SDI_lag1 + IDW nearest-park correction
**Correction:** Σ(resid_i / (dist_i + ε)) / Σ(1 / (dist_i + ε)), K=5, ECO+HUM+GEO features.
**s1_val_rmse:** 0.275242   **s1_val_r2:** 0.727393   **s1_cv_rmse:** 0.250152
**total_seconds:** 0.4
**Status:** keep (current best)
**What changed:** Replaced uniform neighbor weighting with inverse-distance weighting
**Notes:** Negligible improvement over uniform (0.27534→0.27524). Residuals are not strongly
  distance-ordered — nearby parks don't systematically have more similar residuals than distant ones.
  The cross-park generalization bottleneck is structural, not a weighting issue.
  **Current best overall:** val_rmse=0.2752, R²=0.727.

---

## Experiment 049 — K-Means Park Cluster Models (K=2..6)

**Date:** 2026-05-18
**Commit:** 5fd3ca6 (reverted)
**Model:** KMeans(K=2..6) on ECO+HUM+GEO, separate Lasso+lag per cluster
**s1_val_rmse:** 0.295751   **s1_val_r2:** 0.685254   **s1_cv_rmse:** 0.250152
**total_seconds:** 3.1
**Status:** discard
**What changed:** Replaced global model with per-cluster models; grid search K=2..6
**Notes:** K=2 gave val=0.296 (worse than global 0.278). K≥3 produced empty val-park clusters
  (27 val parks couldn't be distributed to 3+ clusters without some being empty). With only 27
  val parks and 108 train parks, clustering fragments the data too much for separate models.
  Park-count bottleneck makes clustering infeasible. Reverted.

---

## Experiment 050 — HistGBM + IDW Correction

**Date:** 2026-05-18
**Commit:** 7f19fce (reverted)
**Model:** HistGradientBoostingRegressor(max_iter=500) + SDI_lag1 + IDW nearest-park correction
**s1_val_rmse:** 0.297982   **s1_val_r2:** 0.680488   **s1_cv_rmse:** 0.233969
**total_seconds:** 15.6
**Status:** discard
**What changed:** Applied IDW correction to HistGBM (to test whether IDW helps tree models)
**Notes:** Base HistGBM val=0.2980. IDW correction worsened it to 0.2987. HistGBM residuals are
  not ecology-correlated (tree already fits park-specific patterns), so nearest-park correction
  adds noise. IDW correction only helps linear models whose residuals carry systematic park-level signal.
  Reverted. Final canonical model: Lasso+lag+IDW (exp048, d5554ec).

---

## Session Summary: Experiments 042–050

| Exp | Config | val_rmse | val_r2 | cv_rmse | Status |
|-----|--------|----------|--------|---------|--------|
| 042 | +SDI_lag1 (PCA+Ridge) | 0.2917 | 0.694 | 0.270 | keep |
| 043 | HistGBM+lag | 0.3031 | 0.669 | 0.236 | discard |
| 044 | Drop PCA, Ridge+lag | 0.2810 | 0.716 | 0.254 | keep |
| 045 | Lasso(0.001)+lag | 0.2781 | 0.722 | 0.250 | keep |
| 046 | HistGBM conservative+lag | 0.2980 | 0.680 | 0.234 | discard |
| 047 | Lasso+NNCorr K=5 uniform | 0.2753 | 0.727 | 0.250 | keep |
| 048 | Lasso+IDW K=5 | 0.2752 | 0.727 | 0.250 | keep ← best |
| 049 | KMeans clusters K=2..6 | 0.2958 | 0.685 | 0.250 | discard |
| 050 | HistGBM+IDW | 0.2980 | 0.680 | 0.234 | discard |

**Overall session improvement:** 0.326 → 0.275 (−0.051, −16%)
**Target not reached:** 0.275 vs 0.20 target. Gap is structural — cross-park generalization
  with 149 parks and current features has a floor near 0.275 for linear models.

**Key findings:**
1. SDI_lag1 is the dominant feature (temporal autocorrelation explains most within-park variance)
2. Removing PCA is strictly better once lag is added (PCA dilutes direct lag coefficient)
3. Lasso > Ridge for sparse selection across correlated eco/human features
4. Tree models overfit park-specific patterns; linear models generalize better cross-park
5. Nearest-park correction provides marginal gain; residuals weakly ecology-correlated
6. Park clustering infeasible with only 27 val parks

<!-- New experiments appended below -->

---

## Exp 051 — GradientBoostingRegressor(max_features=0.7) + SDI_lag1 [XGBoost proxy]

**Date:** 2026-05-18  **Commit:** discarded (git checkout -- train.py)
**Model:** GradientBoostingRegressor(n_estimators=300, max_depth=4, max_features=0.7)
**Change from base (exp048):** Replace Lasso+lag with GBM+column subsampling (max_features=0.7 mimics XGBoost feature subsampling)
**val_rmse:** ~0.350   **val_r2:** ~0.45   **cv_rmse:** ~0.230   **total_seconds:** ~15
**Status:** DISCARD — val=0.350 worse than Lasso+lag (0.278). Same park-overfitting pattern seen in all tree models. Column subsampling does not fix cross-park generalization.
**Notes:** max_features=0.7 is the closest sklearn equivalent to XGBoost colsample_bytree. Still overfits park-specific patterns; tree models learn park signatures rather than transferable ecological relationships.

---

## Exp 052 — PolynomialFeatures(degree=2, interaction_only=True) on ECO + lag1 → Lasso

**Date:** 2026-05-18  **Commit:** discarded (git checkout -- train.py)
**Model:** Lasso(alpha=0.001) + PolynomialFeatures(degree=2, interaction_only) on ECO_FEATS+SDI_lag1 (11 → 66 terms)
**Change from base:** Add all pairwise interactions between ecological features and SDI_lag1, use Lasso to select
**val_rmse:** ~0.277   **val_r2:** ~0.726   **cv_rmse:** ~0.256   **total_seconds:** ~0.5
**Status:** DISCARD — 0.278→0.277, negligible. Interaction terms add noise; Lasso selects a sparse subset but doesn't improve over direct features.
**Notes:** Cross-feature interactions (e.g., lag1 × avg_FPAR) don't generalize better cross-park than individual terms.

---

## Exp 053 — Taxon × SDI_lag1 Interaction Features

**Date:** 2026-05-18  **Commit:** discarded (git checkout -- train.py)
**Model:** Lasso(alpha=0.001) + taxon_group × SDI_lag1 manual interactions (one feature per taxon group)
**Change from base:** Explicit per-taxon autocorrelation coefficients (taxon × lag1 product terms)
**val_rmse:** ~0.279   **val_r2:** ~0.720   **cv_rmse:** ~0.253   **total_seconds:** ~0.5
**Status:** DISCARD — slightly worse (0.278→0.279). Per-taxon lag doesn't generalize to val parks. OHE taxon + direct lag is sufficient.

---

## Exp 054 — Add SDI_lag2 (AR-2 model)

**Date:** 2026-05-18  **Commit:** 39f61db
**Model:** Lasso(alpha=0.001) + SDI_lag1 + SDI_lag2
**Change:** Add SDI_lag2 (2-month prior) to extend AR(1) to AR(2)
**val_rmse:** 0.270369   **val_r2:** 0.732727   **cv_rmse:** 0.240744   **total_seconds:** 0.2
**Status:** KEEP — 0.2752→0.2704 improvement. 2-month memory adds predictive signal.

---

## Exp 055 — Add SDI_lag3 (AR-3 model)

**Date:** 2026-05-18  **Commit:** b884f47
**Model:** Lasso(alpha=0.001) + SDI_lag1 + SDI_lag2 + SDI_lag3
**Change:** Extend to AR(3)
**val_rmse:** 0.266566   **val_r2:** 0.736255   **cv_rmse:** 0.237687   **total_seconds:** 0.2
**Status:** KEEP — 0.2704→0.2666. Each additional recent lag adds signal.

---

## Exp 056 — Add SDI_lag6 (semi-annual anchor)

**Date:** 2026-05-18  **Commit:** 160a804
**Model:** Lasso(alpha=0.001) + lags 1,2,3,6
**Change:** Add SDI_lag6 (6 months prior = same season, prior half-year)
**val_rmse:** 0.259145   **val_r2:** 0.744541   **cv_rmse:** 0.228859   **total_seconds:** 0.1
**Status:** KEEP — 0.2666→0.2591 (biggest single-lag jump). Semi-annual periodicity strong signal.

---

## Exp 057 — Add SDI_lag12 (year-over-year anchor)

**Date:** 2026-05-18  **Commit:** 3dcc8f8
**Model:** Lasso(alpha=0.001) + lags 1,2,3,6,12
**Change:** Add SDI_lag12 (12 months prior = same month last year)
**val_rmse:** 0.255989   **val_r2:** 0.743770   **cv_rmse:** 0.221954   **total_seconds:** 0.1
**Status:** KEEP — 0.2591→0.2560. Year-over-year comparison adds signal; CV improves more (0.229→0.222).

---

## Exp 058 — Add SDI_lag4, SDI_lag5 (intermediate lags)

**Date:** 2026-05-18  **Commit:** discarded (git checkout -- train.py)
**Model:** Lasso(alpha=0.001) + lags 1,2,3,4,5,6,12
**Change:** Add lag4 and lag5 (months 4 and 5 prior, between 3-month and 6-month anchors)
**val_rmse:** ~0.2560   **val_r2:** ~0.744   **cv_rmse:** ~0.222   **total_seconds:** ~0.3
**Status:** DISCARD — identical to lag3+lag6+lag12 alone. Intermediate lags fully redundant once lag3 and lag6 are present. No new information.

---

## Exp 059 — Add SDI_lag9 (Q3 seasonal anchor)

**Date:** 2026-05-18  **Commit:** 10f8e2b
**Model:** Lasso(alpha=0.001) + lags 1,2,3,6,9,12
**Change:** Add SDI_lag9 (9 months prior = one quarter before lag12)
**val_rmse:** 0.252799   **val_r2:** 0.750117   **cv_rmse:** 0.219851   **total_seconds:** 0.1
**Status:** KEEP — 0.2560→0.2528, R2 0.744→0.750. Quarterly seasonal pattern (lag9 ≈ same season, 3 months earlier).

---

## Exp 060 — ElasticNet(alpha=0.001, l1_ratio=0.5) on full lag set

**Date:** 2026-05-18  **Commit:** fb39117
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.5) + lags 1,2,3,6,9,12
**Change:** Replace Lasso with ElasticNet to handle correlated lags (L2 component prevents hard zeroing)
**val_rmse:** 0.252059   **val_r2:** 0.751578   **cv_rmse:** 0.219380   **total_seconds:** 0.2
**Status:** KEEP — 0.2528→0.2521. ElasticNet distributes weight across correlated lags slightly better than Lasso.

---

## Exp 061 — ElasticNet l1_ratio grid search → l1_ratio=0.2 best

**Date:** 2026-05-18  **Commit:** 66607dd
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + lags 1,2,3,6,9,12
**Change:** Grid search over l1_ratio (0.1, 0.2, 0.3, 0.5, 0.7, 0.9) — l1_ratio=0.2 best
**val_rmse:** 0.251674   **val_r2:** 0.752337   **cv_rmse:** 0.219179   **total_seconds:** 0.2
**Status:** KEEP — 0.2521→0.2517. More L2 weight (l1_ratio=0.2 ≈ mostly ridge) better for correlated lags.

---

## Exp 062 — IDW Nearest-Park Correction on ElasticNet+Lags

**Date:** 2026-05-18  **Commit:** 130217f
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + IDW correction (ECO+HUM+GEO K=5)
**Change:** Apply IDW residual correction on top of ElasticNet+lag model (same as exp048 but with multi-lag base)
**val_rmse:** 0.251603   **val_r2:** 0.752476   **cv_rmse:** 0.219179   **total_seconds:** 0.2
**Status:** KEEP (marginal) — 0.2517→0.2516. IDW provides negligible improvement on stronger base model.
**Notes:** IDW correction was more useful on weaker base (exp047/048, lag1 only). With 6 lags, model residuals are less spatially correlated.

---

## Exp 063 — Add SDI_cumean (park×taxon all-time expanding mean)

**Date:** 2026-05-18  **Commit:** c87262a
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + lags 1,2,3,6,9,12 + SDI_cumean
**Change:** Add expanding mean of all prior SDI values per park×taxon (park-specific baseline level)
**val_rmse:** 0.248553   **val_r2:** 0.758439   **cv_rmse:** 0.215166   **total_seconds:** 0.3
**Status:** KEEP — 0.2516→0.2486, R2 0.752→0.758. Cumulative mean captures park×taxon long-run baseline. New information beyond the lag structure.

---

## Exp 064 — Add SDI_month_cumean (seasonal baseline per park×taxon×month)

**Date:** 2026-05-18  **Commit:** 8b9c73a
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + lags 1,2,3,6,9,12 + SDI_cumean + SDI_month_cumean
**Change:** Add expanding mean of same-calendar-month prior SDI values per park×taxon (seasonal baseline)
**val_rmse:** 0.245588   **val_r2:** 0.764170   **cv_rmse:** 0.213279   **total_seconds:** 4.3
**Status:** KEEP — 0.2486→0.2456, R2 0.758→0.764. Month-specific baseline more precise than all-months cumean; "average June SDI for these Aves in Yellowstone."

---

## Exp 065 — HistGBM(max_iter=500) + full feature set (lags + cumeans)

**Date:** 2026-05-18  **Commit:** discarded (git checkout -- train.py)
**Model:** HistGradientBoostingRegressor(max_iter=500) + lags 1,2,3,6,9,12 + SDI_cumean + SDI_month_cumean
**Change:** Replace ElasticNet with HistGBM given the strong temporal feature set
**val_rmse:** ~0.261   **val_r2:** ~0.740   **cv_rmse:** ~0.235   **total_seconds:** ~20
**Status:** DISCARD — val=0.261 worse than ElasticNet 0.246. Tree models continue to overfit park signatures. With rich temporal features, linear model generalizes better.


---

## Exp 066 — Add SDI_dev (deviation of lag1 from seasonal norm)

**Date:** 2026-05-18  **Commit:** 7e9702a
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + lags+cumeans+dev
**Change:** Add SDI_dev = SDI_lag1 - SDI_month_cumean. Captures whether last month's SDI was above/below seasonal norm for this park×taxon×month. Regression-to-mean signal.
**val_rmse:** 0.244347   **val_r2:** 0.766547   **cv_rmse:** 0.211327   **total_seconds:** 1.0
**Status:** KEEP — 0.2456→0.2434. SDI_dev adds directional information the model benefits from having pre-computed.

---

## Exp 067 — Add n_prior_months (cumean reliability signal)

**Date:** 2026-05-18  **Commit:** d06b6bc
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + all prior features + n_prior_months
**Change:** Add n_prior_months = count of prior SDI observations for this park×taxon (cumcount). Park×taxa with few observations have noisier cumean; model can weight accordingly.
**val_rmse:** 0.244103   **val_r2:** 0.767013   **cv_rmse:** 0.211136   **total_seconds:** 1.0
**Status:** KEEP — marginal 0.2443→0.2441. Reliability signal helps slightly.

---

## Exp 068 — Add SDI_dev12 (same-month last year vs seasonal norm)

**Date:** 2026-05-18  **Commit:** 39acb66
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + all prior features + SDI_dev12
**Change:** Add SDI_dev12 = SDI_lag12 - SDI_month_cumean. Interannual variability: was last year's same month above/below the multi-year average? Complements SDI_dev.
**val_rmse:** 0.243872   **val_r2:** 0.767453   **cv_rmse:** 0.211000   **total_seconds:** 1.1
**Status:** KEEP — marginal 0.2441→0.2439.

---

## Exp 069a — traffic_lag1 (lagged monthly traffic) [discarded]

**Date:** 2026-05-18  **Commit:** discarded (git checkout -- train.py)
**Change:** Add monthly_traffic(t-1) as feature — human pressure last month affecting biodiversity this month.
**val_rmse:** ~0.243878   **Status:** DISCARD — essentially identical (0.000006 worse). Human pressure lag adds no signal on top of the rich SDI temporal features already in model.

---

## Exp 069 — Taxon×Season interaction features

**Date:** 2026-05-18  **Commit:** b383c37
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + 63 numeric features
**Change:** Add tg_{taxon}_sin and tg_{taxon}_cos for each of 13 taxon groups (26 new features). Allows model to learn per-taxon seasonal curves — Birds peak in spring, Insects in summer, etc.
**val_rmse:** 0.243534   **val_r2:** 0.768097   **cv_rmse:** 0.211134   **total_seconds:** 1.2
**Status:** KEEP — 0.2439→0.2435. 13 taxon groups × 2 seasonal terms = 26 features.

---

## Exp 070 — Taxon×year_norm temporal trend interactions

**Date:** 2026-05-18  **Commit:** c40aed8
**Model:** ElasticNet(alpha=0.001, l1_ratio=0.2) + 76 numeric features
**Change:** Add tg_{taxon}_year for each taxon group (13 new features). Allows model to learn per-taxon temporal trends over 2021–2025 (recovery from pandemic, climate shifts, etc.).
**val_rmse:** 0.243479   **val_r2:** 0.768203   **cv_rmse:** 0.211336   **total_seconds:** 1.3
**Status:** KEEP — marginal 0.24353→0.24348.
**Notes:** Now using 76 numeric features + taxon OHE. Improvements are becoming very small (~0.0001 per experiment). The model may be approaching the cross-park generalization ceiling for linear methods on this dataset.

---

## Exp 071 — SDI_park_lag1 (cross-taxon park-month mean lag1)

**Date:** 2026-05-18  **Commit:** discarded (git checkout -- train.py)
**Change:** Add mean of SDI_lag1 across ALL taxa for this (park, year, month) — cross-taxon park health signal.
**val_rmse:** ~0.243614   **Status:** DISCARD — 0.2435→0.2436 slightly worse. Cross-taxon mean is redundant given own-taxon lag1 is already in model.

---

## Current State (as of Exp 070)

**Best model:** ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000)
**Feature count:** 76 numeric + taxon_group OHE (13 groups)
**val_rmse:** 0.24348   **val_r2:** 0.768   **cv_rmse:** 0.211
**Experiments run:** 71 (032–071 on rarefied SDI; 025–031 in data structure study)

**Feature groups:**
- 28 base features: ECO (10) + HUM (8) + GEO (2) + TIME (3) + TRAF (2) + log_n_obs (1)
- 6 temporal lags: SDI_lag1,2,3,6,9,12
- 2 baseline features: SDI_cumean, SDI_month_cumean
- 2 deviation features: SDI_dev (=lag1−month_cumean), SDI_dev12 (=lag12−month_cumean)
- 1 reliability feature: n_prior_months
- 39 taxon×time interactions: 13 taxa × (sin, cos, year_norm)
- 1 categorical (OHE): taxon_group

**Gap to target:** val_rmse 0.244 vs 0.20 target = 0.044 remaining.
**Rate of improvement:** ~0.0001–0.0003 per experiment in recent rounds.
**Assessment:** Model is in diminishing-returns territory. Remaining gap likely from cross-park heterogeneity not captured by available static features. A mixed-effects model or richer per-park dynamic data would be needed to close this gap.

