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
