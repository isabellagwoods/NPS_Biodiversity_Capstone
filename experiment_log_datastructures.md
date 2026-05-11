# Experiment Log — Data Structure Study
**Project:** NPS Biodiversity SDI Prediction
**Script:** `train_datastructures.py`
**Model:** Ridge regression (alpha=100) — linear model only throughout
**Split:** Park-based (all data from a park stays in one split, no cross-park leakage)
**Data:** 2021+ only

---

## Study Goal

The pool of 152 parks limits model capacity. This study tests whether restructuring the
data — by organism group, time aggregation, or temporal dynamics — meaningfully increases
effective sample size and improves linear predictability of biodiversity (SDI).

Each experiment changes ONE thing: the data structure. The model (Ridge) and evaluation
protocol (park-based split, RMSE/R²) stay fixed.

---

## Experiment Structures Tested

| ID | Structure | N (approx) | Target | Research Question |
|----|-----------|-----------|--------|-------------------|
| A | Taxon-group SDI per park | ~900 | SDI per organism group | Does SDI vary by organism group after accounting for ecology? |
| B | Monthly + geography | ~5,552 | SDI_monthly | Do location/latitude predict monthly SDI beyond ecology? |
| C | First-differences | ~5,400 | ΔSDI monthly | Does Δtraffic predict ΔSDI across parks? |
| D | Yearly aggregated | ~803 | Annual mean SDI | Does annual human pressure predict annual biodiversity? |
| E | Temporal lag (AR1) | ~5,403 | SDI_monthly | Does last month's SDI predict this month's? |
| F | Taxon × Month × Park | ~15k+ | SDI per group per month | What conditions drive within-group monthly diversity? |

---

## Experiment 025 — Structure A: Taxon-Group SDI per Park

**Date:** 2026-05-11
**Commit:** da626f9
**Model:** Ridge(alpha=100)
**Structure:** One row per (park × organism group). SDI computed from 2021+ iNat observations
  within each taxon group within each park. Taxon group is one-hot encoded as a feature.
  Features: ecological (FPAR, SNOW, burn, ET_range, GPP_range) + human (traffic, visitors,
  pop_density, etc.) + log(n_obs_in_group) + taxon_group (one-hot)
**Research question:** Does SDI vary by organism group after accounting for a park's ecological features?
**Why this helps with small N:** Multiplies rows from 152 parks to 1,462 park×group rows.
  Each organism group is a separate "experiment" of how the same park's ecology supports diversity.
**Split:** Park-based — all taxon groups from a park stay in one split.

**n_train:** 1159   **n_val:** 303
**val_rmse:** 0.7138   **val_r2:** 0.7679   **cv_rmse:** 0.6504
**total_seconds:** 6.2
**Status:** keep
**Notes:** R²=0.77 is the best result yet — far above any prior experiment. Taxon group
  identity is a powerful feature (different organism groups have very different baseline SDIs).
  Ecological features (FPAR, SNOW, burn) align with organism group differences. RMSE is higher
  than monthly in absolute terms because per-group SDI has higher variance than aggregate SDI.
  Key finding: organism group membership is the dominant source of SDI variation across parks.

---

## Experiment 026 — Structure B: Monthly SDI + Geographic Features

**Date:** 2026-05-11
**Commit:** (see run.log)
**Model:** Ridge(alpha=100)
**Structure:** One row per (park × year × month), same as Stage 2 in train.py.
  Added: one-hot encoded US biogeographic region (7 regions) + explicit lat/lon from park table.
  Features: ecological + human + month_sin/cos + year_norm + lat + lon + region
**Research question:** Do biogeographic region and latitude predict monthly SDI beyond ecological features?
**Why interesting:** Latitude captures climate gradients (temperature, precipitation patterns)
  that the MODIS satellite features don't fully encode. Region captures biogeographic context.
**Split:** Park-based.

**n_train:** 4340   **n_val:** 1212
**val_rmse:** 0.9470   **val_r2:** 0.3086   **cv_rmse:** 0.8163
**total_seconds:** 0.1
**Status:** keep
**Notes:** Adding region one-hot and lat/lon to the monthly model with Ridge gives R²=0.31.
  Substantially better than random but worse than Structure A. The park-based split means
  the linear model must generalize across parks without park-level intercepts. Region and lat/lon
  help but don't close the gap. The RMSE (0.947) is comparable to the HistGBM baseline on park-split
  (0.936), confirming that the geographic features don't rescue the cross-park challenge for a linear model.
  Key finding: geography (lat, lon, region) explains only ~31% of cross-park SDI variation linearly.

---

## Experiment 027 — Structure C: First-Differences (ΔSDI ~ Δtraffic)

**Date:** 2026-05-11
**Commit:** (see run.log)
**Model:** Ridge(alpha=100)
**Structure:** One row per (park × month) where target is the month-to-month change in SDI.
  ΔSDI = SDI(t) - SDI(t-1), computed within each park's time series.
  Δtraffic = monthly_traffic(t) - monthly_traffic(t-1).
  Features: Δtraffic + month_sin/cos + year_norm + park-level human predictors
**Research question:** Does month-to-month change in traffic predict month-to-month change in biodiversity, across parks?
**Why no leakage:** Diffs computed within each park; val parks are entirely unseen during training.
  No park mean or fixed effect is used.
**Why interesting:** Stationarizes the time series; removes park-level intercept differences;
  tests the causal direction of human impact on biodiversity change.
**Split:** Park-based.

**n_train:** 4219   **n_val:** 1184
**val_rmse:** 0.6109   **val_r2:** 0.0492   **cv_rmse:** 0.6159
**total_seconds:** 0.1
**Status:** keep
**Notes:** RMSE looks low (0.611) but that's because the target is ΔSDI not SDI — the
  standard deviation of month-over-month changes is smaller. R²=0.05 tells the real story:
  a linear model explains only 5% of the variance in biodiversity CHANGES. Month-to-month
  Δtraffic and seasonal features cannot predict Δ SDI across parks. Key finding: biodiversity
  LEVEL is predictable (from ecology, park identity), but biodiversity CHANGES are near-random
  at the monthly scale for a linear model. Suggests stochastic within-park dynamics dominate
  short-term fluctuations, not measured human-activity signals.

---

## Experiment 028 — Structure D: Yearly Aggregated Monthly SDI

**Date:** 2026-05-11
**Commit:** (see run.log)
**Model:** Ridge(alpha=100)
**Structure:** Monthly data aggregated to yearly means per park.
  One row per (park × year). N ≈ 803 rows (149 parks × ~5 years).
  Target: mean(SDI_monthly) per park per year.
  Features: ecological (park-level) + yearly avg traffic + yearly avg visitors + year_norm + lat/lon
**Research question:** Does average annual human pressure predict average annual biodiversity?
**Why interesting:** Yearly aggregation reduces within-season noise; tests whether the annual
  rhythm of human activity is the relevant scale for biodiversity prediction.
**Split:** Park-based (all years from a park stay together).

**n_train:** 638   **n_val:** 165
**val_rmse:** 0.6292   **val_r2:** 0.6152   **cv_rmse:** 0.5526
**total_seconds:** 0.1
**Status:** keep
**Notes:** Yearly aggregation dramatically improves linear R²: 0.62 vs 0.31 for monthly.
  By averaging 12 months of observations, within-season noise cancels, leaving the
  park-level and annual signal that ecological and human features can capture linearly.
  N=803 vs 5552 (monthly) but the signal-to-noise ratio is much better. RMSE=0.629 is
  comparable. Key finding: the right temporal grain for a linear model is ANNUAL, not monthly.
  Annual averages of human pressure and biodiversity are linearly related.

---

## Experiment 029 — Structure E: Temporal Lag / AR(1)

**Date:** 2026-05-11
**Commit:** (see run.log)
**Model:** Ridge(alpha=100)
**Structure:** Add SDI_monthly(t-1) as a predictor for SDI_monthly(t).
  Lag computed within each park's own time series (sort by park, year, month; shift by 1).
  Features: all monthly + ecological + park human + SDI_lag1 + time
**Research question:** Is next month's biodiversity predictable from last month's level,
  beyond ecological and human-activity features?
**Why no leakage:** Lag is the park's OWN prior month, not data from other parks.
  For val parks, the lag is their own t-1 — no training information crosses over.
**Why interesting:** Tests temporal autocorrelation. If lag1 is a strong predictor, biodiversity
  is persistent (months-long dynamics dominate). If weak, instantaneous conditions matter more.
**Split:** Park-based.

**n_train:** 4219   **n_val:** 1184
**val_rmse:** 0.5927   **val_r2:** 0.5960   **cv_rmse:** 0.5787
**total_seconds:** 0.1
**Status:** keep
**Notes:** Adding SDI_lag1 doubles R² from 0.31 (monthly, no lag) to 0.60. The prior month's
  SDI is the single strongest linear predictor of current month SDI — more powerful than all
  ecological and human features combined. Biodiversity is temporally autocorrelated (months-long
  persistence). Val and CV RMSE both improve (0.593 val, 0.579 CV). No leakage: lag is the
  park's own prior observed month, not data from other parks.
  Key finding: temporal autocorrelation is the dominant structure in monthly SDI data.
  For unseen parks, the lag feature is the most valuable predictor available.

---

## Experiment 030 — Structure F: Monthly SDI per Taxon Group × Park

**Date:** 2026-05-11
**Commit:** (see run.log)
**Model:** Ridge(alpha=100)
**Structure:** Richest structure — one row per (park × year × month × taxon_group).
  SDI computed from raw iNaturalist observations within each cell (≥5 obs required).
  One-hot taxon_group + time + monthly traffic/visitors + ecological + lat/lon.
**Research question:** Across organism groups and months, what ecological and
  human conditions drive within-group species diversity?
**Why interesting:** Largest possible N from available data. Taxon group acts as a
  random-effect-like feature — allows the model to learn group-specific baseline diversity
  levels while sharing ecological/human coefficient estimates.
**Split:** Park-based (all months × groups from a park stay together).

**n_train:** 21743   **n_val:** 6613
**val_rmse:** 0.4427   **val_r2:** 0.8444   **cv_rmse:** 0.3919
**total_seconds:** 11.6
**Status:** keep
**Notes:** Best result of the entire study. R²=0.844 with a linear model! N=28,356 rows
  (taxon × month × park). Organism group is by far the strongest feature — each group has a
  distinct baseline SDI (Plantae >> Insecta >> Aves >> Fungi etc.). Within that structure,
  ecological conditions (FPAR, SNOW, burn) modulate group-specific diversity levels.
  RMSE=0.443 is the lowest across all experiments. CV RMSE=0.392 suggests modest overfitting
  (extra features from monthly merges have some NaN that imputation handles).
  Key finding: combining organism-group disaggregation WITH temporal structure creates the
  richest, most linearly predictable dataset. The key is that organism groups provide
  a natural clustering that a linear model can exploit — one model coefficient per group.

---

## Summary Table (filled in as experiments complete)

| Exp | Structure | N_train | N_val | val_rmse | val_r2 | cv_rmse | Status |
|-----|-----------|---------|-------|----------|--------|---------|--------|
| 025 | A: Taxon-group SDI/park | 1,159 | 303 | 0.7138 | **0.768** | 0.650 | keep |
| 026 | B: Monthly + geography | 4,340 | 1,212 | 0.9470 | 0.309 | 0.816 | keep |
| 027 | C: First-differences (ΔSDI) | 4,219 | 1,184 | 0.6109 | 0.049 | 0.616 | keep |
| 028 | D: Yearly aggregated | 638 | 165 | 0.6292 | **0.615** | 0.553 | keep |
| 029 | E: Temporal lag AR(1) | 4,219 | 1,184 | 0.5927 | **0.596** | 0.579 | keep |
| 030 | F: Taxon × Month × Park | 21,743 | 6,613 | **0.4427** | **0.844** | 0.392 | keep |

**Controlled variable:** Data structure (one change per experiment)
**Fixed:** Ridge(alpha=100), park-based split, 2021+ data, val_rmse metric

### Key Findings

1. **Organism-group disaggregation is the most powerful structural choice** (A, F).
   Treating each taxon group as a separate row gives a linear model a coefficient to learn
   per group — the dominant source of SDI variation is WHICH organism group, not which park.

2. **Temporal aggregation matters** (D vs B). Annual averages (R²=0.62) outperform raw monthly
   (R²=0.31) with the same features. Within-season noise drowns the ecological signal at
   monthly resolution with a linear model.

3. **Temporal autocorrelation is the strongest monthly predictor** (E). Adding one lag feature
   (SDI at t-1) nearly doubles R² from 0.31 to 0.60. Biodiversity is persistent month-to-month.

4. **Month-to-month CHANGES are unpredictable linearly** (C). R²=0.05 for ΔSDI confirms that
   the ecological/human features in this dataset do not capture what drives short-term changes.
   Stochastic population dynamics dominate at the sub-annual scale.

5. **Best structure: Taxon × Month × Park** (F). Combining organism-group disaggregation with
   temporal structure achieves R²=0.844 and RMSE=0.443 — the best linear result. Largest N
   (28k rows) from the same 152 parks, no additional data collection needed.

### Recommended next question
*"Within the F structure, does the relationship between human impact and SDI vary by organism group?"*
→ Add taxon_group × traffic interaction terms (multiplicative features) or run separate Ridge
  models per group. This would test whether traffic suppresses plant vs. insect diversity differently.

<!-- New experiments appended below -->
