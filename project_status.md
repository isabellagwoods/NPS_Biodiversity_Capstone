# NPS Biodiversity Capstone — Project Status
### Northwestern STAT 390 | Isabella Woods | May 2026

---

## 1. What This Project Is Actually Doing

### The Research Question
**Does human activity reduce biodiversity in US National Parks, and can we predict biodiversity from ecological and human impact data?**

### The Data
iNaturalist citizen science observations from 2021–2025 across 149 national parks, merged with park-level ecological features (NASA satellite data), human impact features (NPS visitation, traffic, air quality, population density, toxic releases), and geographic features (lat/lon, land cover, elevation).

### The Target Variable
**Rarefied Shannon Diversity Index (SDI_rarefied)** — a measure of species diversity per park × month × taxonomic group. "Rarefied" means each cell is subsampled to exactly 50 observations before computing SDI, removing the bias introduced by popular parks having more iNaturalist observers. This produces ~8,800 rows across 149 parks and 13 taxonomic groups (Birds, Mammals, Plants, Insects, Fungi, etc.).

### What the Model Does
A single-stage **ElasticNet regression** predicts the rarefied SDI for a given park × month × taxon combination using 76 numeric features and 1 categorical feature (taxon group):

- **Ecological features (10):** FPAR, snow cover, burn events, ET range, GPP range
- **Human impact features (8):** recreation visitors, traffic, pollution facilities, population density
- **Geographic features (2):** latitude, longitude
- **Temporal features (3):** month_sin, month_cos, year_norm
- **Traffic features (2):** monthly traffic, annual visitors
- **Autoregressive lags (6):** SDI at 1, 2, 3, 6, 9, and 12 months prior — the strongest predictors
- **Baseline features (2):** all-time cumulative mean SDI, same-month cumulative mean SDI
- **Deviation features (2):** deviation of last month's SDI from seasonal norm
- **Reliability feature (1):** number of prior observations for this park × taxon
- **Taxon × time interactions (39):** per-taxon seasonal curves and temporal trends
- **Categorical (1):** taxon group (one-hot encoded, 13 groups)

### The AutoResearch Loop
An AI agent autonomously edits only **Section 2 of train.py**, runs experiments, logs results to `results.tsv` and `experiment_log.md`, keeps improvements and reverts failures — without human intervention. 71 experiments have been run to date across two modeling phases.

### Current Performance
- **val_rmse = 0.2435** (target: ≤ 0.20)
- **val_r2 = 0.768** (the model explains 77% of variance in biodiversity)
- **cv_rmse = 0.211** (5-fold cross-validation RMSE)
- **Gap to target:** 0.043 RMSE remaining

---

## 2. Experiment Results Table

| Phase | Exp # | Change Made | val_rmse | val_r2 | Status | Key Finding |
|-------|--------|-------------|----------|--------|--------|-------------|
| **Baseline** | 001 | Ridge(alpha=1) baseline | 0.697 | -0.04 | keep | Severe underfitting — fill values contaminating features |
| **Data cleaning** | 002 | HistGBM stage 1 | 0.834 | -0.49 | discard | Tree models overfit on 121 parks |
| | 003 | Remove fill-value features | 0.661 | 0.06 | keep | R² turned positive — confirmed fill values were the root cause |
| **Regularization** | 005 | Ridge alpha=10 | 0.655 | 0.08 | keep | More regularization helps with small n |
| | 006 | Ridge alpha=100 | 0.648 | 0.10 | keep | Best linear baseline |
| | 007 | Ridge alpha=1000 | 0.670 | 0.04 | discard | Over-regularized |
| **Architecture** | 008 | RandomForest stage 1 | 0.719 | -0.11 | discard | Trees overfit with 121 training parks |
| | 009 | HistGBM stage 2 | 0.477 | 0.78 | keep | Major jump — HistGBM suited for monthly data |
| | 012 | SelectKBest(k=8) | 0.644 | 0.11 | keep | Dropping weak features helped marginally |
| **Restructure to rarefied SDI** | 032 | Ridge + rarefied target | 0.326 | — | keep | New target removes effort bias; reset baseline |
| | 033-038 | Alpha + PCA search | 0.326→0.310 | — | mixed | PCA(n=10)+Ridge best structure |
| **Temporal lags** | 042 | Add SDI_lag1 | 0.290 | — | keep | Single biggest improvement — last month's SDI dominates |
| | 043 | Add SDI_lag12 | 0.271 | — | keep | Same-month last year adds seasonal anchor |
| | 044 | Add SDI_lag3 | 0.264 | — | keep | Quarterly lag adds pattern |
| | 045 | Add SDI_lag6 | 0.260 | — | keep | Semi-annual signal captured |
| | 059 | Add SDI_lag9 | 0.253 | 0.750 | keep | Full seasonal lag set established |
| **Regularization tuning** | 060 | ElasticNet(l1_ratio=0.5) | 0.252 | 0.752 | keep | ElasticNet handles correlated lags better than Lasso |
| | 061 | ElasticNet(l1_ratio=0.2) | 0.252 | 0.752 | keep | More L2 weight optimal for correlated features |
| **Baseline features** | 063 | Add SDI_cumean | 0.249 | 0.758 | keep | Park×taxon long-run baseline adds signal |
| | 064 | Add SDI_month_cumean | 0.246 | 0.764 | keep | Seasonal baseline more precise than all-months mean |
| **Deviation + interactions** | 066 | Add SDI_dev | 0.244 | 0.767 | keep | Deviation from seasonal norm adds regression-to-mean signal |
| | 067 | Add n_prior_months | 0.244 | 0.767 | keep | Reliability signal marginally useful |
| | 068 | Add SDI_dev12 | 0.244 | 0.767 | keep | Interannual deviation marginal gain |
| | 069 | Taxon×season interactions | 0.244 | 0.768 | keep | Per-taxon seasonal curves (26 features) |
| | 070 | Taxon×year interactions | **0.243** | **0.768** | keep | Per-taxon trends — **current best** |
| **Discarded structurals** | 065 | HistGBM on full feature set | 0.261 | 0.740 | discard | Linear model generalizes better than trees with rich temporal features |
| | 071 | Cross-taxon park mean lag | 0.244 | — | discard | Redundant with own-taxon lag1 |

**Total improvement: 0.697 → 0.243 (−65%) over 71 experiments**

---

## 3. Committed Schedule for Remaining Work

### Context
Current val_rmse = 0.2435, target = 0.20. Rate of improvement in recent experiments: ~0.0001–0.0003 per run. The model is in diminishing-returns territory on linear feature engineering. Closing the remaining 0.043 gap likely requires a structural change rather than more feature additions.

---

### Week of May 19–23 — Close the Gap or Pivot

**Goal:** Make a decision about whether the 0.20 target is reachable and commit to a final model.

**AutoResearch experiments to run (Tier 1 — most likely to help):**

| Experiment | What to try | Expected gain |
|------------|-------------|---------------|
| Exp 072 | Taxon×ECO interactions: tg_{taxon} × avg_FPAR | ~0.001–0.003 |
| Exp 073 | Taxon×HUM interactions: tg_{taxon} × pop_density | ~0.001–0.003 |
| Exp 074 | ElasticNet alpha=0.0005 (loosen regularization slightly) | ~0.001 |
| Exp 075 | SDI_park_cumean: mean SDI across all taxa for this park | ~0.001–0.002 |
| Exp 076 | SDI rolling variance (volatility signal, window=3) | ~0.001 |
| Exp 077 | Separate models per taxon group (13 models) | ~0.005–0.015 |

**Decision point (end of week):**
- If val_rmse reaches ≤ 0.20 → **target met, lock the model**
- If val_rmse is 0.21–0.22 → **report honest results, pivot to interpretation**
- If val_rmse is still 0.24+ → **accept 0.243 as final, write up why 0.20 is structurally hard**

**Deliverable:** Updated `results.tsv`, `experiment_result_matrix.csv`, and `metric_over_time.png`

---

### Week of May 26–30 — Analysis & Presentation Prep

**Goal:** Extract findings, build visualizations, write narrative.

**Monday–Tuesday: Feature importance analysis**
- Extract ElasticNet coefficients — which features have the largest weights?
- Rank: autoregressive lags vs ecological vs human impact features
- Key question to answer: *after controlling for temporal patterns, do human impact features still matter?*

**Wednesday: Unsupervised analysis**
- Run K-means clustering on park-level features (k=4–6) — identify park archetypes
- Run PCA and plot parks in 2D — visualize which parks are similar
- Identify the biggest positive and negative residuals (overachievers and underperformers)

**Thursday: Image classifier status**
- Check training progress on multimodal native vs introduced classifier
- If val_accuracy ≥ 0.70: include as a supplementary finding
- If not converged: report as work in progress

**Friday: Build presentation deck**

Recommended slide structure (15 slides, ~12 minutes):

| Slide | Content |
|-------|---------|
| 1 | Title + research question |
| 2 | Why national park biodiversity? Motivation |
| 3 | Data overview — 18 sources, 4M+ observations, 149 parks |
| 4 | What is rarefied SDI? Why does it matter? |
| 5 | Model architecture — feature groups, AutoResearch loop |
| 6 | Experiment results table — 71 experiments, 0.697→0.243 |
| 7 | Metric-over-time plot — show the improvement curve |
| 8 | Key finding 1: temporal lags dominate (lag1, lag12) |
| 9 | Key finding 2: human impact features — which matter? |
| 10 | Park archetypes — K-means clustering results |
| 11 | Residuals — which parks over/underperform? |
| 12 | Image classifier — native vs introduced |
| 13 | Limitations and what would improve the model |
| 14 | Conclusions |
| 15 | Q&A |

---

### Hard Deadlines

| Date | Deliverable |
|------|-------------|
| May 23 | Final model locked — no more AutoResearch after this |
| May 26 | Feature importance analysis complete |
| May 28 | Unsupervised analysis complete |
| May 29 | Presentation draft complete |
| May 30 | Presentation rehearsed and finalized |

---

### The Honest Assessment

The current model (val_rmse=0.243, val_r2=0.768) is already a strong result. A model that explains 77% of variance in monthly biodiversity across 149 parks using publicly available data is scientifically meaningful. The most important finding is already clear: **prior biodiversity (autoregressive lags) is by far the strongest predictor**, which means biodiversity is highly temporally persistent — a park that was biodiverse last month will likely be biodiverse this month. Human impact features add explanatory power on top of that baseline, but the question of *how much* is what the presentation should answer.

The 0.20 RMSE target was ambitious for 149 parks. If you don't hit it, that is itself a finding worth presenting — it means cross-park heterogeneity is the dominant source of variance, and no amount of feature engineering on available data will fully close that gap without park-specific models or richer temporal data.
