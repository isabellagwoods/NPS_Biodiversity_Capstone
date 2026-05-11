# NPS Biodiversity Prediction — Research Summary
**STAT 390 Capstone · Northwestern University · Isabella Woods · 2026-05-11**
**Branch:** `autoresearch/apr27`  |  **Best commit:** `7104b47`  |  **Experiments run:** 41

---

## Project at a Glance

**Goal:** Predict Shannon Diversity Index (SDI) for National Park Service biodiversity from
ecological satellite features and human-impact metrics.

**Dataset:** iNaturalist observations (2021–2025), 149 parks, merged with MODIS satellite
features (FPAR, SNOW, GPP, ET) and park-level human-activity features (traffic, visitors,
population density).

**Key problem solved:** The raw iNaturalist data has a 20× observer-effort growth 2010–2025
(r = 0.88 between log(n_obs) and SDI), inflating apparent diversity over time. Solved via
**rarefaction** (subsample each park × month × taxon cell to N=50 observations before computing
SDI), which reduced the effort correlation to r = 0.35 and eliminated the temporal trend.

**Architecture evolution:**
- Started: two-stage park-level model (Stage 1 = ecology → SDI; Stage 2 = human features → residual)
- Ended: single-stage model predicting rarefied SDI per (park × month × taxon_group), giving
  8,824 rows from 149 parks

**Best result:** val RMSE = **0.326**, val R² = **0.629** (PCA(n=10) + Ridge(α=100))

---

## Figures

> **Figure 1** — `outputs/summary_metric_trajectory.png`
> Full RMSE and R² trajectory across all 34 logged experiments, colored by phase.
> Circles = keep, × = discard.

> **Figure 2** — `outputs/summary_panels.png`
> Left: keep/discard counts by phase. Center: data structure study R² by framing.
> Right: best result vs. very first run, head-to-head metric comparison.

---

## Complete Experiment Log Bundle

### Phase 1 — Two-Stage Model, Leaky Split (Exp 001–015)
*Split: random row-level (months from a park split across train/test — leaky)*
*Stage 1 target: park-level mean SDI. Stage 2 target: monthly residual.*

| # | Commit | S1 Model | S2 Model | s1_val_rmse | s1_val_r2 | s2_val_rmse | s2_val_r2 | Status | Key change |
|---|--------|----------|----------|-------------|-----------|-------------|-----------|--------|------------|
| 001 | dae3c49 | Ridge(α=1) | Ridge(α=1) | 0.697 | −0.043 | 0.831 | 0.329 | keep | Baseline — fix string features |
| 002 | 3b54bbd | HistGBM | Ridge | 0.834 | −0.492 | 0.831 | 0.329 | discard | Stage 1 HistGBM — overfits 121 parks |
| 003 | 4d5bbdd | Ridge(α=1) | Ridge | 0.661 | 0.061 | 0.831 | 0.329 | keep | Trim 6 MODIS fill-value features |
| 004 | b9026b4 | Ridge(α=0.1) | Ridge | 0.662 | 0.059 | 0.831 | 0.329 | discard | α=0.1 no improvement |
| 005 | 2e8fc7d | Ridge(α=10) | Ridge | 0.655 | 0.079 | 0.831 | 0.329 | keep | α=10 small gain |
| 006 | 731aea2 | Ridge(α=100) | Ridge | 0.648 | 0.098 | 0.831 | 0.329 | keep | α=100 monotone improvement |
| 007 | 006204a | Ridge(α=1000) | Ridge | 0.670 | 0.036 | 0.831 | 0.329 | discard | Over-regularized |
| 008 | e90fbd6 | RandomForest | Ridge | 0.719 | −0.110 | 0.831 | 0.329 | discard | RF overfits small park n |
| 009 | 3afcb79 | Ridge(α=100) | HistGBM(200) | 0.648 | 0.098 | 0.478 | 0.779 | keep | Stage 2 HistGBM: **0.831→0.478** |
| 010 | 5d6f43d | Ridge(α=100) | HistGBM(500) | 0.648 | 0.098 | 0.475 | 0.781 | keep | max_iter=500 marginal gain |
| 011 | 26af516 | Ridge(α=100) | HistGBM | 0.656 | 0.077 | 0.475 | 0.781 | discard | log1p target transform — slight loss |
| 012 | 985c25f | SelectKBest(k=8)+Ridge | HistGBM | 0.644 | 0.109 | 0.475 | 0.781 | keep | Feature selection improves Stage 1 |
| 013 | 7d48cf3 | SelectKBest(k=6)+Ridge | HistGBM | 0.649 | 0.096 | 0.475 | 0.781 | discard | k=6 worse than k=8 |
| 014 | c4a542d | SelectKBest(k=8)+Ridge | HistGBM | 0.644 | 0.109 | 0.474 | 0.782 | keep | Trim Stage 2 to top-8 human features |
| 015 | c808cdf | SelectKBest(k=8)+Ridge | HistGBM(lr=0.05) | 0.644 | 0.109 | 0.478 | 0.779 | discard | Lower lr=0.05 worse than default |

*Phase 1 best:* Stage 1 RMSE = 0.644, Stage 2 RMSE = **0.474** (leaky — not comparable to later phases)

---

### Phase 2 — Dimension Reduction Study, Park-Based Split (Exp 016–024)
*Split changed to park-based (all months of a park in one split) — honest evaluation.*
*Stage 2 RMSE jumped 0.474 → 0.936: not a regression, just an honest split.*

| # | Commit | S1 Config | s1_val_rmse | s1_val_r2 | s2_val_rmse | s2_val_r2 | Status | Key change |
|---|--------|-----------|-------------|-----------|-------------|-----------|--------|------------|
| 016 | b54de58 | SelectKBest(k=8)+Ridge(α=100) | 0.654 | 0.108 | 0.936 | −0.033 | keep | Park-split baseline |
| 017 | cd170a2 | PCA(n=5)+Ridge | 0.654 | 0.106 | 0.936 | −0.033 | discard | PCA(5) ≈ baseline |
| 018 | 2ce6948 | PCA(n=8)+Ridge | 0.658 | 0.095 | 0.936 | −0.033 | discard | More components hurt |
| 019 | f861282 | PCA(n=3)+Ridge | 0.655 | 0.105 | 0.936 | −0.033 | discard | Best CV of PCA variants; val still worse |
| 020 | 0996b94 | PCA(n=5) S1+S2 | 0.654 | 0.106 | 1.029 | −0.249 | discard | PCA on Stage 2 catastrophic |
| 021 | d65b62e | KernelPCA(rbf,n=5)+Ridge | 0.693 | −0.002 | 0.936 | −0.033 | discard | Nonlinear kernel fails at n≈122 parks |
| 022 | 236e82d | SelectKBest(k=10)+PCA(n=5)+Ridge | 0.660 | 0.092 | 0.936 | −0.033 | discard | Filter then reduce loses signal |
| 023 | cf4cca2 | PCA(95% var)+Ridge | 0.659 | 0.093 | 0.936 | −0.033 | discard | Auto variance threshold still worse |
| 024 | b70e695 | SelectKBest(k=8)+Ridge(α=100) | 0.654 | 0.108 | 0.936 | −0.033 | keep | Confirm best; dim-reduction study done |

*Phase 2 finding:* SelectKBest beats all PCA variants for Stage 1 with n≈122 training parks.
Feature selection by target correlation outperforms variance-based reduction.

---

### Phase 3 — Data Structure Study (Exp 025–031)
*Model fixed: Ridge(α=100). Changed: data framing. All park-based splits.*

| # | Structure | N_train | N_val | Target | val_rmse | val_r2 | Status | Research question answered |
|---|-----------|---------|-------|--------|----------|--------|--------|---------------------------|
| 025 | A: Taxon-group SDI / park | 1,159 | 303 | SDI per group | 0.714 | 0.768 | keep | Organism group is dominant SDI driver |
| 026 | B: Monthly + geo (lat/lon/region) | 4,340 | 1,212 | SDI_monthly | 0.947 | 0.309 | keep | Geography explains only 31% linearly |
| 027 | C: First-differences (ΔSDI) | 4,219 | 1,184 | ΔSDI | 0.611 | 0.049 | keep | Monthly ΔSDI ≈ white noise; linear R²=0.05 |
| 028 | D: Yearly aggregated | 638 | 165 | Annual mean SDI | 0.629 | 0.615 | keep | Annual scale outperforms monthly (R² 0.31→0.62) |
| 029 | E: Temporal lag AR(1) | 4,219 | 1,184 | SDI_monthly | 0.593 | 0.596 | keep | Lag1 doubles R² (0.31→0.60); biodiversity is persistent |
| 030 | F: Taxon × Month × Park | 21,743 | 6,613 | SDI per group/month | 0.443 | **0.844** | keep | Best linear result; 28k rows from 149 parks |
| 031 | G: Rarefied taxon (N=50) | ~7k | ~1.8k | SDI_rarefied | 0.334 | 0.611 | keep | Rarefaction removes observer-effort confound |

*Phase 3 key finding:* Structure F (taxon × month × park) achieves R²=0.844 with a linear model —
organism group identity is the dominant source of SDI variation, not park or ecological features alone.
Structure G (rarefied) becomes the production architecture.

---

### Phase 4 — Single-Stage Rarefied Taxon SDI (Exp 032–041)
*Architecture: single Ridge model, target = SDI_rarefied (N=50 subsample per cell).*
*8,824 rows, 149 parks, park-based split, 108 train / 27 val parks.*

| # | Config | val_rmse | val_r2 | cv_rmse | Status | Note |
|---|--------|----------|--------|---------|--------|------|
| 032 | Ridge(α=100) — baseline | 0.334 | 0.611 | 0.304 | keep | New architecture baseline |
| 033 | Ridge(α=10) | 0.335 | 0.610 | 0.288 | discard | Less regularization — marginal loss |
| 034 | Ridge(α=500) | 0.376 | 0.509 | 0.363 | discard | Over-shrinks taxon coefficients |
| 035 | Ridge(α=1000) | 0.406 | 0.428 | 0.398 | discard | Monotone degradation above α=100 |
| 036 | PCA(n=15)+Ridge(α=100) | 0.334 | 0.611 | 0.305 | discard | 15 components ≈ no change |
| **037** | **PCA(n=10)+Ridge(α=100)** | **0.326** | **0.629** | 0.309 | **keep — BEST** | Sweet spot: 26→10 orthogonal components |
| 038 | PCA(n=20)+Ridge(α=100) | 0.334 | 0.612 | 0.305 | discard | Too many components; same as baseline |
| 039 | ElasticNet(α=0.01, l1=0.5) | 0.338 | 0.603 | 0.308 | discard | L1+L2 mix marginally worse than Ridge |
| 040 | Lasso(α=0.001) | 0.331 | 0.619 | **0.290** | keep | Sparse solution; 2nd best on val |
| 041 | PolynomialFeatures(deg=2)+Ridge | 0.389 | 0.474 | 0.262 | discard | 630 features overfit 27 val parks |

*Canonical `train.py`:* PCA(n=10)+Ridge(α=100) · commit `7104b47`

---

## Metric Trajectory

```
                RMSE over 34 logged experiments
 1.1 ┤
 1.0 ┤  ●                                    ← Exp 016: park split introduced (Stage 2 0.936)
 0.9 ┤
 0.8 ┤●─●           leaky val RMSEs
 0.7 ┤    ●─────────────────────────────────  ← Stage 1 RMSE (two-stage era: 0.64–0.70)
 0.6 ┤         ●─●─●                         ← Stage 2 RMSE after HistGBM (0.474)
 0.5 ┤
 0.4 ┤                                  ●    ← Exp 041 polynomial overfit
 0.3 ┤────────────────────────────●─────●●   ← Single-stage rarefied (0.326–0.334)
      └──────────────────────────────────────
      Exp 1        Exp 15      Exp 25    Exp 41
      Phase 1──────────Phase 2──Phase 3──Phase 4
```

*See `outputs/summary_metric_trajectory.png` for the full annotated plot.*

**Inflection points:**
- Exp 003: Removed MODIS fill-value features → Stage 1 R² flipped positive (−0.04 → +0.06)
- Exp 009: Added Stage 2 HistGBM → Stage 2 RMSE halved (0.831 → 0.478)
- Exp 016: Switched to park-based split → Stage 2 RMSE jumped 0.474 → 0.936 (honest evaluation)
- Exp 032: New single-stage rarefied model → RMSE dropped to 0.334, comparable to raw Structure G
- Exp 037: PCA(n=10) → new best: 0.334 → **0.326**, R² 0.611 → **0.629**

---

## Keep / Discard / Crash Summary

| Phase | Experiments | Keep | Discard | Crash | Keep rate |
|-------|-------------|------|---------|-------|-----------|
| 1 — two-stage, leaky | 15 | 7 | 8 | 0 | 47% |
| 2 — dim-reduction, park split | 9 | 2 | 7 | 0 | 22% |
| 3 — data structures | 7 | 7 | 0 | 0 | 100% |
| 4 — single-stage rarefied | 10 | 3 | 7 | 0 | 30% |
| **Total** | **41** | **19** | **22** | **0** | **46%** |

*See `outputs/summary_panels.png` (left panel) for the bar chart.*

**Notable:** Zero crashes across 41 experiments. Phase 3 had 100% keep rate because every data
structure was a deliberate controlled exploration of a research question, not a hyper-parameter
search. Phase 2 had the lowest keep rate (22%): all PCA variants on Stage 1 failed to beat
SelectKBest. Phase 4 discards cluster around two patterns: alpha too high (α≥500) and
interaction features that overfit the 27 val parks.

---

## Best Result vs. Very First Run

| Metric | Exp 001 (first run) | Exp 037 (best) | Δ | Direction |
|--------|---------------------|----------------|---|-----------|
| val RMSE | 0.697 | **0.326** | −0.371 | ↓ better |
| val R² | −0.043 | **0.629** | +0.672 | ↑ better |
| CV RMSE | 0.761 | 0.309 | −0.452 | ↓ better |
| Architecture | Two-stage, leaky split, Ridge(α=1), n=152 parks | Single-stage, park-split, PCA(n=10)+Ridge(α=100), n=8,824 rows |

*See `outputs/summary_panels.png` (right panel) for the head-to-head bar chart.*

**What changed between them:**

1. **Target redesigned.** Exp 001 predicted park-level mean SDI (152 rows). Exp 037 predicts
   rarefied SDI per (park × month × taxon_group) — 8,824 rows. Same parks, 58× more data by
   restructuring along organism group and time.

2. **Observer-effort confound removed.** Rarefaction (N=50 subsample per cell) reduced the
   iNaturalist effort correlation from r = 0.88 to r = 0.35 and eliminated the +0.061/yr
   temporal trend that would have inflated apparent prediction accuracy.

3. **Split made honest.** Park-based split prevents cross-park leakage. Exp 001 used a leaky
   row-level split where months from the same park appeared in both train and test.

4. **Numeric features compressed.** PCA(n=10) reduces 26 correlated eco/human features to
   10 orthogonal components, discarding noise dimensions where MODIS satellite bands co-vary.

5. **Taxon group encoded as feature.** One-hot taxon_group is the single strongest predictor —
   organism group identity explains more SDI variance than all ecological features combined.

---

## What Actually Worked — Memo

### 1. The data structure mattered far more than the model

Every model-tuning experiment in Phase 1 and 2 produced marginal gains on Stage 1 RMSE
(0.697 → 0.644, a 7.5% improvement across 15 experiments). Restructuring the data in Phase 3
to per-taxon rows produced R²=0.768 with the same Ridge(α=100) model in one experiment (Structure A).
The final architecture (Structure G, rarefied) achieved R²=0.611 in its first run — already
competitive with the best two-stage result, without any model tuning.

**Practical rule:** When N is small (≤152 parks), ask "can I restructure the problem to get
more rows?" before tuning hyper-parameters. Each taxon group is an independent "experiment"
of how the same park ecology supports diversity; treating them as separate rows multiplied
usable data 58×.

### 2. PCA helps when features are more correlated than informative

In Phase 2, PCA on Stage 1 (n=122 parks, 11 features) consistently hurt relative to SelectKBest.
In Phase 4, PCA(n=10) on the single-stage model (n=6,979 rows, 26 features) provided the only
improvement of the round (0.334 → 0.326). The difference: with 122 training samples, PCA has
too little data to estimate reliable principal directions. With 6,979 samples, the 26 correlated
eco/human features genuinely compress to ~10 orthogonal signal dimensions (FPAR bands are
correlated, traffic metrics overlap), and dropping the bottom 16 components filters real noise.

**Practical rule:** PCA before a linear model is only useful when n_samples ≫ n_features AND
the features are expected to be correlated. With small n, use SelectKBest instead.

### 3. Observer effort is a confound that hides in plain sight

The temporal trend analysis showed a +0.061/yr slope in raw SDI (2005–2025). Even restricting
to 2021+ data, the effort correlation was r = 0.88. Without rarefaction, any model would have
been partially predicting "how many people uploaded observations this month" rather than actual
biodiversity. Rarefaction reduced the correlation to r = 0.35 and made the temporal slope
indistinguishable from zero. This is not just data cleaning — it changed what the model predicts.

**Practical rule:** For citizen science data (iNaturalist, eBird, GBIF), always rarefaction-test
your target variable. If Shannon diversity correlates with log(n_observations) above r=0.5,
the target is partially measuring observer effort, not ecology.

### 4. Interaction features overfit the park generalization bottleneck

Exp 041 (PolynomialFeatures degree=2) had the best CV RMSE of all Phase 4 experiments (0.262)
but the worst val RMSE (0.389). The 630 interaction features learned park-specific patterns in
the 108 training parks that did not transfer to 27 held-out parks. The bottleneck is park
diversity (149 total), not row count (8,824). Ridge(α=100) cannot regularize 630 features toward
a 27-park generalization target.

**Practical rule:** When the generalization unit is parks (not rows), the effective n is the
number of parks in each split — 108 train / 27 val. Interaction features require n_effective ≫
n_features. 630 features on 108 parks is deep in the overfit regime.

### 5. Month-to-month biodiversity CHANGES are not linearly predictable

Structure C (first-differences) achieved R² = 0.05. ΔSDI is near-random at the monthly scale
given the available features. Biodiversity level is predictable (ecology explains where a park
sits on the diversity spectrum); biodiversity change is dominated by stochastic population
dynamics at sub-annual timescales. This has implications for causal claims: the model predicts
ecological state, not ecological response to human disturbance.

### 6. What did not work

| Idea | Why it failed |
|------|--------------|
| Stage 1 HistGBM | Overfits n=122 parks; tree ensembles need more samples |
| KernelPCA (rbf) | Same problem — kernel trick needs n ≫ d; fails at n=122 |
| log1p target transform | Compresses variance; slightly reduces predictable signal |
| HistGBM learning rate 0.05 | Slower convergence with fixed max_iter; marginally worse |
| PCA on Stage 2 | Stage 2 HistGBM relies on semantic feature directions (traffic, visitors); PCA destroys them |
| PolynomialFeatures | 630 interactions overfit 27 val parks; train/val divergence ≈ 0.13 RMSE |

### 7. Where to go next

The current model plateaus at R²=0.629 against a target of 0.70. Two paths have the highest
expected gain:

- **Add SDI_lag1 feature (temporal autocorrelation).** Structure E showed that adding last
  month's SDI nearly doubles R² (0.31 → 0.60) for the monthly model. In the rarefied framework,
  this would require computing lag within each (park × taxon_group) time series. No leakage:
  the lag is the park's own prior month. Expected gain: +0.05–0.10 R².

- **Per-taxon models or taxon × ecology interactions.** The recommended next question from Phase 3
  was: "does the relationship between human impact and SDI vary by organism group?" A separate
  Ridge model per taxon group, or explicit group × traffic interaction terms with enough parks
  per group to avoid overfitting, would test whether Insecta and Plantae respond differently
  to the same traffic signal.

---

*Document generated 2026-05-11. Figures: `outputs/summary_metric_trajectory.png`, `outputs/summary_panels.png`.*
*All experiments are reproducible from the commits listed in `results.tsv` on branch `autoresearch/apr27`.*
