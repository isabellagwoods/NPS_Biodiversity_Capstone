# Failure Analysis Memo
**Project:** NPS Biodiversity SDI Prediction
**Date:** 2026-05-04
**Author:** Isabella Woods

## Dominant Failure Mode

Both stages fail to reach the 0.20 RMSE target primarily because the available feature sets have intrinsically low correlation with the targets — the ecological satellite features explain only ~11% of SDI variance (Stage 1), and the human impact features explain ~78% of monthly SDI variance but from a baseline std of ~1.01, leaving irreducible residual error around 0.47 (Stage 2).

## Evidence

- Best s1_val_rmse: **0.644**  (target 0.20 — 69% reduction still needed)
- Best s2_val_rmse: **0.474**  (target 0.20 — 58% reduction still needed)
- Experiments run: **15**
- Models tried: Ridge (alpha sweep 0.1–1000), Lasso/log-transform (discarded), RandomForest, HistGradientBoostingRegressor, SelectKBest filtering

## Root Cause

**Stage 1:** The canonical ECO feature pool contains six fill-value-contaminated columns (MODIS sentinel values of +32766, +9999, +3276 not caught by the `< -100` filter), which masked signal in early experiments. After removing those, only 11 clean features remain, with max |r| = 0.30 with SDI. With 121 training parks, even perfect models face a theoretical RMSE floor around 0.60–0.65 given the feature informativeness. The small sample size also makes non-linear models (RandomForest, HistGBM) overfit severely.

**Stage 2:** HistGBM is the right model class and achieves R²=0.78 — but SDI_monthly has std ≈ 1.01, meaning R²=0.78 corresponds to RMSE=0.47. Reaching RMSE=0.20 would require R²≈0.96, which demands features that explain park-level fixed effects (latitude, elevation, historical land use, park size) rather than just visitation and traffic signals.

## What I Tried

| Approach | s1_rmse | s2_rmse | Why It Didn't Work |
|----------|---------|---------|-------------------|
| Ridge alpha=1.0 (baseline) | 0.697 | 0.831 | NaN imputation + fill-value contamination masked signal; Stage 2 Ridge too simple |
| Remove fill-contaminated cols | 0.661 | 0.831 | Improved Stage 1 R² from −0.04 to +0.06; Stage 2 still needs better model |
| Ridge alpha sweep (10, 100) | 0.648 | 0.831 | Monotone gain; optimal at alpha=100; Stage 2 unchanged |
| RandomForest(n=100, depth=5) | 0.719 | 0.831 | Overfits on 121 training parks |
| Stage 1 HistGBM(max_iter=200) | 0.834 | 0.831 | Worse — HistGBM also overfits with tiny n |
| Stage 2 HistGBM(max_iter=200) | 0.648 | 0.478 | Major Stage 2 gain (0.83→0.48); HistGBM handles 10k monthly rows well |
| Stage 2 HistGBM(max_iter=500) | 0.648 | 0.475 | Marginal gain from more iterations |
| Log-transform Stage 1 target | 0.656 | 0.475 | Slightly worse; SDI distribution does not benefit from log transform |
| SelectKBest(k=8) + Ridge | 0.644 | 0.475 | Small Stage 1 gain; drops two weakest eco features |
| Trim Stage 2 to top-8 features | 0.644 | 0.474 | Marginal; drops NaN-heavy covid impact and weak pollution cols |

## Next Steps

1. **Stage 1 — add park geography:** Include latitude, longitude, elevation from the park dataset as Stage 1 features; these are strong SDI predictors not in the current ECO pool.
2. **Stage 1 — one-hot encode lc_type:** The string `lc_type` column ('Evergreen Needleleaf Forest', 'Cropland', etc.) encodes habitat type which is highly predictive of biodiversity; encoding it as dummies could improve R² substantially.
3. **Stage 2 — add park-level ecological controls:** Merge the Stage 1 clean eco features (FPAR, GPP_range) into Stage 2 as static park controls; ecology shapes baseline monthly SDI.
4. **Stage 2 — use Stage 1 SDI prediction as a feature:** Pass `SDI_pred` from Stage 1 into Stage 2 to anchor monthly predictions to the park-level baseline.
5. **Data — fix positive fill values upstream:** Update `full_data_clean.csv` preparation to also mask MODIS values > 3000 (ET), > 30000 (GPP), and soil_moisture_range > 100 as NaN before feature engineering.
