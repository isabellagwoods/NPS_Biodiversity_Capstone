"""
train.py — Two-Stage NPS Biodiversity Prediction
Northwestern STAT 390 Capstone

STAGE 1: Predict SDI from ecological features       (park-level, ~170 rows)
STAGE 2: Predict monthly SDI from human + time features (park x year x month)

AutoResearch rules:
  - Edit ONLY Section 2 (model definitions + feature lists)
  - DO NOT touch Sections 0, 1, 4, 5 (data loading + evaluation + outputs)
  - DO NOT change how val_rmse is calculated
  - Runtime limit: 300s per stage / 600s total
  - Target: s1_val_rmse <= 0.20 AND s2_val_rmse <= 0.20
"""

import time
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor,
                               HistGradientBoostingRegressor,
                               GradientBoostingRegressor)
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# ── 0. Runtime budget ─────────────────────────────────────────────────────────
MAX_SECONDS = 600          # 10 min hard cap
start_wall  = time.time()
Path('outputs').mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING   (DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading data...")

# ── Park-level data (one row per park) ───────────────────────────────────────
park = pd.read_csv('full_data_clean.csv')
park = park.copy()

# Replace known MODIS fill values with NaN
for col in park.select_dtypes(include='number').columns:
    park.loc[park[col] < -100, col] = np.nan

print(f"  Park-level:  {park.shape}  (target: SDI)")

# ── Monthly data (one row per park x year x month) ───────────────────────────
monthly = pd.read_csv('monthly_sdi.csv')
monthly['year']  = pd.to_numeric(monthly['year'],  errors='coerce').astype('Int64')
monthly['month'] = pd.to_numeric(monthly['month'], errors='coerce').astype('Int64')
monthly = monthly.dropna(subset=['SDI_monthly', 'year', 'month'])
monthly['year']  = monthly['year'].astype(int)
monthly['month'] = monthly['month'].astype(int)

print(f"  Monthly:     {monthly.shape}  (target: SDI_monthly)")
print(f"  Parks in monthly: {monthly['park_name'].nunique()}")
print(f"  Year range:  {monthly['year'].min()} – {monthly['year'].max()}")

# ── Canonical feature pools (agent may use subsets of these) ─────────────────
_ECO_POOL = [
    # productivity & vegetation
    'avg_ET', 'max_ET', 'ET_range',
    'avg_GPP', 'max_GPP', 'GPP_range',
    'avg_FPAR', 'max_FPAR', 'FPAR_range',
    # snow & soil
    'avg_SNOW', 'max_SNOW', 'SNOW_range',
    'avg_soil_moisture', 'soil_moisture_range',
    # fire
    'n_burn_observations', 'pct_burned',
    # land cover
    'lc_type_code',
] + [c for c in park.columns if c.startswith('lc_') and c != 'lc_type_code']

_HUMAN_POOL = [
    # visitation
    'avg_recreation_visitors', 'max_recreation_visitors',
    'visit_slope', 'hours_per_visitor', 'overnight_ratio',
    'visit_covid_impact', 'backcountry_covid_impact',
    # traffic
    'avg_annual_traffic', 'traffic_slope',
    'traffic_seasonality', 'traffic_cv', 'traffic_covid_impact',
    # pollution & industry
    'weighted_pollution', 'n_facilities', 'total_releases',
    'CO_8hr', 'O3_8hr', 'PM25_wtd', 'SO2_1hr', 'NO2_AM',
    # population
    'pop_density', 'population_2020',
]

ALL_ECO_FEATURES   = [f for f in _ECO_POOL   if f in park.columns]
ALL_HUMAN_FEATURES = [f for f in _HUMAN_POOL if f in park.columns]

# time features always available in monthly
TIME_FEATURES = ['month_sin', 'month_cos', 'year_norm']
for tf in TIME_FEATURES:
    if tf not in monthly.columns:
        if tf == 'month_sin':
            monthly['month_sin'] = np.sin(2 * np.pi * monthly['month'] / 12)
        elif tf == 'month_cos':
            monthly['month_cos'] = np.cos(2 * np.pi * monthly['month'] / 12)
        elif tf == 'year_norm':
            yr_min, yr_max = monthly['year'].min(), monthly['year'].max()
            monthly['year_norm'] = (monthly['year'] - yr_min) / max(yr_max - yr_min, 1)

# ── Stage 1 split ─────────────────────────────────────────────────────────────
s1_df = park.dropna(subset=['SDI']).copy()
X1_all = s1_df[ALL_ECO_FEATURES]
y1_all = s1_df['SDI']
X1_train, X1_val, y1_train, y1_val = train_test_split(
    X1_all, y1_all, test_size=0.2, random_state=42)

print(f"\nStage 1 split: {len(X1_train)} train / {len(X1_val)} val parks")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL DEFINITIONS   (AGENT: EDIT FREELY HERE)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Available estimators (already imported):
#   LinearRegression, Ridge, Lasso, ElasticNet
#   RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
#   TransformedTargetRegressor, PolynomialFeatures, SelectKBest, f_regression
#
# Tips:
#   - HistGradientBoostingRegressor handles NaN natively (no scaler needed)
#   - For log-transform: TransformedTargetRegressor(regressor=..., func=np.log1p,
#                                                   inverse_func=np.expm1)
#   - Change one thing per experiment: model OR alpha OR feature list, not all
#
# Stage 1 search order:
#   Ridge(alpha=1) -> Ridge(alpha=10/100) -> Lasso -> ElasticNet ->
#   RandomForest(n=100) -> HistGBM(max_iter=200) -> log-transform target ->
#   SelectKBest(k=10) + Ridge -> PolynomialFeatures(degree=2) + Ridge
#
# Stage 2 search order:
#   Ridge -> HistGBM -> trim features -> add/remove time features ->
#   add lat/lon -> RandomForest

from sklearn.impute import SimpleImputer

# ── STAGE 1: Ecological baseline (park-level SDI) ────────────────────────────
# exclude: non-numeric + fill-value-contaminated cols (avg_GPP/max_GPP top at 32766,
#   soil_moisture_range tops at 9999, avg_ET/max_ET top at 3276 — MODIS fills > -100)
_FILL_CONTAMINATED = {'avg_GPP', 'max_GPP', 'avg_ET', 'max_ET',
                      'soil_moisture_range', 'avg_soil_moisture'}
stage1_features = [f for f in ALL_ECO_FEATURES
                   if f in park.select_dtypes(include='number').columns
                   and f not in _FILL_CONTAMINATED]

stage1_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_regression, k=8)),
    ('model',  Ridge(alpha=100))
])

# ── STAGE 2: Monthly SDI prediction ──────────────────────────────────────────
# Stage 2 predicts SDI_monthly directly using human + time features
# Monthly traffic is used if available in monthly_sdi.csv

stage2_extra = []                       # add extra cols from monthly if desired
                                        # e.g. ['monthly_traffic', 'annual_visitors']

# trim to top 8 by |correlation| with SDI_monthly; drop NaN-heavy covid and weak pollution cols
stage2_human_features = [f for f in [
    'avg_recreation_visitors', 'max_recreation_visitors', 'visit_slope',
    'pop_density', 'avg_annual_traffic', 'traffic_cv', 'n_facilities',
    'hours_per_visitor'
] if f in ALL_HUMAN_FEATURES]
stage2_time_features  = TIME_FEATURES        # month_sin, month_cos, year_norm

stage2_model = Pipeline([
    ('model', HistGradientBoostingRegressor(max_iter=500))
])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIT   (do not change evaluation logic in section 4)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1 fit ───────────────────────────────────────────────────────────────
s1_start = time.time()
stage1_model.fit(X1_train[stage1_features], y1_train)
s1_fit_sec = time.time() - s1_start

# compute park-level residuals for diagnostics
s1_df['SDI_pred'] = stage1_model.predict(s1_df[stage1_features])
s1_df['residual'] = s1_df['SDI'] - s1_df['SDI_pred']

print(f"Stage 1 fit done in {s1_fit_sec:.1f}s")

# check runtime
if time.time() - start_wall > MAX_SECONDS:
    print("WARNING: approaching runtime limit — skipping Stage 2")
    stage2_available = False
else:
    stage2_available = True

# ── Stage 2 fit ───────────────────────────────────────────────────────────────
s2_fit_sec = 0.0
stage2_final_features = []
X2_val = y2_val = y2_pred = None

if stage2_available:
    # join park-level human features into monthly data
    park_human = s1_df[['park_name'] + stage2_human_features].drop_duplicates('park_name')
    m2 = monthly.merge(park_human, on='park_name', how='inner')

    # build final feature list
    monthly_cols_available = [c for c in stage2_extra if c in m2.columns]
    stage2_final_features = (
        monthly_cols_available +
        [f for f in stage2_human_features if f in m2.columns] +
        [f for f in stage2_time_features  if f in m2.columns]
    )

    # add lat/lon if available
    for geo in ['latitude', 'longitude']:
        if geo in m2.columns and geo not in stage2_final_features:
            stage2_final_features.append(geo)

    # clean rows
    required_cols = ['SDI_monthly'] + stage2_final_features[:3]
    m2_clean = m2.dropna(subset=required_cols).copy()

    print(f"Stage 2: {len(m2_clean):,} clean rows, {len(stage2_final_features)} features")

    if len(m2_clean) >= 30:
        X2 = m2_clean[stage2_final_features]
        y2 = m2_clean['SDI_monthly']

        X2_train, X2_val, y2_train, y2_val = train_test_split(
            X2, y2, test_size=0.2, random_state=42)

        s2_start = time.time()
        stage2_model.fit(X2_train, y2_train)
        s2_fit_sec = time.time() - s2_start
        y2_pred = stage2_model.predict(X2_val)
        print(f"Stage 2 fit done in {s2_fit_sec:.1f}s  "
              f"({len(X2_train)} train / {len(X2_val)} val)")
    else:
        print(f"Stage 2: only {len(m2_clean)} rows after cleaning — skipping")
        stage2_available = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION   (DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════════

# Stage 1
y1_pred     = stage1_model.predict(X1_val[stage1_features])
s1_val_rmse = np.sqrt(mean_squared_error(y1_val, y1_pred))
s1_val_r2   = r2_score(y1_val, y1_pred)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
s1_cv_rmse = -cross_val_score(
    stage1_model, X1_all[stage1_features], y1_all,
    cv=cv, scoring='neg_root_mean_squared_error').mean()

# Stage 2
if stage2_available and y2_pred is not None:
    s2_val_rmse = np.sqrt(mean_squared_error(y2_val, y2_pred))
    s2_val_r2   = r2_score(y2_val, y2_pred)
else:
    s2_val_rmse = float('nan')
    s2_val_r2   = float('nan')

total_seconds = time.time() - start_wall


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OUTPUTS   (DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 5a. Grep-friendly summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS")
print(f"s1_val_rmse:   {s1_val_rmse:.6f}")
print(f"s1_val_r2:     {s1_val_r2:.6f}")
print(f"s1_cv_rmse:    {s1_cv_rmse:.6f}")
print(f"s2_val_rmse:   {s2_val_rmse:.6f}")
print(f"s2_val_r2:     {s2_val_r2:.6f}")
print(f"total_seconds: {total_seconds:.1f}")
print(f"s1_model:      {type(stage1_model.named_steps['model']).__name__}")
print(f"s2_model:      {type(stage2_model.named_steps['model']).__name__ if stage2_available else 'N/A'}")
print(f"s1_n_features: {len(stage1_features)}")
print(f"s2_n_features: {len(stage2_final_features)}")
print("=" * 60)

# ── 5b. Initialize results.tsv if missing ────────────────────────────────────
log_path = Path('results.tsv')
if not log_path.exists():
    log_path.write_text(
        'commit\ts1_val_rmse\ts1_val_r2\ts1_cv_rmse\t'
        's2_val_rmse\ts2_val_r2\ts1_model\ts2_model\t'
        'total_seconds\tstatus\tdescription\n')

try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit = 'no-git'

print(f"\ncommit: {commit}")
print("Agent: append result row to results.tsv (see program.md)")

# ── 5c. Park residuals table ──────────────────────────────────────────────────
try:
    resid_out = s1_df[['park_name', 'SDI', 'SDI_pred', 'residual']].copy()
    resid_out = resid_out.sort_values('residual', key=abs, ascending=False)
    resid_out.to_csv('outputs/park_residuals.csv', index=False)
    print("Saved: outputs/park_residuals.csv")
except Exception as e:
    print(f"Park residuals error: {e}")

# ── 5d. Residual diagnostics plot ────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'Stage 1 Diagnostics — {type(stage1_model.named_steps["model"]).__name__}'
        f'  (RMSE={s1_val_rmse:.4f}, R²={s1_val_r2:.3f})',
        fontsize=12)

    axes[0].scatter(y1_pred, y1_val - y1_pred, alpha=0.6, color='steelblue', s=50)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted SDI')
    axes[0].set_ylabel('Residual (Actual − Predicted)')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(alpha=0.3)

    mn, mx = float(y1_val.min()), float(y1_val.max())
    axes[1].scatter(y1_val, y1_pred, alpha=0.6, color='steelblue', s=50)
    axes[1].plot([mn, mx], [mn, mx], 'r--', label='Perfect prediction')
    axes[1].set_xlabel('Actual SDI')
    axes[1].set_ylabel('Predicted SDI')
    axes[1].set_title('Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/residual_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/residual_diagnostics.png")
except Exception as e:
    print(f"Residual plot error: {e}")

# ── 5e. Metric-over-time plot (reads results.tsv) ────────────────────────────
try:
    if log_path.exists():
        res = pd.read_csv(log_path, sep='\t')
        if len(res) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('AutoResearch — RMSE Over Experiments', fontsize=13)

            for ax, col, label, color in [
                (axes[0], 's1_val_rmse', 'Stage 1 — Ecological', 'steelblue'),
                (axes[1], 's2_val_rmse', 'Stage 2 — Monthly SDI', 'darkorange'),
            ]:
                valid = res[col].dropna()
                if len(valid) > 0:
                    ax.plot(valid.index, valid.values, 'o-', color=color, label=label)
                ax.axhline(0.20, color='red', linestyle=':', linewidth=1.5,
                           label='Target (0.20)')
                ax.set_xlabel('Experiment #')
                ax.set_ylabel('val_rmse')
                ax.set_title(label)
                ax.legend()
                ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig('outputs/metric_over_time.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: outputs/metric_over_time.png")
except Exception as e:
    print(f"Metric-over-time plot error: {e}")

# ── 5f. Experiment-result matrix (CSV snapshot of results.tsv) ───────────────
try:
    if log_path.exists():
        res = pd.read_csv(log_path, sep='\t')
        res.to_csv('outputs/experiment_result_matrix.csv', index=False)
        print("Saved: outputs/experiment_result_matrix.csv")
except Exception as e:
    print(f"Matrix export error: {e}")

print("\nDone.")
