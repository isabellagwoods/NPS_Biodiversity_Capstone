"""
train.py — Single-Stage NPS Biodiversity Prediction (Rarefied Taxon SDI)
Northwestern STAT 390 Capstone

MODEL: Predict rarefied Shannon Diversity Index per (park × month × taxon_group).

Rarefaction removes the iNaturalist observer-effort confound:
  each (park × month × taxon) cell is subsampled to RAREFACTION_N=50 observations
  before computing SDI, making every cell directly comparable.
  Effect: effort correlation 0.88 → 0.35, temporal slope ≈ 0 within 2021-2025.

Data: 2021-2025 iNaturalist (cells with ≥50 obs), ~8,800 rows, 149 parks
Split: park-based — all months × taxa from a park stay in one split
Target: SDI_rarefied  (Shannon entropy on 50-obs subsample)
Features: ecological + human + geographic (park-level) + time + taxon_group (one-hot)
Goal: val_r2 ≥ 0.70

AutoResearch rules:
  - Edit ONLY Section 2 (model definition + feature lists)
  - DO NOT touch Sections 0, 1, 4, 5
  - DO NOT change how val_rmse or val_r2 are calculated (Section 4)
  - Change ONE thing per experiment: model type, alpha, OR pipeline step
  - Runtime limit: 600s total
"""

import time
import warnings
import subprocess
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ── 0. Runtime budget ─────────────────────────────────────────────────────────
MAX_SECONDS   = 600
RAREFACTION_N = 50    # fixed — changing this invalidates the cache
start_wall    = time.time()
Path('outputs').mkdir(exist_ok=True)
Path('outputs/cache').mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING   (DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading data...")

# ── Park-level features ───────────────────────────────────────────────────────
park = pd.read_csv('full_data_clean.csv')
for col in park.select_dtypes(include='number').columns:
    park.loc[park[col] < -100, col] = np.nan

# ── Rarefied taxon SDI dataset (built from raw iNat on first run, cached) ─────
CACHE = Path(f'outputs/cache/taxon_monthly_sdi_rarefied_n{RAREFACTION_N}.csv')

if CACHE.exists():
    df = pd.read_csv(CACHE)
    print(f"  Rarefied dataset (N={RAREFACTION_N}): {df.shape}")
else:
    print(f"  Cache missing — building from raw iNaturalist files (~60s)...")

    monthly = pd.read_csv('monthly_sdi.csv')
    monthly['year']  = pd.to_numeric(monthly['year'],  errors='coerce').astype('Int64')
    monthly['month'] = pd.to_numeric(monthly['month'], errors='coerce').astype('Int64')
    monthly = monthly.dropna(subset=['SDI_monthly', 'year', 'month'])
    monthly['year']  = monthly['year'].astype(int)
    monthly['month'] = monthly['month'].astype(int)
    monthly = monthly[monthly['year'] >= 2021].copy()
    for tf, fn in [('month_sin', lambda m: np.sin(2*np.pi*m/12)),
                   ('month_cos', lambda m: np.cos(2*np.pi*m/12))]:
        if tf not in monthly.columns:
            monthly[tf] = fn(monthly['month'])
    if 'year_norm' not in monthly.columns:
        yr_min, yr_max = monthly['year'].min(), monthly['year'].max()
        monthly['year_norm'] = (monthly['year'] - yr_min) / max(yr_max - yr_min, 1)

    def _compute_sdi_rarefied(species_vals, n):
        if len(species_vals) < n:
            return np.nan
        sampled = np.random.choice(species_vals, size=n, replace=False)
        counts  = pd.Series(sampled).value_counts()
        if len(counts) < 2:
            return np.nan
        p = counts / n
        return float(-(p * np.log(p)).sum())

    inat_cols = ['observed_on', 'TaxonGroup', 'park_name', 'ScientificName']
    inat_chunks = [pd.read_csv(f, low_memory=False, usecols=inat_cols)
                   for f in sorted(glob.glob('../iNaturalist seperated/iNat_*.csv'))]
    inat = pd.concat(inat_chunks, ignore_index=True)
    inat['observed_on'] = pd.to_datetime(inat['observed_on'], errors='coerce')
    inat['year']  = inat['observed_on'].dt.year
    inat['month'] = inat['observed_on'].dt.month
    inat = inat[inat['year'] >= 2021].dropna(subset=['year', 'month']).copy()

    np.random.seed(42)
    rows = []
    for (pname, yr, mo, grp), g in inat.groupby(
            ['park_name', 'year', 'month', 'TaxonGroup']):
        if len(g) < RAREFACTION_N:
            continue
        sdi = _compute_sdi_rarefied(g['ScientificName'].values, RAREFACTION_N)
        if not np.isnan(sdi):
            rows.append({'park_name': pname, 'year': int(yr), 'month': int(mo),
                         'taxon_group': grp, 'SDI_rarefied': sdi,
                         'n_obs': len(g), 'log_n_obs': np.log1p(len(g))})

    built = pd.DataFrame(rows)

    _eco = [c for c in ['ET_range','GPP_range','avg_FPAR','max_FPAR','FPAR_range',
                         'avg_SNOW','max_SNOW','SNOW_range',
                         'n_burn_observations','pct_burned'] if c in park.columns]
    _hum = [c for c in ['avg_recreation_visitors','max_recreation_visitors',
                         'visit_slope','pop_density','avg_annual_traffic',
                         'traffic_cv','n_facilities','hours_per_visitor']
            if c in park.columns]
    _geo = [c for c in ['latitude','longitude'] if c in park.columns]
    park_sub    = park[['park_name'] + _eco + _hum + _geo].drop_duplicates('park_name')
    monthly_sub = monthly[['park_name','year','month',
                            'monthly_traffic','annual_visitors',
                            'month_sin','month_cos','year_norm']]
    built = built.merge(park_sub,    on='park_name',              how='left')
    built = built.merge(monthly_sub, on=['park_name','year','month'], how='left')
    built.to_csv(CACHE, index=False)
    df = built
    print(f"  Built + cached rarefied dataset: {df.shape}")

# ── Target and feature pools ──────────────────────────────────────────────────
TARGET = 'SDI_rarefied'
df = df.dropna(subset=[TARGET])

ECO_FEATS  = [c for c in ['ET_range','GPP_range','avg_FPAR','max_FPAR','FPAR_range',
                            'avg_SNOW','max_SNOW','SNOW_range',
                            'n_burn_observations','pct_burned'] if c in df.columns]
HUM_FEATS  = [c for c in ['avg_recreation_visitors','max_recreation_visitors',
                            'visit_slope','pop_density','avg_annual_traffic',
                            'traffic_cv','n_facilities','hours_per_visitor']
              if c in df.columns]
GEO_FEATS  = [c for c in ['latitude','longitude'] if c in df.columns]
TIME_FEATS = [c for c in ['month_sin','month_cos','year_norm'] if c in df.columns]
TRAF_FEATS = [c for c in ['monthly_traffic','annual_visitors'] if c in df.columns]
ALL_NUM    = ECO_FEATS + HUM_FEATS + GEO_FEATS + TIME_FEATS + TRAF_FEATS + ['log_n_obs']
CAT_FEATS  = ['taxon_group']

# ── Park-based split (fixed seed — identical every run) ───────────────────────
_parks    = np.array(df['park_name'].unique())
np.random.seed(42)
_shuffled = _parks[np.random.permutation(len(_parks))]
_n_val    = max(1, int(len(_shuffled) * 0.2))
VAL_PARKS   = set(_shuffled[:_n_val].tolist())
TRAIN_PARKS = set(_shuffled[_n_val:].tolist())

train_df = df[df['park_name'].isin(TRAIN_PARKS)].copy()
val_df   = df[df['park_name'].isin(VAL_PARKS)].copy()
y_train  = train_df[TARGET].values
y_val    = val_df[TARGET].values

print(f"  {len(df):,} rows  |  {len(y_train):,} train / {len(y_val):,} val")
print(f"  {len(TRAIN_PARKS)} train parks / {len(VAL_PARKS)} val parks")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL DEFINITION   (AGENT: EDIT HERE)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Available estimators (already imported):
#   Ridge, Lasso, ElasticNet
#   PCA, KernelPCA, PolynomialFeatures
#
# Feature pools (use any subset):
#   ECO_FEATS  — ecological satellite features (10 cols)
#   HUM_FEATS  — human-impact park-level features (8 cols)
#   GEO_FEATS  — latitude, longitude
#   TIME_FEATS — month_sin, month_cos, year_norm
#   TRAF_FEATS — monthly_traffic, annual_visitors
#   ALL_NUM    — all numeric (27 cols total)
#   CAT_FEATS  — ['taxon_group']  (one-hot encoded, always include)
#
# Pipeline shape:
#   ColumnTransformer splits numeric / categorical preprocessing,
#   then a model step at the end.
#   To add PCA: insert a ('pca', PCA(n_components=N)) step inside the numeric
#   sub-pipeline (after 'sc') — keeps taxon_group one-hot separate.
#
# Controlled experiment sequence:
#   Exp 032: Ridge(alpha=100)            — baseline (already run)
#   Round 1: alpha search  (033–035)
#   Round 2: PCA n_components (036–038)
#   Round 3: model type / interaction features (039–041)
#   Round 4: temporal lag + tree models (042–046)

# ── Exp 070: Add taxon×year_norm interactions — taxon-specific temporal trends ──
# Base: exp069 (val=0.2435). Change: add tg_{taxon}_year for each taxon group.
# Some taxa may be increasing/decreasing over 2021-2025 (recovery from pandemic,
# land management changes, climate shift). Single year_norm coefficient applies
# one trend to all taxa; per-taxon slopes allow taxon-specific trajectories.

df = df.sort_values(['park_name', 'taxon_group', 'year', 'month']).copy()
_grp = df.groupby(['park_name', 'taxon_group'])['SDI_rarefied']
df['SDI_lag1']  = _grp.shift(1)
df['SDI_lag2']  = _grp.shift(2)
df['SDI_lag3']  = _grp.shift(3)
df['SDI_lag6']  = _grp.shift(6)
df['SDI_lag9']  = _grp.shift(9)
df['SDI_lag12'] = _grp.shift(12)
df['SDI_cumean'] = _grp.transform(lambda s: s.expanding().mean().shift(1))
df['SDI_month_cumean'] = (df.groupby(['park_name', 'taxon_group', 'month'])['SDI_rarefied']
                          .transform(lambda s: s.expanding().mean().shift(1)))
df['SDI_dev']   = df['SDI_lag1'] - df['SDI_month_cumean']
df['SDI_dev12'] = df['SDI_lag12'] - df['SDI_month_cumean']
df['n_prior_months'] = df.groupby(['park_name', 'taxon_group']).cumcount()
_tg_dummies = pd.get_dummies(df['taxon_group'], prefix='tg', dtype=float)
for _tgc in _tg_dummies.columns:
    df[f'{_tgc}_sin']  = _tg_dummies[_tgc] * df['month_sin']
    df[f'{_tgc}_cos']  = _tg_dummies[_tgc] * df['month_cos']
    df[f'{_tgc}_year'] = _tg_dummies[_tgc] * df['year_norm']  # taxon-specific trend
_tg_interact_feats = [c for c in df.columns if c.startswith('tg_') and
                      (c.endswith('_sin') or c.endswith('_cos') or c.endswith('_year'))]
df = df.dropna(subset=['SDI_lag1', 'SDI_lag2', 'SDI_lag3', 'SDI_lag6', 'SDI_lag9', 'SDI_lag12'])
train_df = df[df['park_name'].isin(TRAIN_PARKS)].copy()
val_df   = df[df['park_name'].isin(VAL_PARKS)].copy()
y_train  = train_df[TARGET].values
y_val    = val_df[TARGET].values
num_features = ALL_NUM + ['SDI_lag1', 'SDI_lag2', 'SDI_lag3', 'SDI_lag6', 'SDI_lag9',
                           'SDI_lag12', 'SDI_cumean', 'SDI_month_cumean', 'SDI_dev',
                           'SDI_dev12', 'n_prior_months'] + _tg_interact_feats

model = Pipeline([
    ('prep', ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
        ]), num_features),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CAT_FEATS),
    ])),
    ('model', ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000)),
])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIT   (do not change evaluation logic in section 4)
# ═══════════════════════════════════════════════════════════════════════════════

X_train = train_df[num_features + CAT_FEATS]
X_val   = val_df[num_features + CAT_FEATS]
X_all   = df[num_features + CAT_FEATS]

fit_start = time.time()
model.fit(X_train, y_train)
fit_sec = time.time() - fit_start
y_pred = model.predict(X_val)
print(f"\nModel fit: {fit_sec:.1f}s  |  {type(model.named_steps['model']).__name__}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION   (DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════════

val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
val_r2   = r2_score(y_val, y_pred)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(
    model, X_all, y_all := df[TARGET].values,
    cv=cv, scoring='neg_root_mean_squared_error').mean()

total_seconds = time.time() - start_wall


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OUTPUTS   (DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 5a. Grep-friendly summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS")
print(f"s1_val_rmse:   {val_rmse:.6f}")
print(f"s1_val_r2:     {val_r2:.6f}")
print(f"s1_cv_rmse:    {cv_rmse:.6f}")
print(f"s2_val_rmse:   nan")
print(f"s2_val_r2:     nan")
print(f"total_seconds: {total_seconds:.1f}")
print(f"s1_model:      {type(model.named_steps['model']).__name__}")
print(f"s2_model:      single-stage")
print(f"n_features:    {len(num_features)} numeric + {len(CAT_FEATS)} categorical")
print("=" * 60)

try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit = 'no-git'
print(f"\ncommit: {commit}")
print("Agent: append result row to results.tsv (see program.md)")

# ── 5b. Initialize results.tsv if missing ────────────────────────────────────
log_path = Path('results.tsv')
if not log_path.exists():
    log_path.write_text(
        'commit\ts1_val_rmse\ts1_val_r2\ts1_cv_rmse\t'
        's2_val_rmse\ts2_val_r2\ts1_model\ts2_model\t'
        'total_seconds\tstatus\tdescription\n')

# ── 5c. Residual diagnostics ──────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    m_name = type(model.named_steps['model']).__name__
    fig.suptitle(f'Diagnostics — {m_name}  (RMSE={val_rmse:.4f}, R²={val_r2:.3f})',
                 fontsize=12)
    axes[0].scatter(y_pred, y_val - y_pred, alpha=0.3, s=12, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted SDI_rarefied')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(alpha=0.3)
    mn, mx = float(y_val.min()), float(y_val.max())
    axes[1].scatter(y_val, y_pred, alpha=0.3, s=12, color='steelblue')
    axes[1].plot([mn, mx], [mn, mx], 'r--', label='Perfect')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/residual_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/residual_diagnostics.png")
except Exception as e:
    print(f"Residual plot error: {e}")

# ── 5d. Metric-over-time plot ─────────────────────────────────────────────────
try:
    if log_path.exists():
        res = pd.read_csv(log_path, sep='\t')
        if len(res) >= 2:
            fig, ax = plt.subplots(figsize=(12, 5))
            valid = res['s1_val_rmse'].dropna()
            ax.plot(valid.index, valid.values, 'o-', color='steelblue')
            ax.set_xlabel('Experiment #')
            ax.set_ylabel('val_rmse')
            ax.set_title('Rarefied SDI Model — RMSE Over Experiments')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/metric_over_time.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: outputs/metric_over_time.png")
except Exception as e:
    print(f"Metric-over-time plot error: {e}")

# ── 5e. Results matrix snapshot ───────────────────────────────────────────────
try:
    if log_path.exists():
        pd.read_csv(log_path, sep='\t').to_csv(
            'outputs/experiment_result_matrix.csv', index=False)
        print("Saved: outputs/experiment_result_matrix.csv")
except Exception as e:
    print(f"Matrix export error: {e}")

print("\nDone.")
