"""
train.py — Single-Stage NPS Biodiversity Prediction (Rarefied Taxon SDI)
Northwestern STAT 390 Capstone — Isabella Woods, 2026

=============================================================================
OVERVIEW
=============================================================================
Predicts the rarefied Shannon Diversity Index (SDI_rarefied) for each
(park × month × taxon_group) cell in US National Parks using iNaturalist
citizen-science observations merged with park-level ecological and human-
impact features from MODIS, traffic surveys, and NPS visitor data.

WHY RAREFACTION?
  iNaturalist observation counts are driven by observer effort, not true
  biodiversity. A park with 10,000 observers will trivially show more species
  than one with 100. Rarefaction fixes this by subsampling every cell to
  RAREFACTION_N=50 observations before computing SDI, making cells directly
  comparable regardless of how many observers visited.

  Effect: effort-SDI correlation drops from 0.88 → 0.35 after rarefaction.
  Temporal SDI slope ≈ 0 within 2021–2025, confirming effort bias is removed.

DATA FLOW:
  raw iNaturalist CSVs  →  rarefied SDI cache  →  merge park features
  →  build AR lags + interaction features  →  ElasticNet  →  evaluate

AUTORESEARCH RULES (DO NOT VIOLATE):
  - Edit ONLY Section 2 (model definition + feature lists)
  - DO NOT touch Sections 0, 1, 4, 5
  - DO NOT change how val_rmse or val_r2 are calculated (Section 4)
  - Change ONE thing per experiment: model type, alpha, OR pipeline step
  - Runtime limit: 600 seconds total
"""

import time
import warnings
import subprocess
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripted runs
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

warnings.filterwarnings('ignore')   # suppress sklearn convergence/deprecation noise

# ── 0. Runtime budget ─────────────────────────────────────────────────────────
# Hard wall: if the model or CV takes more than 600 s total, something is wrong.
# HistGBM with many features can silently run for hours — kill early.
MAX_SECONDS   = 600
RAREFACTION_N = 50    # FIXED — changing this invalidates the cached rarefied dataset
start_wall    = time.time()
Path('outputs').mkdir(exist_ok=True)          # output plots go here
Path('outputs/cache').mkdir(exist_ok=True)    # rarefied SDI cache lives here


# =============================================================================
# SECTION 1 — DATA LOADING   (DO NOT MODIFY)
# =============================================================================
# This section builds the rarefied taxon SDI dataset.  It runs once and caches
# the result to outputs/cache/taxon_monthly_sdi_rarefied_n50.csv so subsequent
# runs take seconds rather than minutes.
# =============================================================================

print("=" * 60)
print("Loading data...")

# ── Park-level features ───────────────────────────────────────────────────────
# full_data_clean.csv contains one row per park with static ecological and
# human-impact features: MODIS satellite products (FPAR, ET, GPP, SNOW),
# NPS visitor stats, traffic, facilities, population density, lat/lon.
# Values < -100 are MODIS fill values (e.g., 32766 for avg_GPP) — treat as NaN.
park = pd.read_csv('full_data_clean.csv')
for col in park.select_dtypes(include='number').columns:
    park.loc[park[col] < -100, col] = np.nan

# ── Rarefied taxon SDI dataset ────────────────────────────────────────────────
# Cache path includes the rarefaction depth N so changing N auto-invalidates it.
CACHE = Path(f'outputs/cache/taxon_monthly_sdi_rarefied_n{RAREFACTION_N}.csv')

if CACHE.exists():
    # Fast path: load the pre-built rarefied dataset (~0.1 s).
    df = pd.read_csv(CACHE)
    print(f"  Rarefied dataset (N={RAREFACTION_N}): {df.shape}")
else:
    # Slow path: build the rarefied dataset from raw iNat files (~60–90 min on CPU).
    # This only runs once; subsequent runs use the cache.
    print(f"  Cache missing — building from raw iNaturalist files (~60s)...")

    # monthly_sdi.csv contains precomputed monthly SDI from the non-rarefied pipeline
    # plus the temporal features (month_sin, month_cos, year_norm, monthly_traffic,
    # annual_visitors) that were merged into each monthly row.
    monthly = pd.read_csv('monthly_sdi.csv')
    monthly['year']  = pd.to_numeric(monthly['year'],  errors='coerce').astype('Int64')
    monthly['month'] = pd.to_numeric(monthly['month'], errors='coerce').astype('Int64')
    monthly = monthly.dropna(subset=['SDI_monthly', 'year', 'month'])
    monthly['year']  = monthly['year'].astype(int)
    monthly['month'] = monthly['month'].astype(int)
    monthly = monthly[monthly['year'] >= 2021].copy()   # restrict to study period

    # Encode calendar month as two circular features so January and December are
    # numerically close (month 1 and month 12 have |Δ| = 30° on the unit circle,
    # not |12 - 1| = 11 in a linear encoding).
    for tf, fn in [('month_sin', lambda m: np.sin(2*np.pi*m/12)),
                   ('month_cos', lambda m: np.cos(2*np.pi*m/12))]:
        if tf not in monthly.columns:
            monthly[tf] = fn(monthly['month'])

    # Normalise year to [0, 1] so temporal trend features have the same scale as
    # other features after StandardScaler.  0 = 2021, 1 = 2025.
    if 'year_norm' not in monthly.columns:
        yr_min, yr_max = monthly['year'].min(), monthly['year'].max()
        monthly['year_norm'] = (monthly['year'] - yr_min) / max(yr_max - yr_min, 1)

    def _compute_sdi_rarefied(species_vals, n):
        """
        Compute rarefied Shannon Diversity Index for one (park, year, month, taxon) cell.

        Randomly subsample n observations from species_vals (without replacement),
        then compute H = -Σ p_i log(p_i) on the subsample.

        Returns NaN if:
          - the cell has fewer than n observations (excluded from dataset)
          - the subsample contains only one species (SDI = 0 is uninformative)
        """
        if len(species_vals) < n:
            return np.nan
        sampled = np.random.choice(species_vals, size=n, replace=False)
        counts  = pd.Series(sampled).value_counts()
        if len(counts) < 2:
            return np.nan   # monoculture cell — drop
        p = counts / n
        return float(-(p * np.log(p)).sum())

    # Load raw iNaturalist CSVs (split across multiple files by taxon group).
    # Only read the four columns we need to keep memory footprint small.
    inat_cols = ['observed_on', 'TaxonGroup', 'park_name', 'ScientificName']
    inat_chunks = [pd.read_csv(f, low_memory=False, usecols=inat_cols)
                   for f in sorted(glob.glob('../iNaturalist seperated/iNat_*.csv'))]
    inat = pd.concat(inat_chunks, ignore_index=True)
    inat['observed_on'] = pd.to_datetime(inat['observed_on'], errors='coerce')
    inat['year']  = inat['observed_on'].dt.year
    inat['month'] = inat['observed_on'].dt.month
    inat = inat[inat['year'] >= 2021].dropna(subset=['year', 'month']).copy()

    # Compute rarefied SDI for each (park, year, month, taxon) group.
    # Groups with < RAREFACTION_N observations are silently skipped (NaN returned).
    np.random.seed(42)   # reproducible rarefaction subsampling
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

    # Select only the park-level feature columns that survived the fill-value filter.
    # (avg_GPP, max_GPP, avg_ET, max_ET, soil_moisture* were dropped at exp003
    # because MODIS fill values contaminated them and broke the Ridge baseline.)
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

    # Left-join park features and monthly traffic/visitor data onto the rarefied SDI rows.
    built = built.merge(park_sub,    on='park_name',              how='left')
    built = built.merge(monthly_sub, on=['park_name','year','month'], how='left')
    built.to_csv(CACHE, index=False)
    df = built
    print(f"  Built + cached rarefied dataset: {df.shape}")

# ── Target and feature pools ──────────────────────────────────────────────────
# TARGET: SDI_rarefied — the rarefied Shannon Diversity Index.
# Drop rows where the target is missing (shouldn't happen if cache is clean,
# but defensive against edge cases in the build path).
TARGET = 'SDI_rarefied'
df = df.dropna(subset=[TARGET])

# Feature pool definitions.  These are the column names available from the cache.
# Section 2 can use any subset; ALL_NUM is the union used by default.
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
CAT_FEATS  = ['taxon_group']   # one-hot encoded in the pipeline

# ── Park-based train/val split ────────────────────────────────────────────────
# CRITICAL DESIGN DECISION: split by park, not by row.
# All months and taxa from a given park go entirely into either train or val.
# This prevents the model from memorising park-specific patterns (e.g., Yosemite
# always has high SDI for Plantae) and forces it to generalise to unseen parks.
# Fixed seed 42 → identical split on every run → comparable experiment results.
_parks    = np.array(df['park_name'].unique())
np.random.seed(42)
_shuffled = _parks[np.random.permutation(len(_parks))]
_n_val    = max(1, int(len(_shuffled) * 0.2))   # 20% of parks → validation
VAL_PARKS   = set(_shuffled[:_n_val].tolist())   # 27 parks held out
TRAIN_PARKS = set(_shuffled[_n_val:].tolist())   # 108 parks for training

train_df = df[df['park_name'].isin(TRAIN_PARKS)].copy()
val_df   = df[df['park_name'].isin(VAL_PARKS)].copy()
y_train  = train_df[TARGET].values
y_val    = val_df[TARGET].values

print(f"  {len(df):,} rows  |  {len(y_train):,} train / {len(y_val):,} val")
print(f"  {len(TRAIN_PARKS)} train parks / {len(VAL_PARKS)} val parks")


# =============================================================================
# SECTION 2 — MODEL DEFINITION   (AGENT: EDIT HERE)
# =============================================================================
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
# Experiment history (key milestones):
#   Exp 032: Ridge(alpha=100)            — baseline RMSE=0.326
#   Exp 042–059: SDI AR lags lag1–lag12  — biggest jump → RMSE=0.253
#   Exp 060–062: ElasticNet tuning       — RMSE=0.252
#   Exp 063–064: SDI_cumean baselines    — RMSE=0.246
#   Exp 066–070: SDI_dev + taxon×season  — RMSE=0.243
#   Exp 072–075: taxon×ECO/HUM + greedy  — RMSE=0.238 (current best)
#   Exp 079: SDI_park_cumean             — RMSE=0.238 (locked final)

# ── Exp 079: SDI_park_cumean — park-level SDI baseline across all taxa ──────────
# Base: exp075 (val=0.23846).  Added SDI_park_cumean = expanding mean of all-taxon
# SDI for this park (independent of taxon_group). Captures overall park health trend
# that is shared across all taxa — a park-level diversity baseline.

# Sort chronologically within each (park, taxon) series so shifts are correct.
df = df.sort_values(['park_name', 'taxon_group', 'year', 'month']).copy()

# ── Autoregressive temporal lags ───────────────────────────────────────────────
# Compute SDI_lag1 … lag12 within each (park, taxon_group) series.
# shift(k) gives the SDI from k steps ago in the SORTED series.  Because the data
# is monthly but cells are sparse (not every park has every month), these are
# "k prior observations", not strictly "k months ago".
# These lags are the single most important feature group (exp042–059).
_grp = df.groupby(['park_name', 'taxon_group'])['SDI_rarefied']
df['SDI_lag1']  = _grp.shift(1)   # SDI from the previous observation
df['SDI_lag2']  = _grp.shift(2)   # two observations ago
df['SDI_lag3']  = _grp.shift(3)
df['SDI_lag6']  = _grp.shift(6)   # ~semi-annual (6 months ago in dense series)
df['SDI_lag9']  = _grp.shift(9)
df['SDI_lag12'] = _grp.shift(12)  # ~annual comparison

# ── Cumulative baseline features ───────────────────────────────────────────────
# SDI_cumean: expanding mean of all prior SDI values for this (park, taxon).
# Captures the long-run typical diversity level — a "home base" to deviate from.
# Shifted by 1 to avoid using the current observation in the baseline (no leakage).
df['SDI_cumean'] = _grp.transform(lambda s: s.expanding().mean().shift(1))

# SDI_month_cumean: same-month expanding mean (e.g., all prior Julys for this park×taxon).
# Captures seasonal norms — what diversity "should be" in this month historically.
df['SDI_month_cumean'] = (df.groupby(['park_name', 'taxon_group', 'month'])['SDI_rarefied']
                          .transform(lambda s: s.expanding().mean().shift(1)))

# ── Deviation features ─────────────────────────────────────────────────────────
# SDI_dev: how much the most recent observation deviated from the seasonal norm.
# Positive = better than usual for this month; negative = worse than usual.
# This captures momentum: a park recovering from disturbance vs. one in decline.
df['SDI_dev']   = df['SDI_lag1'] - df['SDI_month_cumean']

# SDI_dev12: same comparison but against the observation 12 steps ago (interannual).
# Captures year-over-year change relative to seasonal baseline.
df['SDI_dev12'] = df['SDI_lag12'] - df['SDI_month_cumean']

# ── Reliability signal ─────────────────────────────────────────────────────────
# n_prior_months: how many prior observations exist for this (park, taxon) series.
# Early in the series (few priors), cumean/lag features are noisy — the model can
# down-weight them by learning a lower coefficient when n_prior_months is small.
df['n_prior_months'] = df.groupby(['park_name', 'taxon_group']).cumcount()

# ── Taxon-interaction features ─────────────────────────────────────────────────
# Create one-hot taxon indicators, then multiply by continuous features to give
# the model per-taxon slopes for each environmental variable.
# E.g., tg_Aves_fpar captures how FPAR relates to bird diversity specifically,
# independent of the overall FPAR effect shared across all taxa.
#
# Interaction terms added across experiments 069–075 (greedy forward selection):
#   ×sin/cos: per-taxon seasonal shape  (exp069)
#   ×year:    per-taxon temporal trend  (exp070)
#   ×fpar:    per-taxon FPAR response   (exp072) — biggest single interaction gain
#   ×lat:     per-taxon lat gradient    (exp073)
#   ×popd:    per-taxon urbanization    (exp074)
#   ×etrange: per-taxon ET variability  (exp075 greedy)
#   ×gpprange, ×maxsnow, ×maxrec, ×lon, ×lognobs: also from exp075 greedy
_tg_dummies = pd.get_dummies(df['taxon_group'], prefix='tg', dtype=float)
for _tgc in _tg_dummies.columns:
    # Temporal interaction terms — allows each taxon to have its own seasonal curve
    df[f'{_tgc}_sin']      = _tg_dummies[_tgc] * df['month_sin']
    df[f'{_tgc}_cos']      = _tg_dummies[_tgc] * df['month_cos']
    df[f'{_tgc}_year']     = _tg_dummies[_tgc] * df['year_norm']
    # Ecological interaction terms — per-taxon response to habitat productivity
    df[f'{_tgc}_fpar']     = _tg_dummies[_tgc] * df['avg_FPAR']
    # Geographic interaction terms — per-taxon latitudinal diversity gradient
    df[f'{_tgc}_lat']      = _tg_dummies[_tgc] * df['latitude']
    # Human disturbance — some taxa are sensitive (large mammals), others thrive (corvids)
    df[f'{_tgc}_popd']     = _tg_dummies[_tgc] * df['pop_density']
    # Additional interactions found by greedy forward selection (exp075)
    df[f'{_tgc}_etrange']  = _tg_dummies[_tgc] * df['ET_range']        # ET variability
    df[f'{_tgc}_gpprange'] = _tg_dummies[_tgc] * df['GPP_range']       # GPP variability
    df[f'{_tgc}_maxsnow']  = _tg_dummies[_tgc] * df['max_SNOW']        # max snow depth
    df[f'{_tgc}_maxrec']   = _tg_dummies[_tgc] * df['max_recreation_visitors']  # peak visitors
    df[f'{_tgc}_lon']      = _tg_dummies[_tgc] * df['longitude']       # east-west position
    df[f'{_tgc}_lognobs']  = _tg_dummies[_tgc] * df['log_n_obs']       # per-taxon effort correction

# Collect all generated interaction column names matching the kept suffixes.
_KEEP_SUFFIXES = ('_sin','_cos','_year','_fpar','_lat','_popd',
                  '_etrange','_gpprange','_maxsnow','_maxrec','_lon','_lognobs')
_tg_interact_feats = [c for c in df.columns if c.startswith('tg_') and
                      any(c.endswith(s) for s in _KEEP_SUFFIXES)]

# ── AR × seasonal cross-feature ───────────────────────────────────────────────
# lag6_sin: SDI_lag6 × month_sin — captures whether the 6-step-ago observation
# predicts more strongly in certain seasons (found by greedy selection, exp075).
df['lag6_sin'] = df['SDI_lag6'] * df['month_sin']

# ── Park-level SDI baseline ───────────────────────────────────────────────────
# SDI_park_cumean: expanding mean of SDI across ALL taxa for this park, sorted by
# time.  Unlike SDI_cumean (which is per-taxon), this is a park-wide signal —
# "is this park generally doing well or poorly, regardless of which taxon we're
# predicting?"  Added at exp079 for a marginal improvement.
_park_sorted = df.sort_values(['park_name', 'year', 'month'])
df['SDI_park_cumean'] = (_park_sorted.groupby('park_name')[TARGET]
                         .transform(lambda s: s.expanding().mean().shift(1)))

# ── Drop rows with missing lag values ─────────────────────────────────────────
# After shift() operations, the first k rows of each (park, taxon) series have
# NaN lags.  Drop them so the model always has a complete autoregressive context.
# This reduces ~8,824 → ~7,000 rows (series beginnings are discarded).
df = df.dropna(subset=['SDI_lag1', 'SDI_lag2', 'SDI_lag3', 'SDI_lag6', 'SDI_lag9', 'SDI_lag12'])

# Rebuild train/val splits after dropping rows (same park membership, fewer rows).
train_df = df[df['park_name'].isin(TRAIN_PARKS)].copy()
val_df   = df[df['park_name'].isin(VAL_PARKS)].copy()
y_train  = train_df[TARGET].values
y_val    = val_df[TARGET].values

# ── Feature list ───────────────────────────────────────────────────────────────
# Final feature set: 195 numeric + 1 categorical (taxon_group, one-hot encoded).
# Order: base → AR lags → baselines → deviations → reliability → AR×seasonal
#        → park baseline → taxon interaction block
num_features = ALL_NUM + ['SDI_lag1', 'SDI_lag2', 'SDI_lag3', 'SDI_lag6', 'SDI_lag9',
                           'SDI_lag12', 'SDI_cumean', 'SDI_month_cumean', 'SDI_dev',
                           'SDI_dev12', 'n_prior_months', 'lag6_sin',
                           'SDI_park_cumean'] + _tg_interact_feats

# ── Model pipeline ─────────────────────────────────────────────────────────────
# ColumnTransformer applies different preprocessing to numeric vs. categorical:
#   Numeric:     SimpleImputer(median) → StandardScaler
#                  Median imputation handles NaN park features (e.g., SNOW for
#                  tropical parks); StandardScaler needed because ElasticNet is
#                  sensitive to feature scale.
#   Categorical: SimpleImputer(constant) → OneHotEncoder
#                  taxon_group → 13 dummy columns (handled_unknown='ignore' is
#                  safe if a new taxon appears at inference time).
#
# ElasticNet hyperparameters confirmed optimal via experiments 076–078:
#   alpha=0.001  — mild regularization; confirmed better than 0.0005 or 0.003
#   l1_ratio=0.2 — 80% Ridge, 20% Lasso; confirmed better than 0.5 (more sparsity)
#   max_iter=10000 — enough iterations for convergence on 195 features
model = Pipeline([
    ('prep', ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),   # fill NaN park features
            ('sc',  StandardScaler()),                   # unit-variance scaling
        ]), num_features),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CAT_FEATS),
    ])),
    ('model', ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000)),
])


# =============================================================================
# SECTION 3 — FIT   (do not change evaluation logic in section 4)
# =============================================================================
# Build the feature matrices from the processed df, fit the pipeline, and
# generate val predictions.  X_all is used for cross-validation in Section 4.
# =============================================================================

X_train = train_df[num_features + CAT_FEATS]
X_val   = val_df[num_features + CAT_FEATS]
X_all   = df[num_features + CAT_FEATS]   # full dataset for 5-fold CV

fit_start = time.time()
model.fit(X_train, y_train)
fit_sec = time.time() - fit_start
y_pred = model.predict(X_val)
print(f"\nModel fit: {fit_sec:.1f}s  |  {type(model.named_steps['model']).__name__}")


# =============================================================================
# SECTION 4 — EVALUATION   (DO NOT MODIFY)
# =============================================================================
# val_rmse and val_r2 are computed on the held-out park split.
# cv_rmse is 5-fold cross-validation on the full dataset (different split than val).
# DO NOT change this section — it is the fixed evaluator for all experiments.
# =============================================================================

val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
val_r2   = r2_score(y_val, y_pred)

# 5-fold CV across all data (not just train parks) gives an estimate of
# in-distribution generalization; compare with val_rmse for overfitting signal.
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(
    model, X_all, y_all := df[TARGET].values,
    cv=cv, scoring='neg_root_mean_squared_error').mean()

total_seconds = time.time() - start_wall


# =============================================================================
# SECTION 5 — OUTPUTS   (DO NOT MODIFY)
# =============================================================================
# Print grep-friendly result lines, then generate diagnostic plots and
# update the results matrix CSV.  All output files go to outputs/.
# =============================================================================

# ── 5a. Grep-friendly summary ─────────────────────────────────────────────────
# These exact prefixes are what the shell grep in program.md extracts.
print("\n" + "=" * 60)
print("RESULTS")
print(f"s1_val_rmse:   {val_rmse:.6f}")
print(f"s1_val_r2:     {val_r2:.6f}")
print(f"s1_cv_rmse:    {cv_rmse:.6f}")
print(f"s2_val_rmse:   nan")    # kept for legacy compatibility with results.tsv format
print(f"s2_val_r2:     nan")
print(f"total_seconds: {total_seconds:.1f}")
print(f"s1_model:      {type(model.named_steps['model']).__name__}")
print(f"s2_model:      single-stage")
print(f"n_features:    {len(num_features)} numeric + {len(CAT_FEATS)} categorical")
print("=" * 60)

# Print the current git commit so results.tsv rows are traceable.
try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit = 'no-git'
print(f"\ncommit: {commit}")
print("Agent: append result row to results.tsv (see program.md)")

# ── 5b. Initialise results.tsv header if file is missing ─────────────────────
log_path = Path('results.tsv')
if not log_path.exists():
    log_path.write_text(
        'commit\ts1_val_rmse\ts1_val_r2\ts1_cv_rmse\t'
        's2_val_rmse\ts2_val_r2\ts1_model\ts2_model\t'
        'total_seconds\tstatus\tdescription\n')

# ── 5c. Residual diagnostics plot ─────────────────────────────────────────────
# Saved to outputs/residual_diagnostics.png on every run.
# Left panel: residuals vs. predicted (should be randomly scattered around 0).
# Right panel: actual vs. predicted (dots on the diagonal = perfect prediction).
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

# ── 5d. RMSE over experiments plot ────────────────────────────────────────────
# Tracks val_rmse across all rows in results.tsv so you can see the improvement
# trajectory visually.  Only plotted if there are at least 2 logged experiments.
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
# Exports results.tsv as a CSV for easier inspection in spreadsheet tools.
try:
    if log_path.exists():
        pd.read_csv(log_path, sep='\t').to_csv(
            'outputs/experiment_result_matrix.csv', index=False)
        print("Saved: outputs/experiment_result_matrix.csv")
except Exception as e:
    print(f"Matrix export error: {e}")

print("\nDone.")
