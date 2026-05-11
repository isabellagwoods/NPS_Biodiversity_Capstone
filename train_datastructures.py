"""
train_datastructures.py — Data Structure Study for NPS Biodiversity
Northwestern STAT 390 Capstone — 2026-05-11

Controlled study: test different data framings with a SINGLE LINEAR MODEL (Ridge).
Each EXPERIMENT tests one way of structuring the data, with a clear research question.
Change only the EXPERIMENT variable (and notes) in Section 2 between runs.

All experiments share:
  - Ridge regression (alpha=100), no tree models
  - Park-based train/test split — all data from a park stays in one split
  - Data filtered to 2021+
  - Reported metrics: val_rmse, val_r2, cv_rmse, n_train, n_val

EXPERIMENT options:
  'A_taxon_sdi'     N~900  Q: Does SDI vary by organism group after accounting for ecology?
  'B_monthly_geo'   N~5552 Q: Do location/latitude predict monthly SDI beyond ecology?
  'C_firstdiff'     N~5400 Q: Does change in traffic predict change in biodiversity?
  'D_yearly'        N~803  Q: Does annual human pressure predict annual biodiversity?
  'E_lag'           N~5403 Q: Does last month's SDI predict this month's?
  'F_taxon_monthly' N~15k+ Q: Across organism groups×months, what drives diversity?

Results logged to: results_datastructures.tsv
Experiment notes:  experiment_log_datastructures.md
"""

import time
import warnings
import glob
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
MAX_SECONDS = 600
start_wall = time.time()
Path('outputs').mkdir(exist_ok=True)
Path('outputs/cache').mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAW DATA LOADING  (do not modify)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading data...")

park = pd.read_csv('full_data_clean.csv')
for col in park.select_dtypes(include='number').columns:
    park.loc[park[col] < -100, col] = np.nan

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

print(f"  Park: {park.shape}   Monthly (2021+): {monthly.shape}")

# ── Canonical feature pools ───────────────────────────────────────────────────
_ECO = [c for c in ['ET_range','GPP_range','avg_FPAR','max_FPAR','FPAR_range',
                     'avg_SNOW','max_SNOW','SNOW_range','n_burn_observations','pct_burned']
        if c in park.columns]
_HUM = [c for c in ['avg_recreation_visitors','max_recreation_visitors','visit_slope',
                     'pop_density','avg_annual_traffic','traffic_cv','n_facilities',
                     'hours_per_visitor'] if c in park.columns]
_GEO = [c for c in ['latitude','longitude'] if c in park.columns]
PARK_FEATS = _ECO + _HUM + _GEO

# US region lookup
_REGION = {
    **{s:'Northeast' for s in ['Maine','New Hampshire','Vermont','Massachusetts',
                                'Rhode Island','Connecticut','New York','New Jersey','Pennsylvania']},
    **{s:'Southeast' for s in ['Delaware','Maryland','Virginia','West Virginia','North Carolina',
                                'South Carolina','Georgia','Florida','Alabama','Mississippi',
                                'Tennessee','Kentucky','Arkansas','Louisiana']},
    **{s:'Midwest'   for s in ['Ohio','Indiana','Illinois','Michigan','Wisconsin','Minnesota',
                                'Iowa','Missouri','North Dakota','South Dakota','Nebraska','Kansas']},
    **{s:'SouthCentral' for s in ['Oklahoma','Texas']},
    **{s:'West'      for s in ['Montana','Idaho','Wyoming','Colorado','New Mexico','Arizona',
                                'Utah','Nevada','California','Oregon','Washington']},
    'Alaska':'Alaska', 'Hawaii':'Hawaii',
}
if 'state' in park.columns:
    park['region'] = park['state'].map(_REGION).fillna('Other')
else:
    park['region'] = 'Other'

INAT_GLOB = '../iNaturalist seperated/iNat_*.csv'

RAREFACTION_N = 50   # subsample every cell to this many obs before computing SDI
# N=20 ceiling = log(20)≈3.0, below 75th pctile of raw SDI (3.33) — too compressed
# N=50 ceiling = log(50)≈3.9, above 75th pctile, keeps 58% of cells (8.8k rows)

def compute_sdi(species_series):
    """Shannon Diversity Index; returns NaN if fewer than 2 species or 5 obs."""
    counts = species_series.value_counts()
    n = counts.sum()
    if n < 5 or len(counts) < 2:
        return np.nan
    p = counts / n
    return float(-(p * np.log(p)).sum())

def compute_sdi_rarefied(species_series, n=RAREFACTION_N):
    """
    Rarefied SDI: subsample to exactly n observations before computing Shannon entropy.
    Makes SDI values comparable across cells regardless of observer effort.
    Returns NaN if the cell has fewer than n observations.
    """
    vals = species_series.values
    if len(vals) < n:
        return np.nan
    sampled = np.random.choice(vals, size=n, replace=False)
    counts = pd.Series(sampled).value_counts()
    if len(counts) < 2:
        return np.nan
    p = counts / n
    return float(-(p * np.log(p)).sum())

def load_inat_2021():
    """Load all iNaturalist CSVs (2021+) with minimal columns."""
    cols = ['observed_on', 'TaxonGroup', 'park_name', 'ScientificName']
    chunks = [pd.read_csv(f, low_memory=False, usecols=cols)
              for f in sorted(glob.glob(INAT_GLOB))]
    inat = pd.concat(chunks, ignore_index=True)
    inat['observed_on'] = pd.to_datetime(inat['observed_on'], errors='coerce')
    inat['year']  = inat['observed_on'].dt.year
    inat['month'] = inat['observed_on'].dt.month
    return inat[inat['year'] >= 2021].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPERIMENT CONFIGURATION  (AGENT: EDIT HERE)
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENT = 'G_rarefied_taxon'

EXPERIMENT_NOTES = """
Research question: After removing the observer-effort confound via rarefaction, do
  ecological conditions and human activity predict taxon-group monthly SDI?
What changed: Structure G — same (park × month × taxon_group) rows as F, but SDI is
  computed from a fixed-size subsample (RAREFACTION_N=20 obs) of each cell instead of
  all available observations. This makes every cell directly comparable: a cell with 20
  obs and a cell with 500 obs yield SDI values on the same scale.
  Single-stage model (no Stage 1 / Stage 2 split).
  log(n_obs) included as predictor to absorb any residual effort signal.
  Park-based split — all months × taxon groups from a park stay together.
  Model: Ridge(alpha=100).
"""

RIDGE_ALPHA = 100   # keep constant across all experiments


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD DATASET  (one branch per experiment)
# ═══════════════════════════════════════════════════════════════════════════════

park_lookup = park[['park_name'] + PARK_FEATS + ['region']].drop_duplicates('park_name')

if EXPERIMENT == 'A_taxon_sdi':
    # ── Structure A: SDI per taxon group × park ───────────────────────────────
    # N ≈ 800-1000   target = SDI within organism group
    RESEARCH_Q = ("Does SDI vary by organism group after accounting "
                  "for a park's ecological features?")
    TARGET = 'SDI_group'
    cache = Path('outputs/cache/taxon_sdi_per_park.csv')
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"  Loaded cached A: {df.shape}")
    else:
        print("  Building Structure A (loading ~4M iNat rows — ~30s)...")
        inat = load_inat_2021()
        rows = []
        for (pname, grp), g in inat.groupby(['park_name', 'TaxonGroup']):
            sdi = compute_sdi(g['ScientificName'])
            if not np.isnan(sdi):
                rows.append({'park_name': pname, 'taxon_group': grp,
                             TARGET: sdi, 'log_n_obs': np.log1p(len(g)),
                             'n_species': g['ScientificName'].nunique()})
        df = pd.DataFrame(rows).merge(park_lookup, on='park_name', how='inner')
        df.to_csv(cache, index=False)
        print(f"  Built + cached A: {df.shape}")
    df = df.dropna(subset=[TARGET])
    NUM_FEATS = PARK_FEATS + ['log_n_obs']
    CAT_FEATS = ['taxon_group']
    GROUP_COL = 'park_name'

elif EXPERIMENT == 'B_monthly_geo':
    # ── Structure B: Monthly SDI + geographic region + lat/lon ────────────────
    # N ≈ 5552   target = SDI_monthly
    # Added vs current train.py: one-hot US region + explicit lat/lon from park table
    RESEARCH_Q = ("Do biogeographic region and latitude predict monthly SDI "
                  "beyond satellite-derived ecological features?")
    TARGET = 'SDI_monthly'
    df = monthly.merge(park_lookup, on='park_name', how='inner')
    df = df.dropna(subset=[TARGET])
    NUM_FEATS = PARK_FEATS + ['month_sin', 'month_cos', 'year_norm']
    CAT_FEATS = ['region']
    GROUP_COL = 'park_name'
    print(f"  Structure B: {df.shape}")

elif EXPERIMENT == 'C_firstdiff':
    # ── Structure C: First-differences (Δ SDI ~ Δ traffic + seasonality) ──────
    # N ≈ 5400   target = month-over-month change in SDI_monthly
    # Park-split means val parks are entirely unseen — tests whether Δ traffic
    # generalizes across parks, not just within a park's own time series.
    RESEARCH_Q = ("Does month-to-month change in traffic predict "
                  "month-to-month change in biodiversity across parks?")
    TARGET = 'delta_SDI'
    df = monthly.merge(park_lookup[['park_name'] + _HUM], on='park_name', how='inner')
    df = df.sort_values(['park_name', 'year', 'month'])
    df['delta_SDI']     = df.groupby('park_name')['SDI_monthly'].diff()
    df['delta_traffic'] = df.groupby('park_name')['monthly_traffic'].diff()
    df = df.dropna(subset=[TARGET])
    NUM_FEATS = ['delta_traffic', 'month_sin', 'month_cos', 'year_norm'] + _HUM + _GEO
    CAT_FEATS = []
    GROUP_COL = 'park_name'
    print(f"  Structure C: {df.shape}")

elif EXPERIMENT == 'D_yearly':
    # ── Structure D: Yearly aggregated monthly SDI ────────────────────────────
    # N ≈ 803   target = mean annual SDI per park per year
    # Aggregation reduces within-year noise; park×year gives medium N.
    RESEARCH_Q = ("Does average annual human pressure predict "
                  "annual average biodiversity across parks and years?")
    TARGET = 'SDI_yearly'
    yearly = (monthly.groupby(['park_name', 'year'])
              .agg(SDI_yearly=('SDI_monthly','mean'),
                   traffic_yr=('monthly_traffic','mean'),
                   visitors_yr=('annual_visitors','mean'),
                   n_months=('SDI_monthly','count'))
              .reset_index())
    yr_min, yr_max = yearly['year'].min(), yearly['year'].max()
    yearly['year_norm'] = (yearly['year'] - yr_min) / max(yr_max - yr_min, 1)
    df = yearly.merge(park_lookup, on='park_name', how='inner')
    df = df.dropna(subset=[TARGET])
    NUM_FEATS = PARK_FEATS + ['traffic_yr', 'visitors_yr', 'year_norm', 'n_months']
    CAT_FEATS = []
    GROUP_COL = 'park_name'
    print(f"  Structure D: {df.shape}")

elif EXPERIMENT == 'E_lag':
    # ── Structure E: Temporal lag — AR(1) with exogenous features ────────────
    # N ≈ 5403   target = SDI_monthly(t),  lag feature = SDI_monthly(t-1)
    # Lag computed within each park's own time series — no cross-park leakage.
    # Tests whether biodiversity is temporally autocorrelated.
    RESEARCH_Q = ("Is next month's biodiversity predictable from last month's "
                  "level, beyond ecological and human-activity features?")
    TARGET = 'SDI_monthly'
    df = monthly.merge(park_lookup, on='park_name', how='inner')
    df = df.sort_values(['park_name', 'year', 'month'])
    df['SDI_lag1'] = df.groupby('park_name')['SDI_monthly'].shift(1)
    df = df.dropna(subset=[TARGET, 'SDI_lag1'])
    NUM_FEATS = PARK_FEATS + ['SDI_lag1', 'month_sin', 'month_cos', 'year_norm']
    CAT_FEATS = []
    GROUP_COL = 'park_name'
    print(f"  Structure E: {df.shape}")

elif EXPERIMENT == 'F_taxon_monthly':
    # ── Structure F: Monthly SDI per taxon group × park ──────────────────────
    # N >> 5552  target = SDI within organism group per park per month
    # Richest structure — taxon identity + time + ecology + human
    RESEARCH_Q = ("Across organism groups and months, what ecological and "
                  "human conditions drive within-group species diversity?")
    TARGET = 'SDI_group_monthly'
    cache = Path('outputs/cache/taxon_monthly_sdi.csv')
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"  Loaded cached F: {df.shape}")
    else:
        print("  Building Structure F (loading ~4M iNat rows — ~60s)...")
        inat = load_inat_2021()
        rows = []
        for (pname, yr, mo, grp), g in inat.groupby(
                ['park_name', 'year', 'month', 'TaxonGroup']):
            if len(g) < 5:
                continue
            sdi = compute_sdi(g['ScientificName'])
            if not np.isnan(sdi):
                rows.append({'park_name': pname, 'year': int(yr), 'month': int(mo),
                             'taxon_group': grp, TARGET: sdi,
                             'log_n_obs': np.log1p(len(g))})
        df = pd.DataFrame(rows)
        traffic_sub = monthly[['park_name','year','month',
                                'monthly_traffic','annual_visitors',
                                'month_sin','month_cos','year_norm']]
        df = df.merge(park_lookup, on='park_name', how='left')
        df = df.merge(traffic_sub, on=['park_name','year','month'], how='left')
        df.to_csv(cache, index=False)
        print(f"  Built + cached F: {df.shape}")
    df = df.dropna(subset=[TARGET])
    NUM_FEATS = PARK_FEATS + ['month_sin','month_cos','year_norm',
                               'monthly_traffic','annual_visitors','log_n_obs']
    CAT_FEATS = ['taxon_group']
    GROUP_COL = 'park_name'

elif EXPERIMENT == 'G_rarefied_taxon':
    # ── Structure G: Rarefied monthly SDI per taxon group × park ─────────────
    # Each (park × month × taxon) cell is subsampled to RAREFACTION_N obs before
    # computing SDI, so observer effort no longer inflates the target.
    # Single-stage model — no Stage 1 / Stage 2 split.
    RESEARCH_Q = ("After removing the iNaturalist observer-effort confound via "
                  "rarefaction, do ecological conditions and human activity "
                  "predict taxon-group monthly SDI?")
    TARGET = 'SDI_rarefied'
    cache = Path(f'outputs/cache/taxon_monthly_sdi_rarefied_n{RAREFACTION_N}.csv')

    if cache.exists():
        df = pd.read_csv(cache)
        print(f"  Loaded cached G (rarefied, N={RAREFACTION_N}): {df.shape}")
    else:
        print(f"  Building Structure G — rarefied SDI (N={RAREFACTION_N} obs/cell)...")
        print("  Loading ~4M iNat rows...")
        np.random.seed(42)
        inat = load_inat_2021()
        rows = []
        for (pname, yr, mo, grp), g in inat.groupby(
                ['park_name', 'year', 'month', 'TaxonGroup']):
            if len(g) < RAREFACTION_N:
                continue
            sdi = compute_sdi_rarefied(g['ScientificName'], n=RAREFACTION_N)
            if not np.isnan(sdi):
                rows.append({
                    'park_name':   pname,
                    'year':        int(yr),
                    'month':       int(mo),
                    'taxon_group': grp,
                    TARGET:        sdi,
                    'n_obs':       len(g),          # original count (not rarefied)
                    'log_n_obs':   np.log1p(len(g)),
                })
        df = pd.DataFrame(rows)
        traffic_sub = monthly[['park_name', 'year', 'month',
                                'monthly_traffic', 'annual_visitors',
                                'month_sin', 'month_cos', 'year_norm']]
        df = df.merge(park_lookup, on='park_name', how='left')
        df = df.merge(traffic_sub, on=['park_name', 'year', 'month'], how='left')
        df.to_csv(cache, index=False)
        print(f"  Built + cached G: {df.shape}")

    df = df.dropna(subset=[TARGET])
    # log_n_obs kept to absorb any residual effort variation after rarefaction
    NUM_FEATS = PARK_FEATS + ['month_sin', 'month_cos', 'year_norm',
                               'monthly_traffic', 'annual_visitors', 'log_n_obs']
    CAT_FEATS = ['taxon_group']
    GROUP_COL = 'park_name'

else:
    raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT!r}. "
                     "Choose from: A_taxon_sdi, B_monthly_geo, C_firstdiff, "
                     "D_yearly, E_lag, F_taxon_monthly, G_rarefied_taxon")

# resolve available columns
avail_num = [f for f in NUM_FEATS if f in df.columns]
avail_cat = [f for f in CAT_FEATS  if f in df.columns]

print(f"\n{'='*60}")
print(f"Experiment : {EXPERIMENT}")
print(f"Question   : {RESEARCH_Q}")
print(f"Dataset    : {len(df):,} rows  target={TARGET}")
print(f"Features   : {len(avail_num)} numeric + {len(avail_cat)} categorical")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PARK-BASED SPLIT + LINEAR FIT
# ═══════════════════════════════════════════════════════════════════════════════

unique_parks = np.array(df[GROUP_COL].unique())
np.random.seed(42)
shuffled  = unique_parks[np.random.permutation(len(unique_parks))]
n_val     = max(1, int(len(shuffled) * 0.2))
val_parks  = set(shuffled[:n_val].tolist())
train_parks = set(shuffled[n_val:].tolist())

train_df = df[df[GROUP_COL].isin(train_parks)].copy()
val_df   = df[df[GROUP_COL].isin(val_parks)].copy()
y_train  = train_df[TARGET].values
y_val    = val_df[TARGET].values

print(f"\nPark-based split: {len(y_train):,} train / {len(y_val):,} val rows")
print(f"  ({len(train_parks)} train parks / {len(val_parks)} val parks)")

# Build preprocessing + Ridge pipeline
transformers = []
if avail_num:
    transformers.append(('num', Pipeline([
        ('imp',    SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), avail_num))
if avail_cat:
    transformers.append(('cat', Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), avail_cat))

model = Pipeline([
    ('prep',  ColumnTransformer(transformers)),
    ('ridge', Ridge(alpha=RIDGE_ALPHA)),
])

X_train = train_df[avail_num + avail_cat]
X_val   = val_df[avail_num + avail_cat]

fit_start = time.time()
model.fit(X_train, y_train)
fit_sec = time.time() - fit_start

y_pred = model.predict(X_val)

X_all = df[avail_num + avail_cat]
y_all = df[TARGET].values
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(model, X_all, y_all, cv=cv,
                            scoring='neg_root_mean_squared_error').mean()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATE + OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
val_r2   = r2_score(y_val, y_pred)
total_sec = time.time() - start_wall

print("\n" + "=" * 60)
print("RESULTS")
print(f"experiment:    {EXPERIMENT}")
print(f"n_train:       {len(y_train)}")
print(f"n_val:         {len(y_val)}")
print(f"val_rmse:      {val_rmse:.6f}")
print(f"val_r2:        {val_r2:.6f}")
print(f"cv_rmse:       {cv_rmse:.6f}")
print(f"fit_seconds:   {fit_sec:.1f}")
print(f"total_seconds: {total_sec:.1f}")
print(f"model:         Ridge(alpha={RIDGE_ALPHA})")
print(f"target:        {TARGET}")
print(f"n_features:    {len(avail_num)+len(avail_cat)}")
print("=" * 60)

try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit = 'no-git'
print(f"\ncommit: {commit}")
print("Agent: append result to results_datastructures.tsv")

# Residual diagnostics plot
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{EXPERIMENT} — Ridge  (RMSE={val_rmse:.4f}, R²={val_r2:.3f})',
                 fontsize=12)
    axes[0].scatter(y_pred, y_val - y_pred, alpha=0.3, s=15, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals vs Predicted'); axes[0].grid(alpha=0.3)
    mn, mx = float(y_val.min()), float(y_val.max())
    axes[1].scatter(y_val, y_pred, alpha=0.3, s=15, color='steelblue')
    axes[1].plot([mn, mx], [mn, mx], 'r--', label='Perfect')
    axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')
    axes[1].set_title('Actual vs Predicted'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out_plot = f'outputs/diagnostics_{EXPERIMENT}.png'
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_plot}")
except Exception as e:
    print(f"Plot error: {e}")

# Append to results TSV
log_path = Path('results_datastructures.tsv')
if not log_path.exists():
    log_path.write_text(
        'commit\texperiment\ttarget\tn_train\tn_val\t'
        'val_rmse\tval_r2\tcv_rmse\ttotal_seconds\tstatus\tnotes\n')

print("\nDone.")
