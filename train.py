import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Aggregate features per park ---
# Traffic: mean annual traffic count per park
traffic_features = (traffic
                    .groupby('UnitCode')['TrafficCount']
                    .agg(mean_traffic='mean', max_traffic='max', total_traffic='sum')
                    .reset_index())

# Visitation: mean annual visitors + camping per park
# strip commas from numeric columns first
num_cols = ['RecreationVisitors', 'NonRecreationVisitors', 'RecreationHours',
            'TentCampers', 'RVCampers', 'Backcountry']

for col in num_cols:
    parkvisits[col] = pd.to_numeric(
        parkvisits[col].astype(str).str.replace(',', ''), errors='coerce')

visit_features = (parkvisits
                  .groupby('ParkName')
                  .agg(
                      mean_recreation_visitors = ('RecreationVisitors', 'mean'),
                      mean_recreation_hours    = ('RecreationHours', 'mean'),
                      mean_tent_campers        = ('TentCampers', 'mean'),
                      mean_rv_campers          = ('RVCampers', 'mean'),
                      mean_backcountry         = ('Backcountry', 'mean'),
                  )
                  .reset_index())

# --- 2. Join everything ---
# assumes df already has columns: UnitCode, ParkName, SDI
df = (df
      .merge(traffic_features, on='UnitCode', how='left')
      .merge(visit_features,   on='ParkName', how='left'))

# --- 3. Define features and target ---
feature_cols = [
    'mean_traffic',
    'max_traffic',
    'total_traffic',
    'mean_recreation_visitors',
    'mean_recreation_hours',
    'mean_tent_campers',
    'mean_rv_campers',
    'mean_backcountry',
]

df_model = df[feature_cols + ['SDI']].dropna()
X = df_model[feature_cols]
y = df_model['SDI']

print(f"Modeling on {len(df_model)} parks")
print(f"Features: {feature_cols}")

# --- 4. Baseline linear model ---
model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr',     LinearRegression())
])

# Cross-validated R² and RMSE
r2_scores   = cross_val_score(model, X, y, cv=5, scoring='r2')
rmse_scores = cross_val_score(model, X, y, cv=5,
                              scoring='neg_root_mean_squared_error')

print(f"\n--- Baseline LinearRegression ---")
print(f"val_r2:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
print(f"val_rmse: {(-rmse_scores).mean():.4f} ± {(-rmse_scores).std():.4f}")

# Fit on full data to inspect coefficients
model.fit(X, y)
coef_df = (pd.DataFrame({'feature': feature_cols,
                          'coefficient': model.named_steps['lr'].coef_})
             .sort_values('coefficient', key=abs, ascending=False))
print(f"\nCoefficients:\n{coef_df.to_string(index=False)}")
