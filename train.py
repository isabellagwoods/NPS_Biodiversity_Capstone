"""
train.py — Baseline model for NPS Biodiversity SDI Prediction
AutoResearch loop: edit ONLY this file.
Metric: val_rmse (lower is better). DO NOT change the metric.
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ── 0. Runtime budget ────────────────────────────────────────────────────────
MAX_SECONDS = 300   # 5-minute hard cap — DO NOT exceed
start_wall  = time.time()

# ── 1. Load data ─────────────────────────────────────────────────────────────
data = pd.read_csv('full_data_clean.csv')

X = data.drop(columns=['SDI', 'n_observations', 'n_species'])
y = data['SDI']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Model definition ───────────────────────────────────────────────────────
# AutoResearch: modify anything in this section.
# Allowed: model type, hyperparameters, feature selection, preprocessing.
# NOT allowed: changing val_rmse / val_r2 calculation below.

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('select',  SelectKBest(score_func=f_regression, k=30)),
    ('model',   Ridge(alpha=100)),
])

param_grid = {
    'select__k':    [30],
    'model__alpha': [100],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=0,
)

# ── 3. Fit ────────────────────────────────────────────────────────────────────
fit_start = time.time()
grid_search.fit(X_train, y_train)
fit_seconds = time.time() - fit_start

# ── 4. Evaluate — DO NOT MODIFY ───────────────────────────────────────────────
best_model  = grid_search.best_estimator_
y_pred      = best_model.predict(X_val)
val_rmse    = np.sqrt(mean_squared_error(y_val, y_pred))
val_r2      = r2_score(y_val, y_pred)
total_seconds = time.time() - start_wall

# ── 5. Print summary — DO NOT MODIFY ─────────────────────────────────────────
print("---")
print(f"val_rmse:      {val_rmse:.6f}")
print(f"val_r2:        {val_r2:.6f}")
print(f"best_params:   {grid_search.best_params_}")
print(f"fit_seconds:   {fit_seconds:.1f}")
print(f"total_seconds: {total_seconds:.1f}")
print(f"num_features:  {X_train.shape[1]}")
print(f"num_parks:     {len(X_train) + len(X_val)}")
print(f"model_type:    {type(best_model.named_steps['model']).__name__}")
