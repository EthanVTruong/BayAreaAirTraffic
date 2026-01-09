"""
Bay Area Grid-Based Congestion Forecasting Model
===================================================================
Generates: 
- model_15min.joblib, model_30min.joblib
- metrics_comparison.png
- feature_importance_30min.png
- forecast_timeseries.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import logging
import warnings
from datetime import timedelta
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CORE_BBOX = {"west": -122.6, "south": 37.0, "east": -121.5, "north": 38.0}
GRID_RESOLUTION = 0.05
MAX_EXPANSION_ROWS = 50_000_000 

WINDOW_MINUTES = 5
GAP_THRESHOLD_MINUTES = 5
HORIZONS = {'15min': 3, '30min': 6}

ARRIVAL_ALTITUDE_MIN_FT, ARRIVAL_ALTITUDE_MAX_FT = 1500, 10000
ARRIVAL_VERTRATE_THRESHOLD = -2.0

WEIGHTS = {'density': 0.5, 'arrival_pressure': 0.3, 'vertical_dispersion': 0.2}

RF_PARAMS = {
    'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 5,
    'n_jobs': -1, 'random_state': 42
}

FEATURE_COLS = [
    'lat_bin', 'lon_bin', 'aircraft_count', 'arrival_pressure',
    'altitude_std', 'mean_speed', 'lag_1', 'lag_2', 'hour_sin', 'hour_cos'
]

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(output_dir):
    logger = logging.getLogger('congestion_model')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S'))
        logger.addHandler(ch)
    return logger

log = logging.getLogger('congestion_model')

# =============================================================================
# DATA PROCESSING PHASES
# =============================================================================

def add_spatial_bins(df):
    log.info("Phase 1: Deterministic Spatial Binning...")
    df['lat_bin'] = np.floor((df['lat'] - CORE_BBOX['south']) / GRID_RESOLUTION) * GRID_RESOLUTION + CORE_BBOX['south']
    df['lon_bin'] = np.floor((df['lon'] - CORE_BBOX['west']) / GRID_RESOLUTION) * GRID_RESOLUTION + CORE_BBOX['west']
    return df[(df['lat_bin'] >= CORE_BBOX['south']) & (df['lat_bin'] < CORE_BBOX['north']) &
              (df['lon_bin'] >= CORE_BBOX['west']) & (df['lon_bin'] < CORE_BBOX['east'])]

def detect_islands(df):
    log.info("Phase 2: Temporal Integrity (Per-Aircraft Islands)...")
    df = df.sort_values(['icao24', 'timestamp']).reset_index(drop=True)
    df['time_gap'] = df.groupby('icao24')['timestamp'].diff()
    df['new_island'] = (df['time_gap'] > timedelta(minutes=GAP_THRESHOLD_MINUTES)) | (df['time_gap'].isna())
    df['island_id'] = df['new_island'].cumsum()
    return df.drop(columns=['time_gap', 'new_island'])

def aggregate_to_grid(df):
    log.info("Phase 3: Fast Vectorized Aggregation...")
    df['window'] = df['timestamp'].dt.floor(f'{WINDOW_MINUTES}min')
    is_arrival = df['altitude_ft'].between(ARRIVAL_ALTITUDE_MIN_FT, ARRIVAL_ALTITUDE_MAX_FT) & \
                 (df['vertrate'] < ARRIVAL_VERTRATE_THRESHOLD)
    df['arrival_icao'] = np.where(is_arrival, df['icao24'], np.nan)

    agg_df = df.groupby(['window', 'lat_bin', 'lon_bin']).agg({
        'icao24': 'nunique',
        'arrival_icao': 'nunique',
        'velocity': ['mean', 'std'],
        'altitude_ft': ['mean', 'std']
    })
    agg_df.columns = ['aircraft_count', 'arrival_pressure', 'mean_speed', 'speed_std', 'mean_altitude', 'altitude_std']
    agg_df = agg_df.reset_index().rename(columns={'window': 'timestamp'}).fillna(0)

    agg_df = agg_df.sort_values('timestamp')
    agg_df['global_gap'] = agg_df['timestamp'].diff() > timedelta(minutes=GAP_THRESHOLD_MINUTES)
    agg_df['global_island_id'] = agg_df['global_gap'].cumsum()
    return agg_df.drop(columns=['global_gap'])

def fill_empty_cells(df):
    log.info("Phase 4: Cartesian Filling...")
    active_cells = df[['lat_bin', 'lon_bin']].drop_duplicates()
    active_times = df[['global_island_id', 'timestamp']].drop_duplicates()
    
    expansion_size = len(active_times) * len(active_cells)
    if expansion_size > MAX_EXPANSION_ROWS:
        raise MemoryError(f"Cartesian product exceeds safety limit! ({expansion_size:,} > {MAX_EXPANSION_ROWS:,})")
    
    full_index = pd.merge(active_times.assign(k=1), active_cells.assign(k=1), on='k').drop(columns='k')
    df = pd.merge(full_index, df, on=['global_island_id', 'timestamp', 'lat_bin', 'lon_bin'], how='left').fillna(0)
    return df.sort_values(['global_island_id', 'timestamp', 'lat_bin', 'lon_bin'])

def engineer_features(df):
    log.info("Phase 5: Feature Engineering...")
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    g = df.groupby(['global_island_id', 'lat_bin', 'lon_bin'])
    df['lag_1'] = g['aircraft_count'].shift(1).fillna(0)
    df['lag_2'] = g['aircraft_count'].shift(2).fillna(0)
    return df

def construct_targets(df):
    log.info("Phase 6: Target Construction...")
    def local_norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    df['d_n'] = df.groupby('global_island_id')['aircraft_count'].transform(local_norm)
    df['a_n'] = df.groupby('global_island_id')['arrival_pressure'].transform(local_norm)
    df['v_n'] = df.groupby('global_island_id')['altitude_std'].transform(local_norm)
    
    df['congestion_score'] = (WEIGHTS['density']*df['d_n'] + WEIGHTS['arrival_pressure']*df['a_n'] + 
                              WEIGHTS['vertical_dispersion']*df['v_n']).clip(0, 1)
    
    g = df.groupby(['global_island_id', 'lat_bin', 'lon_bin'])
    df['target_15min'] = g['congestion_score'].shift(-3)
    df['target_30min'] = g['congestion_score'].shift(-6)
    return df.drop(columns=['d_n', 'a_n', 'v_n'])

# =============================================================================
# MODELING & VISUALIZATION
# =============================================================================

def train_model(df, target_col):
    log.info(f"Training Model for {target_col}...")
    valid = df[df[target_col].notna()].sort_values('timestamp')
    X, y = valid[FEATURE_COLS].values, valid[target_col].values
    
    tscv = TimeSeriesSplit(n_splits=3)
    mae, rmse, r2 = [], [], []
    
    for train_idx, val_idx in tscv.split(X):
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        mae.append(mean_absolute_error(y[val_idx], preds))
        rmse.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        r2.append(r2_score(y[val_idx], preds))
    
    final_model = RandomForestRegressor(**RF_PARAMS)
    final_model.fit(X, y)
    
    return {
        'model': final_model,
        'mae': np.mean(mae), 'rmse': np.mean(rmse), 'r2': np.mean(r2),
        'y_true': y, 'y_pred': final_model.predict(X),
        'feature_importances': final_model.feature_importances_,
        'df': valid
    }

def generate_review_artifacts(res_15, res_30, output_dir):
    log.info("Generating Review Artifacts...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Metrics Comparison
    metrics = ['MAE', 'RMSE', 'R2']
    m15 = [res_15['mae'], res_15['rmse'], res_15['r2']]
    m30 = [res_30['mae'], res_30['rmse'], res_30['r2']]
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(metrics):
        ax[i].bar(['15m', '30m'], [m15[i], m30[i]], color=['skyblue', 'salmon'])
        ax[i].set_title(name)
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))

    # 2. Feature Importance (30m)
    plt.figure(figsize=(10, 6))
    plt.barh(FEATURE_COLS, res_30['feature_importances'], color='teal')
    plt.title("Feature Importance (30-min Horizon)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_30min.png'))

    # 3. Timeseries (System-wide Average)
    df_plot = res_30['df'].copy()
    df_plot['y_pred'] = res_30['y_pred']
    ts_agg = df_plot.groupby('timestamp')[['target_30min', 'y_pred']].mean()
    
    plt.figure(figsize=(15, 6))
    plt.plot(ts_agg.index, ts_agg['target_30min'], label='Actual', alpha=0.7)
    plt.plot(ts_agg.index, ts_agg['y_pred'], label='Predicted', linestyle='--')
    plt.title("Bay Area Congestion: Actual vs 30-min Prediction")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'forecast_timeseries.png'))

def run_pipeline(input_path, output_dir):
    setup_logging(output_dir)
    df = pd.read_parquet(input_path)
    df = add_spatial_bins(df)
    df = detect_islands(df)
    df = aggregate_to_grid(df)
    df = fill_empty_cells(df)
    df = engineer_features(df)
    df = construct_targets(df)
    
    res_15 = train_model(df, 'target_15min')
    res_30 = train_model(df, 'target_30min')
    
    generate_review_artifacts(res_15, res_30, output_dir)
    
    joblib.dump(res_15['model'], os.path.join(output_dir, 'model_15min.joblib'))
    joblib.dump(res_30['model'], os.path.join(output_dir, 'model_30min.joblib'))
    log.info("PIPELINE COMPLETE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)