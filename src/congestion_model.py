#!/usr/bin/env python3
"""
Congestion Model - ML Training Pipeline
=============================================
Description: Trains the Hybrid Advection-Delta model for congestion prediction.
             Uses RandomForestRegressor to learn congestion change patterns from
             flow dynamics, neighbor pressure, and temporal features.
Author: RT Forecast Team
Version: 3.1.0

Architecture:
    - Kinematic Advection: Physics-based prediction of WHERE traffic moves
    - ML Delta Model: Learned prediction of HOW MUCH congestion changes

Key Design Decisions:
    - SPATIALLY BLIND: No lat/lon features (prevents map memorization)
    - UNIT INTEGRITY: Velocities in knots, displacement in degrees
    - HIGH-SIGNAL ONLY: Filters empty sky to focus on actual traffic

Usage:
    python src/congestion_model.py --input data/processed --output models/

Note: This is a standalone training utility, not part of the live pipeline.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress sklearn warnings during CV
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    format='%(asctime)s | %(levelname)-5s | %(name)s | %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ',
    level=logging.INFO
)
log = logging.getLogger('train_model')

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Spatial Configuration (matches rt_forecast/config.py)
CORE_BBOX = {
    'min_lat': 37.0,
    'max_lat': 38.0,
    'min_lon': -122.6,
    'max_lon': -121.5
}

GRID_RES = 0.05  # degrees

# Padded bounds (1-cell buffer for edge handling)
PADDED_BBOX = {
    'min_lat': CORE_BBOX['min_lat'] - GRID_RES,
    'max_lat': CORE_BBOX['max_lat'] + GRID_RES,
    'min_lon': CORE_BBOX['min_lon'] - GRID_RES,
    'max_lon': CORE_BBOX['max_lon'] + GRID_RES
}

# Neighbor offsets (cardinal directions only for high signal)
NEIGHBOR_OFFSETS = {
    'N':  ( GRID_RES,  0.0),
    'S':  (-GRID_RES,  0.0),
    'E':  ( 0.0,       GRID_RES),
    'W':  ( 0.0,      -GRID_RES),
}

# Physical constants
KNOTS_TO_DEG_PER_HOUR = 1.0 / 60.0  # 1 knot ≈ 1 nm/hr, 1° ≈ 60 nm
FORECAST_HORIZON_HOURS = 0.25  # 15 minutes
WINDOW_MINUTES = 5

# Descent pressure thresholds
DESCENT_VERT_RATE = -2.0  # m/s
DESCENT_ALT_CEILING = 10000.0  # feet

# Congestion calibration
CALIB_ANCHOR = 8.0  # aircraft count = 1.0 congestion

# =============================================================================
# FEATURE SCHEMA (SPATIALLY BLIND - NO LAT/LON)
# =============================================================================

FEATURE_COLS = [
    # Local cell metrics (current state)
    'aircraft_count',
    'arrival_pressure',
    'altitude_std',
    'mean_speed',
    # Vector flow (knots)
    'avg_u',
    'avg_v',
    # Temporal encoding
    'hour_sin',
    'hour_cos',
    # Neighbor context (t-5min lag for causality)
    'nb_N_count',
    'nb_S_count',
    'nb_E_count',
    'nb_W_count',
]

def compute_feature_hash() -> str:
    """SHA256 hash of feature schema for contract validation."""
    feature_str = ','.join(FEATURE_COLS)
    return hashlib.sha256(feature_str.encode()).hexdigest()[:16]

FEATURE_HASH = compute_feature_hash()

# =============================================================================
# DATA LOADING
# =============================================================================

def load_historical_data(data_dir: Path) -> pd.DataFrame:
    """
    Load historical ADS-B data from Parquet files.
    
    Expected schema:
        icao24, lat, lon, altitude_ft, velocity (knots), 
        vert_rate (m/s), true_track (degrees), timestamp
    """
    parquet_files = sorted(data_dir.glob('*.parquet'))
    
    if not parquet_files:
        log.warning(f"No parquet files found in {data_dir}")
        return pd.DataFrame()
    
    log.info(f"Loading {len(parquet_files)} parquet files from {data_dir}")
    
    dfs = []
    total_records = 0
    
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            total_records += len(df)
            dfs.append(df)
            log.debug(f"  Loaded {f.name}: {len(df)} records")
        except Exception as e:
            log.warning(f"  Failed to load {f.name}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Total records loaded: {total_records:,}")
    
    return combined


def generate_synthetic_week(seed: int = 42) -> pd.DataFrame:
    """
    Generate 1 week of synthetic ADS-B data for testing.
    
    Simulates realistic Bay Area traffic patterns:
    - Morning/evening rush hours
    - SFO/OAK approach corridors
    - Random GA traffic
    """
    np.random.seed(seed)
    log.info("Generating 1 week of synthetic ADS-B data...")
    
    records = []
    start_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    # Generate 5-minute windows for 7 days
    n_windows = 7 * 24 * 12  # 2016 windows
    
    for w in range(n_windows):
        window_time = start_time + timedelta(minutes=w * 5)
        hour = window_time.hour + window_time.minute / 60.0
        
        # Traffic volume varies by time of day (sinusoidal pattern)
        # Peak at 8am and 6pm, trough at 3am
        base_traffic = 30 + 25 * np.sin(np.pi * (hour - 3) / 12)
        base_traffic = max(5, int(base_traffic))
        
        # Add day-of-week variation (less on weekends)
        dow = window_time.weekday()
        if dow >= 5:  # Weekend
            base_traffic = int(base_traffic * 0.7)
        
        n_aircraft = np.random.poisson(base_traffic)
        
        for i in range(n_aircraft):
            # Aircraft type distribution
            aircraft_type = np.random.choice(
                ['commercial_arr', 'commercial_dep', 'ga', 'cargo'],
                p=[0.35, 0.30, 0.25, 0.10]
            )
            
            if aircraft_type == 'commercial_arr':
                # Approaching SFO/OAK - coming from various directions
                approach_dir = np.random.choice(['N', 'S', 'E'])
                if approach_dir == 'N':
                    lat = np.random.uniform(37.7, 38.0)
                    lon = np.random.uniform(-122.5, -122.2)
                    heading = np.random.uniform(160, 200)  # Southbound
                elif approach_dir == 'S':
                    lat = np.random.uniform(37.0, 37.4)
                    lon = np.random.uniform(-122.3, -121.8)
                    heading = np.random.uniform(340, 380) % 360  # Northbound
                else:
                    lat = np.random.uniform(37.4, 37.7)
                    lon = np.random.uniform(-121.6, -121.3)
                    heading = np.random.uniform(250, 290)  # Westbound
                
                altitude = np.random.uniform(2000, 8000)
                velocity = np.random.uniform(180, 250)
                vert_rate = np.random.uniform(-8, -2)
                
            elif aircraft_type == 'commercial_dep':
                # Departing - climbing out
                lat = np.random.uniform(37.5, 37.8)
                lon = np.random.uniform(-122.5, -122.1)
                heading = np.random.uniform(0, 360)
                altitude = np.random.uniform(3000, 15000)
                velocity = np.random.uniform(200, 300)
                vert_rate = np.random.uniform(5, 15)
                
            elif aircraft_type == 'ga':
                # General aviation - random patterns
                lat = np.random.uniform(37.2, 37.8)
                lon = np.random.uniform(-122.4, -121.8)
                heading = np.random.uniform(0, 360)
                altitude = np.random.uniform(1500, 8000)
                velocity = np.random.uniform(90, 150)
                vert_rate = np.random.uniform(-3, 3)
                
            else:  # cargo
                lat = np.random.uniform(37.6, 37.8)
                lon = np.random.uniform(-122.4, -122.0)
                heading = np.random.uniform(0, 360)
                altitude = np.random.uniform(5000, 20000)
                velocity = np.random.uniform(200, 280)
                vert_rate = np.random.uniform(-5, 5)
            
            records.append({
                'icao24': f'{aircraft_type[:2].upper()}{w:04d}{i:03d}',
                'lat': lat,
                'lon': lon,
                'altitude_ft': altitude,
                'velocity': velocity,  # Already in knots
                'vert_rate': vert_rate,
                'true_track': heading,
                'timestamp': window_time
            })
    
    df = pd.DataFrame(records)
    log.info(f"Generated {len(df):,} synthetic aircraft records over {n_windows} windows")
    
    return df


# =============================================================================
# GRID PROCESSING (STEP 1)
# =============================================================================

def process_to_grid(
    df: pd.DataFrame, 
    window_time: datetime,
    log_boundary_exits: bool = True
) -> Tuple[pd.DataFrame, int]:
    """
    Aggregate aircraft states to spatial grid with vector decomposition.
    
    UNIT INTEGRITY:
    - Input velocity: knots
    - Output avg_u/avg_v: knots (East+/North+)
    
    Args:
        df: Aircraft DataFrame for single window
        window_time: Window timestamp
        log_boundary_exits: Whether to count boundary exits
    
    Returns:
        Tuple of (grid DataFrame, boundary_exit_count)
    """
    if df.empty:
        return pd.DataFrame(), 0
    
    df = df.copy()
    
    # Count aircraft outside padded bounds (for audit)
    boundary_exits = 0
    if log_boundary_exits:
        outside_mask = (
            (df['lat'] < PADDED_BBOX['min_lat']) |
            (df['lat'] >= PADDED_BBOX['max_lat']) |
            (df['lon'] < PADDED_BBOX['min_lon']) |
            (df['lon'] >= PADDED_BBOX['max_lon'])
        )
        boundary_exits = outside_mask.sum()
    
    # Deterministic binning anchored to padded bbox
    df['lat_bin'] = np.round(
        np.floor((df['lat'] - PADDED_BBOX['min_lat']) / GRID_RES) * GRID_RES 
        + PADDED_BBOX['min_lat'], 4
    )
    df['lon_bin'] = np.round(
        np.floor((df['lon'] - PADDED_BBOX['min_lon']) / GRID_RES) * GRID_RES 
        + PADDED_BBOX['min_lon'], 4
    )
    
    # Filter to padded bounds
    mask = (
        (df['lat_bin'] >= PADDED_BBOX['min_lat']) &
        (df['lat_bin'] < PADDED_BBOX['max_lat']) &
        (df['lon_bin'] >= PADDED_BBOX['min_lon']) &
        (df['lon_bin'] < PADDED_BBOX['max_lon'])
    )
    df = df[mask]
    
    if df.empty:
        return pd.DataFrame(), boundary_exits
    
    # Vector decomposition (velocity already in knots)
    # Heading: 0=North, 90=East, 180=South, 270=West
    heading_rad = np.deg2rad(df['true_track'].fillna(0))
    
    # U = East component (positive = East)
    # V = North component (positive = North)
    df['u_component'] = df['velocity'].fillna(0) * np.sin(heading_rad)
    df['v_component'] = df['velocity'].fillna(0) * np.cos(heading_rad)
    
    # Arrival pressure: descending aircraft below ceiling
    is_arriving = (
        (df['vert_rate'] < DESCENT_VERT_RATE) &
        (df['altitude_ft'] < DESCENT_ALT_CEILING)
    )
    df['arrival_icao'] = np.where(is_arriving, df['icao24'], np.nan)
    
    # Aggregate by grid cell
    agg = df.groupby(['lat_bin', 'lon_bin']).agg({
        'icao24': 'nunique',
        'arrival_icao': 'nunique',
        'velocity': 'mean',
        'altitude_ft': 'std',
        'u_component': 'mean',
        'v_component': 'mean',
    }).reset_index()
    
    agg.columns = [
        'lat_bin', 'lon_bin', 'aircraft_count', 'arrival_pressure',
        'mean_speed', 'altitude_std', 'avg_u', 'avg_v'
    ]
    
    # Fill NaN altitude_std with 0 (single aircraft in cell)
    agg['altitude_std'] = agg['altitude_std'].fillna(0)
    
    # Add temporal features
    hour = window_time.hour + window_time.minute / 60.0
    agg['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    agg['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    agg['window_time'] = window_time
    
    return agg, boundary_exits


def compute_congestion(aircraft_count: float) -> float:
    """
    Convert aircraft count to congestion score [0, 1].
    
    Uses sigmoid calibration with CALIB_ANCHOR as midpoint.
    """
    density = aircraft_count / CALIB_ANCHOR
    return 1 / (1 + np.exp(-4 * (density - 0.5)))


# =============================================================================
# ADVECTION & TARGET CALCULATION (STEP 2)
# =============================================================================

def advect_position(
    lat: float, 
    lon: float, 
    avg_u: float, 
    avg_v: float,
    hours: float = FORECAST_HORIZON_HOURS
) -> Tuple[float, float]:
    """
    Calculate Expected Future Coordinate (EFC) via kinematic advection.
    
    Physics:
    - displacement (degrees) = velocity (knots) × time (hours) × (1°/60nm)
    - 1 knot = 1 nautical mile per hour
    - 1 degree ≈ 60 nautical miles
    
    Args:
        lat, lon: Current position
        avg_u: East-West velocity in knots (positive = East)
        avg_v: North-South velocity in knots (positive = North)
        hours: Forecast horizon (default 0.25 = 15 min)
    
    Returns:
        (efc_lat, efc_lon): Expected future coordinates
    """
    # Displacement in degrees
    delta_lat = avg_v * hours * KNOTS_TO_DEG_PER_HOUR
    delta_lon = avg_u * hours * KNOTS_TO_DEG_PER_HOUR
    
    efc_lat = lat + delta_lat
    efc_lon = lon + delta_lon
    
    return round(efc_lat, 4), round(efc_lon, 4)


def snap_to_grid(lat: float, lon: float) -> Tuple[float, float]:
    """Snap coordinates to nearest grid cell center."""
    lat_bin = np.round(
        np.floor((lat - PADDED_BBOX['min_lat']) / GRID_RES) * GRID_RES 
        + PADDED_BBOX['min_lat'], 4
    )
    lon_bin = np.round(
        np.floor((lon - PADDED_BBOX['min_lon']) / GRID_RES) * GRID_RES 
        + PADDED_BBOX['min_lon'], 4
    )
    return lat_bin, lon_bin


def build_training_samples(
    window_grids: Dict[datetime, pd.DataFrame],
    windows: List[datetime],
    horizon_minutes: int = 15,
    min_delta_threshold: float = 0.02
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build training dataset with Advection-Delta targets.
    
    TARGET: target_delta = congestion(T+15, EFC) - congestion(T, current)
    
    HIGH-SIGNAL FILTERING:
    - Keep rows where aircraft_count > 0 OR |target_delta| > threshold
    - This focuses the model on actual traffic patterns
    
    Args:
        window_grids: Dict mapping window_time -> grid DataFrame
        windows: Sorted list of window timestamps
        horizon_minutes: Forecast horizon
        min_delta_threshold: Minimum delta to keep empty cells
    
    Returns:
        Tuple of (training DataFrame, audit_stats dict)
    """
    horizon_delta = timedelta(minutes=horizon_minutes)
    lag_delta = timedelta(minutes=WINDOW_MINUTES)
    
    samples = []
    audit = {
        'total_cells': 0,
        'high_signal_cells': 0,
        'empty_dropped': 0,
        'boundary_advections': 0,
    }
    
    log.info(f"Building training samples with {horizon_minutes}min horizon...")
    
    for i, w in enumerate(windows):
        if w not in window_grids:
            continue
        
        # Need future window for target
        w_future = w + horizon_delta
        if w_future not in window_grids:
            continue
        
        # Get t-5min window for neighbor lags (causal)
        w_lag = w - lag_delta
        lag_grid = window_grids.get(w_lag, pd.DataFrame())
        lag_lookup = {}
        if not lag_grid.empty:
            for _, row in lag_grid.iterrows():
                key = (row['lat_bin'], row['lon_bin'])
                lag_lookup[key] = row['aircraft_count']
        
        current_grid = window_grids[w]
        future_grid = window_grids[w_future]
        
        # Build future lookup
        future_lookup = {}
        for _, row in future_grid.iterrows():
            key = (row['lat_bin'], row['lon_bin'])
            future_lookup[key] = row['aircraft_count']
        
        # Process each cell in current window
        for _, row in current_grid.iterrows():
            audit['total_cells'] += 1
            
            current_lat = row['lat_bin']
            current_lon = row['lon_bin']
            current_count = row['aircraft_count']
            
            # Current congestion
            current_cong = compute_congestion(current_count)
            
            # Advect to EFC
            efc_lat, efc_lon = advect_position(
                current_lat, current_lon,
                row['avg_u'], row['avg_v']
            )
            
            # Snap EFC to grid
            efc_lat_bin, efc_lon_bin = snap_to_grid(efc_lat, efc_lon)
            
            # Check if EFC is within bounds
            if not (PADDED_BBOX['min_lat'] <= efc_lat_bin < PADDED_BBOX['max_lat'] and
                    PADDED_BBOX['min_lon'] <= efc_lon_bin < PADDED_BBOX['max_lon']):
                audit['boundary_advections'] += 1
                # Clamp to boundary
                efc_lat_bin = np.clip(efc_lat_bin, PADDED_BBOX['min_lat'], 
                                       PADDED_BBOX['max_lat'] - GRID_RES)
                efc_lon_bin = np.clip(efc_lon_bin, PADDED_BBOX['min_lon'],
                                       PADDED_BBOX['max_lon'] - GRID_RES)
            
            # Future congestion at EFC
            future_count = future_lookup.get((efc_lat_bin, efc_lon_bin), 0)
            future_cong = compute_congestion(future_count)
            
            # TARGET: Delta congestion
            target_delta = future_cong - current_cong
            
            # HIGH-SIGNAL FILTER
            is_high_signal = (current_count > 0) or (abs(target_delta) > min_delta_threshold)
            
            if not is_high_signal:
                audit['empty_dropped'] += 1
                continue
            
            audit['high_signal_cells'] += 1
            
            # Neighbor features (from t-5min for causality)
            nb_features = {}
            for direction, (d_lat, d_lon) in NEIGHBOR_OFFSETS.items():
                nb_lat = round(current_lat + d_lat, 4)
                nb_lon = round(current_lon + d_lon, 4)
                nb_count = lag_lookup.get((nb_lat, nb_lon), 0)
                nb_features[f'nb_{direction}_count'] = nb_count
            
            # Build sample (SPATIALLY BLIND - no lat_bin/lon_bin)
            sample = {
                'aircraft_count': current_count,
                'arrival_pressure': row['arrival_pressure'],
                'altitude_std': row['altitude_std'],
                'mean_speed': row['mean_speed'],
                'avg_u': row['avg_u'],
                'avg_v': row['avg_v'],
                'hour_sin': row['hour_sin'],
                'hour_cos': row['hour_cos'],
                **nb_features,
                # Target
                'target_delta': target_delta,
                # Metadata (not features)
                '_lat_bin': current_lat,
                '_lon_bin': current_lon,
                '_efc_lat': efc_lat_bin,
                '_efc_lon': efc_lon_bin,
                '_window_time': w,
                '_current_cong': current_cong,
                '_future_cong': future_cong,
            }
            samples.append(sample)
    
    if not samples:
        return pd.DataFrame(), audit
    
    df = pd.DataFrame(samples)
    
    # Compute audit stats
    audit['filter_rate'] = audit['empty_dropped'] / max(audit['total_cells'], 1) * 100
    
    log.info(f"Training samples: {len(df):,}")
    log.info(f"  Total cells processed: {audit['total_cells']:,}")
    log.info(f"  High-signal retained: {audit['high_signal_cells']:,}")
    log.info(f"  Empty sky dropped: {audit['empty_dropped']:,} ({audit['filter_rate']:.1f}%)")
    log.info(f"  Boundary advections: {audit['boundary_advections']:,}")
    
    return df, audit


# =============================================================================
# FEATURE ENGINEERING (STEP 3)
# =============================================================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix and target vector.
    
    Handles:
    - NaN filling (zero for empty sky, forward fill for telemetry gaps)
    - Feature ordering per FEATURE_COLS contract
    """
    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    
    # Fill NaNs
    # Zero fill for count-based features (empty sky)
    zero_fill_cols = ['aircraft_count', 'arrival_pressure', 
                      'nb_N_count', 'nb_S_count', 'nb_E_count', 'nb_W_count']
    for col in zero_fill_cols:
        df[col] = df[col].fillna(0)
    
    # Median fill for noisy telemetry (velocity, altitude)
    median_fill_cols = ['altitude_std', 'mean_speed', 'avg_u', 'avg_v']
    for col in median_fill_cols:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    # Temporal features should never be NaN
    df['hour_sin'] = df['hour_sin'].fillna(0)
    df['hour_cos'] = df['hour_cos'].fillna(1)
    
    X = df[FEATURE_COLS].copy()
    y = df['target_delta'].copy()
    
    # Final NaN check
    if X.isna().any().any():
        nan_counts = X.isna().sum()
        log.warning(f"Remaining NaNs: {nan_counts[nan_counts > 0].to_dict()}")
        X = X.fillna(0)
    
    return X, y


# =============================================================================
# MODEL TRAINING (STEP 4)
# =============================================================================

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 20,
    min_samples_leaf: int = 4,
    cv_splits: int = 5,
    persistence_threshold: float = 0.85
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Train RandomForest with TimeSeriesSplit cross-validation.
    
    PERSISTENCE PENALTY:
    If aircraft_count importance > threshold, reduce max_depth and retrain
    to force the model to learn neighbor/velocity patterns.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
        cv_splits: Number of CV splits
        persistence_threshold: Max allowed importance for aircraft_count
    
    Returns:
        Tuple of (trained model, training_metrics dict)
    """
    log.info(f"Training RandomForest (n={n_estimators}, depth={max_depth})...")
    
    metrics = {
        'n_samples': len(X),
        'n_features': len(FEATURE_COLS),
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'retrained': False,
    }
    
    # Initial training
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    # TimeSeriesSplit cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    log.info(f"  CV RMSE: {cv_rmse:.4f} (+/- {np.sqrt(-cv_scores).std():.4f})")
    metrics['cv_rmse'] = cv_rmse
    metrics['cv_rmse_std'] = np.sqrt(-cv_scores).std()
    
    # Fit on full data
    model.fit(X, y)
    
    # Check feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    aircraft_importance = importance.get('aircraft_count', 0)
    
    log.info(f"  aircraft_count importance: {aircraft_importance:.2%}")
    
    # PERSISTENCE PENALTY CHECK
    if aircraft_importance > persistence_threshold:
        log.warning(f"  PERSISTENCE PENALTY: aircraft_count > {persistence_threshold:.0%}")
        log.warning(f"  Retraining with max_depth={max_depth - 5} to reduce memorization...")
        
        new_depth = max(5, max_depth - 5)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=new_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        metrics['retrained'] = True
        metrics['original_max_depth'] = max_depth
        metrics['max_depth'] = new_depth
        
        # Recalculate importance
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))
        new_aircraft_importance = importance.get('aircraft_count', 0)
        log.info(f"  New aircraft_count importance: {new_aircraft_importance:.2%}")
    
    metrics['feature_importance'] = importance
    
    return model, metrics


# =============================================================================
# PRODUCTION DIAGNOSTICS (STEP 5)
# =============================================================================

def compute_diagnostics(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute comprehensive production diagnostics.
    
    Metrics:
    - RMSE: Root Mean Square Error
    - R²: Coefficient of Determination
    - MAE: Mean Absolute Error
    - RAE: Relative Absolute Error
    - MAPE: Mean Absolute Percentage Error (for active cells)
    """
    y_pred = model.predict(X)
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # RAE (Relative Absolute Error)
    y_mean = y.mean()
    rae = mae / (np.abs(y - y_mean).mean() + 1e-8)
    
    # MAPE for active cells only (where current_cong > 0.05)
    active_mask = df_full['_current_cong'] > 0.05
    if active_mask.sum() > 0:
        y_active = y[active_mask]
        y_pred_active = y_pred[active_mask]
        # Avoid division by zero
        denom = np.abs(y_active) + 1e-8
        mape = (np.abs(y_active - y_pred_active) / denom).mean() * 100
    else:
        mape = np.nan
    
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'rae': rae,
        'mape_active': mape,
        'y_mean': y.mean(),
        'y_std': y.std(),
        'y_pred_mean': y_pred.mean(),
        'y_pred_std': y_pred.std(),
    }
    
    log.info("=" * 60)
    log.info("PRODUCTION DIAGNOSTICS")
    log.info("=" * 60)
    log.info(f"  RMSE:  {rmse:.4f}")
    log.info(f"  R²:    {r2:.4f}")
    log.info(f"  MAE:   {mae:.4f}")
    log.info(f"  RAE:   {rae:.4f}")
    log.info(f"  MAPE (active cells): {mape:.2f}%")
    log.info(f"  Target mean: {y.mean():.4f}, std: {y.std():.4f}")
    
    return metrics


def plot_diagnostics(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame,
    output_dir: Path
):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available - skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=True)
    
    colors = ['#FF6B6B' if 'count' in name else '#4ECDC4' for name in importance.index]
    importance.plot(kind='barh', ax=ax, color=colors)
    
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Red=Count, Teal=Other)')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150)
    plt.close()
    log.info(f"  Saved: feature_importance.png")
    
    # 2. Prediction vs Actual
    fig, ax = plt.subplots(figsize=(8, 8))
    y_pred = model.predict(X)
    
    ax.scatter(y, y_pred, alpha=0.3, s=2)
    ax.plot([-0.5, 0.5], [-0.5, 0.5], 'r--', lw=2)
    ax.set_xlabel('Actual Delta')
    ax.set_ylabel('Predicted Delta')
    ax.set_title('Prediction vs Actual')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=150)
    plt.close()
    log.info(f"  Saved: prediction_scatter.png")
    
    # 3. Forecast Timeseries (sample)
    if '_window_time' in df_full.columns:
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Sample a day
        df_plot = df_full.copy()
        df_plot['y_pred'] = y_pred
        
        # Aggregate by window
        ts = df_plot.groupby('_window_time').agg({
            '_current_cong': 'mean',
            '_future_cong': 'mean',
            'target_delta': 'mean',
            'y_pred': 'mean'
        }).reset_index()
        
        # Plot first 24 hours
        ts = ts.head(288)  # 24 hours × 12 windows/hour
        
        ax.plot(ts['_window_time'], ts['_current_cong'], label='Current Cong', alpha=0.7)
        ax.plot(ts['_window_time'], ts['_future_cong'], label='Future Cong (+15m)', alpha=0.7)
        ax.plot(ts['_window_time'], ts['target_delta'], label='Actual Delta', alpha=0.7)
        ax.plot(ts['_window_time'], ts['y_pred'], label='Predicted Delta', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Congestion / Delta')
        ax.set_title('Forecast Timeseries (24h sample)')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'forecast_timeseries.png', dpi=150)
        plt.close()
        log.info(f"  Saved: forecast_timeseries.png")


# =============================================================================
# SERIALIZATION
# =============================================================================

def save_model(
    model: RandomForestRegressor,
    training_metrics: Dict,
    diagnostics: Dict,
    audit: Dict,
    output_dir: Path,
    horizon: int = 15
):
    """
    Save model with compression and metadata sidecar.
    
    Includes SHA256 checksum for integrity verification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f'model_{horizon}min.joblib'
    meta_path = output_dir / f'model_{horizon}min.meta.json'
    
    # Save compressed model
    log.info(f"Saving model (compress=3): {model_path}")
    joblib.dump(model, model_path, compress=3)
    
    # Compute checksum
    with open(model_path, 'rb') as f:
        model_checksum = hashlib.sha256(f.read()).hexdigest()
    
    # Build metadata
    metadata = {
        'version': '3.1.0',
        'architecture': 'advection_delta',
        'feature_hash': FEATURE_HASH,
        'feature_cols': FEATURE_COLS,
        'horizon_minutes': horizon,
        'model_type': 'RandomForestRegressor',
        'model_checksum_sha256': model_checksum,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'training': {
            'n_samples': training_metrics['n_samples'],
            'n_features': training_metrics['n_features'],
            'n_estimators': training_metrics['n_estimators'],
            'max_depth': training_metrics['max_depth'],
            'cv_rmse': training_metrics['cv_rmse'],
            'retrained_for_persistence': training_metrics['retrained'],
        },
        'diagnostics': {
            'rmse': diagnostics['rmse'],
            'r2': diagnostics['r2'],
            'mae': diagnostics['mae'],
            'rae': diagnostics['rae'],
            'mape_active': diagnostics['mape_active'],
        },
        'audit': {
            'total_cells': audit['total_cells'],
            'high_signal_cells': audit['high_signal_cells'],
            'empty_dropped': audit['empty_dropped'],
            'filter_rate_pct': audit['filter_rate'],
        },
        'feature_importance': training_metrics['feature_importance'],
        'physical_constants': {
            'grid_res_deg': GRID_RES,
            'forecast_horizon_hours': FORECAST_HORIZON_HOURS,
            'knots_to_deg_per_hour': KNOTS_TO_DEG_PER_HOUR,
            'calib_anchor': CALIB_ANCHOR,
        }
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log.info(f"Saved metadata: {meta_path}")
    log.info(f"  Model checksum: {model_checksum[:16]}...")
    
    return model_path, meta_path


# =============================================================================
# MODEL INTERFACE (STEP 6)
# =============================================================================

class ModelInterface:
    """
    Helper class demonstrating inference integration.
    
    Shows how to:
    1. Load model and validate contract
    2. Prepare features (spatially blind)
    3. Predict delta
    4. Map delta back to EFC coordinates
    
    USAGE IN infer.py:
        interface = ModelInterface('models/model_15min.joblib')
        delta = interface.predict_delta(features)
        efc_lat, efc_lon = interface.advect(lat, lon, avg_u, avg_v)
        future_congestion = current_congestion + delta
    """
    
    def __init__(self, model_path: str):
        """Load model and validate feature contract."""
        self.model_path = Path(model_path)
        self.meta_path = self.model_path.with_suffix('.meta.json')
        
        # Load model
        self.model = joblib.load(self.model_path)
        
        # Load and validate metadata
        if self.meta_path.exists():
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)
            
            stored_hash = self.metadata.get('feature_hash', '')
            if stored_hash != FEATURE_HASH:
                raise ValueError(
                    f"Feature hash mismatch! "
                    f"Expected: {FEATURE_HASH}, Got: {stored_hash}"
                )
        else:
            self.metadata = {}
    
    @staticmethod
    def advect(
        lat: float, 
        lon: float, 
        avg_u: float, 
        avg_v: float,
        hours: float = FORECAST_HORIZON_HOURS
    ) -> Tuple[float, float]:
        """
        Calculate Expected Future Coordinate (EFC).
        
        This is the KINEMATIC part of the hybrid model.
        
        Args:
            lat, lon: Current grid cell coordinates
            avg_u: East-West velocity (knots, positive=East)
            avg_v: North-South velocity (knots, positive=North)
            hours: Forecast horizon (default 0.25 = 15min)
        
        Returns:
            (efc_lat, efc_lon): Where congestion is expected to move
        """
        delta_lat = avg_v * hours * KNOTS_TO_DEG_PER_HOUR
        delta_lon = avg_u * hours * KNOTS_TO_DEG_PER_HOUR
        
        return lat + delta_lat, lon + delta_lon
    
    def prepare_features(self, row: Dict) -> np.ndarray:
        """
        Prepare feature vector for single prediction.
        
        IMPORTANT: Features are SPATIALLY BLIND (no lat/lon).
        
        Args:
            row: Dictionary with feature values
        
        Returns:
            numpy array in FEATURE_COLS order
        """
        features = []
        for col in FEATURE_COLS:
            features.append(row.get(col, 0))
        return np.array(features).reshape(1, -1)
    
    def predict_delta(self, features: np.ndarray) -> float:
        """
        Predict congestion delta.
        
        This is the ML part of the hybrid model.
        
        Args:
            features: Feature array (1, n_features)
        
        Returns:
            Predicted delta (future_congestion - current_congestion)
        """
        return self.model.predict(features)[0]
    
    def forecast(
        self, 
        row: Dict,
        current_congestion: float
    ) -> Tuple[float, float, float, float]:
        """
        Full forecast: WHERE and HOW MUCH.
        
        Args:
            row: Feature dictionary including lat_bin, lon_bin, avg_u, avg_v
            current_congestion: Current congestion score
        
        Returns:
            (efc_lat, efc_lon, delta, future_congestion)
        """
        # WHERE (kinematic)
        efc_lat, efc_lon = self.advect(
            row['lat_bin'], row['lon_bin'],
            row['avg_u'], row['avg_v']
        )
        
        # HOW MUCH (ML)
        features = self.prepare_features(row)
        delta = self.predict_delta(features)
        
        # Combined forecast
        future_congestion = np.clip(current_congestion + delta, 0, 1)
        
        return efc_lat, efc_lon, delta, future_congestion


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RT Forecast - Production Model Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ARCHITECTURE: Hybrid Advection-Delta
  - Kinematic: Predicts WHERE congestion moves (physics)
  - ML Delta: Predicts HOW MUCH congestion changes (learned)

EXAMPLES:
  # Train with synthetic data (1 week)
  python train_model.py --synthetic
  
  # Train with historical data
  python train_model.py --data-dir ./data/historical
  
  # Custom configuration
  python train_model.py --synthetic --n-estimators 300 --max-depth 25
        """
    )
    
    parser.add_argument('--data-dir', type=Path, default=Path('data/historical'),
                        help='Directory containing historical parquet files')
    parser.add_argument('--output-dir', type=Path, default=Path('models'),
                        help='Output directory for model artifacts')
    parser.add_argument('--plot-dir', type=Path, default=Path('output'),
                        help='Directory for diagnostic plots')
    parser.add_argument('--horizon', type=int, default=15,
                        help='Forecast horizon in minutes (default: 15)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees (default: 200)')
    parser.add_argument('--max-depth', type=int, default=20,
                        help='Max tree depth (default: 20)')
    parser.add_argument('--cv-splits', type=int, default=5,
                        help='TimeSeriesSplit CV folds (default: 5)')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating diagnostic plots')
    
    args = parser.parse_args()
    
    # Banner
    log.info("=" * 70)
    log.info("RT FORECAST - PRODUCTION MODEL TRAINING PIPELINE")
    log.info("=" * 70)
    log.info(f"Architecture: Hybrid Advection-Delta")
    log.info(f"Feature Hash: {FEATURE_HASH}")
    log.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    log.info("=" * 70)
    
    # =================================================================
    # STEP 1: Load/Generate Data
    # =================================================================
    log.info("\n[STEP 1] DATA LOADING")
    
    if args.synthetic:
        raw_df = generate_synthetic_week()
    else:
        raw_df = load_historical_data(args.data_dir)

    if raw_df.empty:
        log.error("No data found. Use --synthetic for testing.")
        return 1

    # =================================================================
    # Schema Standardization (align with bayarea_state_vectors.parquet)
    # =================================================================
    log.info("Standardizing column names...")

    # Rename heading variants to true_track
    if 'heading' in raw_df.columns:
        raw_df = raw_df.rename(columns={'heading': 'true_track'})
        log.info("  Renamed 'heading' → 'true_track'")
    elif 'track' in raw_df.columns:
        raw_df = raw_df.rename(columns={'track': 'true_track'})
        log.info("  Renamed 'track' → 'true_track'")

    # Rename vertrate to vert_rate
    if 'vertrate' in raw_df.columns:
        raw_df = raw_df.rename(columns={'vertrate': 'vert_rate'})
        log.info("  Renamed 'vertrate' → 'vert_rate'")

    # Defensive validation
    required_cols = ['true_track', 'vert_rate', 'velocity', 'lat', 'lon', 'altitude_ft', 'icao24', 'timestamp']
    missing_cols = [col for col in required_cols if col not in raw_df.columns]

    if missing_cols:
        log.error(f"Missing required columns: {missing_cols}")
        log.error(f"Available columns: {raw_df.columns.tolist()}")
        return 1

    # Fill NaN values before conversion (prevent math errors)
    raw_df['true_track'] = raw_df['true_track'].fillna(0)
    raw_df['velocity'] = raw_df['velocity'].fillna(0)
    raw_df['vert_rate'] = raw_df['vert_rate'].fillna(0)

    log.info(f"Schema validation passed. Columns: {raw_df.columns.tolist()}")
    
    # Parse timestamps
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    raw_df['window'] = raw_df['timestamp'].dt.floor(f'{WINDOW_MINUTES}min')
    
    windows = sorted(raw_df['window'].unique())
    log.info(f"Data spans {len(windows)} windows ({windows[0]} to {windows[-1]})")
    
    # =================================================================
    # STEP 2: Grid Processing
    # =================================================================
    log.info("\n[STEP 2] GRID PROCESSING")
    
    window_grids = {}
    total_boundary_exits = 0
    
    for w in windows:
        window_df = raw_df[raw_df['window'] == w]
        grid, boundary_exits = process_to_grid(window_df, w)
        total_boundary_exits += boundary_exits
        if not grid.empty:
            window_grids[w] = grid
    
    log.info(f"Processed {len(window_grids)} windows with data")
    log.info(f"Total boundary exits logged: {total_boundary_exits}")
    
    # =================================================================
    # STEP 3: Build Training Samples
    # =================================================================
    log.info("\n[STEP 3] TRAINING SAMPLE CONSTRUCTION")
    
    df_train, audit = build_training_samples(
        window_grids, windows, 
        horizon_minutes=args.horizon
    )
    
    if df_train.empty:
        log.error("No training samples generated!")
        return 1
    
    # =================================================================
    # STEP 4: Feature Engineering
    # =================================================================
    log.info("\n[STEP 4] FEATURE ENGINEERING")
    
    X, y = prepare_features(df_train)
    log.info(f"Feature matrix shape: {X.shape}")
    log.info(f"Target vector shape: {y.shape}")
    
    # =================================================================
    # STEP 5: Model Training
    # =================================================================
    log.info("\n[STEP 5] MODEL TRAINING")
    
    model, training_metrics = train_model(
        X, y,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        cv_splits=args.cv_splits
    )
    
    # =================================================================
    # STEP 6: Production Diagnostics
    # =================================================================
    log.info("\n[STEP 6] PRODUCTION DIAGNOSTICS")
    
    diagnostics = compute_diagnostics(model, X, y, df_train)
    
    if not args.skip_plots:
        log.info("\nGenerating diagnostic plots...")
        plot_diagnostics(model, X, y, df_train, args.plot_dir)
    
    # =================================================================
    # STEP 7: Serialization
    # =================================================================
    log.info("\n[STEP 7] MODEL SERIALIZATION")
    
    model_path, meta_path = save_model(
        model, training_metrics, diagnostics, audit,
        args.output_dir, args.horizon
    )
    
    # =================================================================
    # STEP 8: Interface Demo
    # =================================================================
    log.info("\n[STEP 8] INFERENCE INTERFACE DEMO")
    
    interface = ModelInterface(str(model_path))
    
    # Demo prediction
    sample_row = {
        'lat_bin': 37.6,
        'lon_bin': -122.3,
        'aircraft_count': 5,
        'arrival_pressure': 2,
        'altitude_std': 1500,
        'mean_speed': 180,
        'avg_u': 50,   # 50 knots eastward
        'avg_v': -30,  # 30 knots southward
        'hour_sin': 0.5,
        'hour_cos': 0.87,
        'nb_N_count': 3,
        'nb_S_count': 2,
        'nb_E_count': 1,
        'nb_W_count': 4,
    }
    
    current_cong = compute_congestion(sample_row['aircraft_count'])
    efc_lat, efc_lon, delta, future_cong = interface.forecast(sample_row, current_cong)
    
    log.info("  Sample forecast:")
    log.info(f"    Current position: ({sample_row['lat_bin']}, {sample_row['lon_bin']})")
    log.info(f"    Current congestion: {current_cong:.3f}")
    log.info(f"    Predicted EFC: ({efc_lat:.4f}, {efc_lon:.4f})")
    log.info(f"    Predicted delta: {delta:+.4f}")
    log.info(f"    Future congestion: {future_cong:.3f}")
    
    # =================================================================
    # Summary
    # =================================================================
    log.info("\n" + "=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"""
ARTIFACTS:
  Model:    {model_path}
  Metadata: {meta_path}
  Plots:    {args.plot_dir}/*.png

METRICS:
  RMSE: {diagnostics['rmse']:.4f}
  R²:   {diagnostics['r2']:.4f}
  RAE:  {diagnostics['rae']:.4f}
  MAPE: {diagnostics['mape_active']:.2f}%

FEATURE IMPORTANCE (Top 5):
""")
    
    importance = training_metrics['feature_importance']
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importance[:5]:
        log.info(f"    {feat}: {imp:.4f}")
    
    log.info("\n" + "=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())