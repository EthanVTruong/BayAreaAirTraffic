"""
Preproc - Grid Preprocessing
=============================================
Description: Aggregates raw aircraft states into a spatial grid with vector
             flow extraction. Calculates U/V velocity components from speed
             and heading for advection physics. Detects arrival pressure via
             descent rate and altitude thresholds.
Author: RT Forecast Team
Version: 3.1.0
"""

import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    CORE_BBOX, PADDED_BBOX, GRID_RES,
    DESCENT_VERT_RATE, DESCENT_ALT_CEILING
)

log = logging.getLogger('rt_forecast.preproc')

def process_grid(states: List[Dict], window_time: datetime) -> pd.DataFrame:
    """
    Aggregate aircraft states into a grid with physical flow vectors.
    """
    if not states:
        log.warning("Empty states list received")
        return pd.DataFrame()
    
    df = pd.DataFrame(states)
    
    # 1. Deterministic Binning (Round to 4 decimals for O(1) state lookups)
    df['lat_bin'] = np.round(
        np.floor((df['lat'] - PADDED_BBOX['min_lat']) / GRID_RES) * GRID_RES + PADDED_BBOX['min_lat'],
        4
    )
    df['lon_bin'] = np.round(
        np.floor((df['lon'] - PADDED_BBOX['min_lon']) / GRID_RES) * GRID_RES + PADDED_BBOX['min_lon'],
        4
    )
    
    # 2. Bound Filtering (Padded area for neighbor context)
    mask = (
        (df['lat_bin'] >= PADDED_BBOX['min_lat']) & 
        (df['lat_bin'] < PADDED_BBOX['max_lat']) &
        (df['lon_bin'] >= PADDED_BBOX['min_lon']) & 
        (df['lon_bin'] < PADDED_BBOX['max_lon'])
    )
    df = df[mask].copy()
    
    if df.empty:
        return pd.DataFrame()

    # 3. Vector Decomposition (Aviation Convention: 0=N, 90=E)
    # [Image of a vector field diagram showing fluid flow or air traffic movement across a grid]
    rads = np.deg2rad(df['true_track'].fillna(0))
    df['u_comp'] = df['velocity'] * np.sin(rads)  # Eastward
    df['v_comp'] = df['velocity'] * np.cos(rads)  # Northward
    
    # 4. Semantic Features
    is_descending = (df['vert_rate'] < DESCENT_VERT_RATE) & (df['altitude_ft'] < DESCENT_ALT_CEILING)
    df['descent_icao'] = np.where(is_descending, df['icao24'], np.nan)
    
    # 5. Fast Grouped Aggregation
    agg = df.groupby(['lat_bin', 'lon_bin']).agg({
        'icao24': 'nunique',
        'descent_icao': 'nunique',
        'velocity': 'mean',
        'u_comp': 'mean',
        'v_comp': 'mean',
        'altitude_ft': ['mean', 'std'],
        'vert_rate': 'mean'
    }).reset_index()
    
    # Flatten multi-index columns
    agg.columns = [
        'lat_bin', 'lon_bin', 'aircraft_count', 'descent_pressure', 
        'mean_speed', 'avg_u', 'avg_v', 'altitude_mean', 'altitude_std', 'vertical_rate_mean'
    ]
    
    log.debug(f"Grid v3.1: Generated {len(agg)} flow-aware cells")
    return agg.fillna(0)

def filter_to_core(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to visible (CORE) bounds for UI and final metrics."""
    if df.empty: return df
    return df[
        (df['lat_bin'] >= CORE_BBOX['min_lat']) & 
        (df['lat_bin'] < CORE_BBOX['max_lat']) &
        (df['lon_bin'] >= CORE_BBOX['min_lon']) & 
        (df['lon_bin'] < CORE_BBOX['max_lon'])
    ]