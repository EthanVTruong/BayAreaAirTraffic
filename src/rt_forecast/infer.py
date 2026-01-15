"""
Infer - ML Inference Engine
=============================================
Description: Hybrid Advection-Delta inference engine for congestion prediction.
             Combines kinematic flow physics (WHERE traffic moves) with ML delta
             prediction (HOW MUCH congestion changes). Uses Z-score normalization
             for surge detection and sigmoid scaling for radar-style visualization.
Author: RT Forecast Team
Version: 3.5.4

Architecture:
    1. Kinematic Flow Compression: Inflow vs Outflow per grid cell
    2. Z-Score Normalization: Drift-protected intensity scaling
    3. ML Delta Prediction: RandomForest predicts congestion changes
    4. EMA Smoothing: Temporal consistency for visual stability
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from .config import (
    FEATURE_COLS, FEATURE_HASH, GRID_RES, 
    DR_DECAY_BASE, EMA_ALPHA, EMA_TTL, MODELS_DIR, AIRPORTS,
    ADVECTION_HOURS, NM_PER_DEGREE, SIGMOID_CENTER
)

log = logging.getLogger('rt_forecast.infer')

# === FLOW DYNAMICS CONSTANTS ===
CELL_CAPACITY = 8.0           # Normalization baseline
DESCENT_THRESHOLD = -500.0    # fpm
DESCENT_WEIGHT = 2.0          # Multiplier for arriving aircraft
RUNWAY_TOLERANCE = 10.0       # Degrees

# Major runway headings (magnetic) for Bay Area corridor detection
RUNWAY_HEADINGS = {
    'KSFO': [284, 104, 10, 190],
    'KOAK': [282, 102, 120, 300],
    'KSJC': [302, 122, 119, 299],
}

MODEL_15MIN = MODELS_DIR / 'model_15min.joblib'
MODEL_METADATA = MODELS_DIR / 'model_15min.meta.json'

class ModelContractError(Exception):
    """Raised when the loaded model does not match the current feature hash."""
    pass

class InferenceEngine:
    def __init__(self):
        self.model_15: Any = None
        self.ema_state: Dict[Tuple[float, float], Dict] = {}
        self.ema_state_current: Dict[Tuple[float, float], Dict] = {}
        self.iteration_count: int = 0
        self.grid_state_t1: Dict[Tuple[float, float], Dict] = {}
        self.flow_buffer: List[Dict[Tuple, float]] = []
        self.zscore_stats: Dict[str, Tuple[float, float]] = {}

    def load_models(self) -> bool:
        """Load and verify ML models with strict contract validation."""
        if not MODEL_15MIN.exists():
            log.error(f"Model missing: {MODEL_15MIN}")
            return False

        if MODEL_METADATA.exists():
            try:
                with open(MODEL_METADATA, 'r') as f:
                    meta = json.load(f)
                if meta.get('feature_hash') != FEATURE_HASH:
                    raise ModelContractError(f"Hash mismatch. Expected: {FEATURE_HASH}")
                log.info(f"Engine Ready: {meta.get('architecture')} v{meta.get('version')}")
            except Exception as e:
                log.error(f"Contract check failed: {e}")
                return False
        
        try:
            self.model_15 = joblib.load(MODEL_15MIN)
            return True
        except Exception as e:
            log.error(f"Model Load Failure: {e}")
            return False

    def predict(self, agg_df: pd.DataFrame, window_time: datetime, status: str, dr_count: int = 0) -> Dict[str, Any]:
        """
        Main Inference Pipeline.
        Calculates current congestion and projects 15-minute forecast.
        """
        if agg_df.empty or self.model_15 is None:
            return {'current': pd.DataFrame(), '15min': pd.DataFrame()}
        
        self.iteration_count += 1
        current_grid = {(round(r['lat_bin'], 4), round(r['lon_bin'], 4)): r.to_dict() for _, r in agg_df.iterrows()}
        
        # 1. Feature Construction
        df = self._engineer_features(agg_df, window_time, current_grid)
        
        # 2. Kinematic Flow Compression
        flow_pressure = self._compute_compression(df, current_grid)
        df['flow_pressure'] = flow_pressure
        
        # 3. Surge Normalization (Z-Score)
        z_flow = self._zscore_normalize(flow_pressure, 'flow_pressure')
        
        # 4. LIVE RADAR SCALING
        # Centered at 0.25 for higher sensitivity to background traffic.
        # Slope of -15 for aggressive visual transitions.
        raw_cur = 1 / (1 + np.exp(-15 * (z_flow - 0.25)))
        if status == 'DR': raw_cur *= (DR_DECAY_BASE ** dr_count)
        df['current_congestion'] = self._process_ema(df, np.clip(raw_cur, 0, 1), self.ema_state_current)
        
        # 5. Advection Coordinates (Where planes will be in 15m)
        df['efc_lat'] = df['lat_bin'] + (df['avg_v'].fillna(0) * ADVECTION_HOURS) / NM_PER_DEGREE
        df['efc_lon'] = df['lon_bin'] + (df['avg_u'].fillna(0) * ADVECTION_HOURS) / NM_PER_DEGREE
        
        # 6. ML Delta Prediction
        X_df = pd.DataFrame(df[FEATURE_COLS]).fillna(0.0)
        ml_delta = self.model_15.predict(X_df)

        # 7. FORECAST RADAR SCALING
        # Use z-score normalized flow as base (same as live), delta as adjustment.
        # This ensures forecast values are on the same scale as live radar.
        # ml_delta is centered at 0, so adding it adjusts the normalized base up/down.
        severity_base = np.clip(z_flow + ml_delta, 0, 1)
        calib_fct = 1 / (1 + np.exp(-12 * (severity_base - SIGMOID_CENTER)))
        if status == 'DR': calib_fct *= (DR_DECAY_BASE ** dr_count)
        df['predicted_15min'] = self._process_ema(df, calib_fct, self.ema_state)
        
        # 8. Tactical Trend Labeling
        df['trend_direction'] = self._compute_trend(flow_pressure, df['inflow_accel'].values)
        
        # Housekeeping
        self._update_flow_buffer(df)
        self.grid_state_t1 = current_grid
        self._prune_ema()
        
        return {'current': df.copy(), '15min': df.copy()}

    def _compute_compression(self, df: pd.DataFrame, current_grid: Dict) -> np.ndarray:
        """Calculates net aircraft flow (Inflow - Outflow)."""
        compression = np.zeros(len(df))
        for i, row in enumerate(df.itertuples()):
            lat, lon = round(row.lat_bin, 4), round(row.lon_bin, 4)
            u, v = getattr(row, 'avg_u', 0) or 0, getattr(row, 'avg_v', 0) or 0
            speed, count = np.sqrt(u**2 + v**2), getattr(row, 'aircraft_count', 0)
            
            # Outflow: based on speed relative to cell size
            outflow = count * min(1.0, (speed * ADVECTION_HOURS) / (NM_PER_DEGREE * GRID_RES))
            
            # Inflow: Sum traffic from 4 cardinal neighbors
            inflow = 0.0
            for d, (dl, dln) in {'N':(GRID_RES,0), 'S':(-GRID_RES,0), 'E':(0,GRID_RES), 'W':(0,-GRID_RES)}.items():
                nb = current_grid.get((round(lat + dl, 4), round(lon + dln, 4)), {})
                if not nb: continue
                nu, nv, nc = nb.get('avg_u', 0) or 0, nb.get('avg_v', 0) or 0, nb.get('aircraft_count', 0)
                
                # Check if vector points toward center cell
                toward = 0.0
                if d == 'N' and nv < 0: toward = abs(nv)
                elif d == 'S' and nv > 0: toward = abs(nv)
                elif d == 'E' and nu < 0: toward = abs(nu)
                elif d == 'W' and nu > 0: toward = abs(nu)
                
                weight = DESCENT_WEIGHT if nb.get('is_descending', False) else 1.0
                inflow += nc * min(1.0, (toward * ADVECTION_HOURS) / (NM_PER_DEGREE * GRID_RES)) * weight
            
            compression[i] = (inflow - outflow) / CELL_CAPACITY
        return compression

    def _engineer_features(self, df: pd.DataFrame, window_time: datetime, current_grid: Dict) -> pd.DataFrame:
        """Constructs the spatial-blind feature vector for the ML model."""
        df = df.copy()
        hour = window_time.hour + window_time.minute / 60.0
        df['hour_sin'], df['hour_cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        
        # 5-VARIABLE UNPACKING FIX
        airport_coords = [(lat, lon) for lat, lon, _, _, _ in AIRPORTS.values()]
        df['dist_to_airport'] = df.apply(
            lambda r: min(np.sqrt((r['lat_bin'] - a[0])**2 + (r['lon_bin'] - a[1])**2) for a in airport_coords), axis=1)
        
        # Descent status
        df['is_descending'] = (df['vertical_rate_mean'].fillna(0) < DESCENT_THRESHOLD).astype(float) if 'vertical_rate_mean' in df.columns else 0.0
        
        # Corridor alignment
        def check_runway_aligned(row):
            track = row.get('mean_track', row.get('true_track', 0)) or 0
            for apt, headings in RUNWAY_HEADINGS.items():
                # Unpack 5 variables
                apt_lat, apt_lon, _, _, _ = AIRPORTS[apt]
                if np.sqrt((row['lat_bin'] - apt_lat)**2 + (row['lon_bin'] - apt_lon)**2) < 0.3:
                    for hdg in headings:
                        if abs((track - hdg + 180) % 360 - 180) <= RUNWAY_TOLERANCE: return 1.0
            return 0.0
        df['runway_aligned'] = df.apply(check_runway_aligned, axis=1)
        
        # Neighbor context
        src = self.grid_state_t1 or current_grid
        for d, (dl, dln) in {'N':(GRID_RES,0), 'S':(-GRID_RES,0), 'E':(0,GRID_RES), 'W':(0,-GRID_RES)}.items():
            df[f'nb_{d}_count'] = df.apply(
                lambda r, dl=dl, dln=dln: src.get((round(r['lat_bin']+dl,4), round(r['lon_bin']+dln,4)), {}).get('aircraft_count', 0), axis=1)
        
        df['inflow_accel'] = self._compute_inflow_acceleration(df)
        for col in FEATURE_COLS:
            if col not in df.columns: df[col] = 0.0
        return df

    def _compute_inflow_acceleration(self, df: pd.DataFrame) -> np.ndarray:
        if len(self.flow_buffer) < 2: return np.zeros(len(df))
        accel = np.zeros(len(df))
        for i, row in enumerate(df.itertuples()):
            key = (round(row.lat_bin, 4), round(row.lon_bin, 4))
            hist = [buf.get(key, 0) for buf in self.flow_buffer[-3:]]
            if len(hist) >= 2: accel[i] = (hist[-1] - hist[0]) / len(hist)
        return accel

    def _update_flow_buffer(self, df: pd.DataFrame):
        snap = {(round(r.lat_bin, 4), round(r.lon_bin, 4)): getattr(r, 'flow_pressure', 0) for r in df.itertuples()}
        self.flow_buffer.append(snap)
        if len(self.flow_buffer) > 3: self.flow_buffer = self.flow_buffer[-3:]

    def _zscore_normalize(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        # SENSITIVITY FLOOR: 0.5 prevents the map from emptying when traffic is stable
        curr_m, curr_s = np.mean(values), max(np.std(values), 0.5)
        if feature_name in self.zscore_stats:
            old_m, old_s = self.zscore_stats[feature_name]
            new_m, new_s = 0.9 * old_m + 0.1 * curr_m, max(0.9 * old_s + 0.1 * curr_s, 0.5)
        else:
            new_m, new_s = curr_m, curr_s
        self.zscore_stats[feature_name] = (new_m, new_s)
        return np.clip(((values - new_m) / new_s + 2) / 4, 0, 1)

    def _compute_trend(self, fp: np.ndarray, accel: np.ndarray) -> np.ndarray:
        trend = np.full(len(fp), 'constant', dtype=object)
        trend[(fp > 0.1) & (accel > 0)] = 'rising'
        trend[(fp < -0.05) | (accel < 0)] = 'falling'
        return trend

    def _process_ema(self, df: pd.DataFrame, vals: np.ndarray, state_dict: Dict) -> List[float]:
        res = []
        for i, row in enumerate(df.itertuples()):
            key, val = (round(row.lat_bin, 4), round(row.lon_bin, 4)), vals[i]
            prev = state_dict.get(key, {'val': val})
            prev_v = prev.get('val', val) if isinstance(prev, dict) else float(prev)
            new_v = (EMA_ALPHA * val) + (1 - EMA_ALPHA) * prev_v
            state_dict[key] = {'val': new_v, 'last_iter': self.iteration_count}
            res.append(new_v)
        return res

    def _prune_ema(self):
        for d in [self.ema_state, self.ema_state_current]:
            stale = [k for k, v in d.items() if self.iteration_count - v.get('last_iter', self.iteration_count) > EMA_TTL]
            for k in stale: del d[k]

    def restore_state(self, state: Dict):
        if not state: return
        self.iteration_count = state.get('iteration_count', 0)
        self.zscore_stats = state.get('zscore_stats', {})
        for key in ['ema_state', 'ema_state_current']:
            target = getattr(self, key)
            for k, v in state.get(key, {}).items():
                try:
                    coords = tuple(map(float, k.split(',')))
                    target[coords] = v if isinstance(v, dict) else {'val': float(v), 'last_iter': self.iteration_count}
                except: continue

    def get_state(self) -> Dict:
        stringify = lambda d: {f"{k[0]},{k[1]}": v for k, v in d.items()}
        return {
            'iteration_count': self.iteration_count,
            'zscore_stats': self.zscore_stats,
            'ema_state': stringify(self.ema_state),
            'ema_state_current': stringify(self.ema_state_current)
        }