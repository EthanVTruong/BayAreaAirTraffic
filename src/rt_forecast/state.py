"""
State - Persistence Manager
=============================================
Description: Atomic state persistence for system recovery and warm restarts.
             Handles serialization of coordinate tuples, EMA caches, and
             temporal fields. Supports Vector Flow components (U, V) for
             advection-based forecasting continuity.
Author: RT Forecast Team
Version: 3.2.0
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

from .config import STATE_FILE, GRID_RES

log = logging.getLogger('rt_forecast.state')

# State schema version for future migrations
STATE_SCHEMA_VERSION = '1.1.0'

def _make_serializable(obj):
    """
    Deep-converts coordinate tuples to strings for JSON safety.
    Handles nested dictionaries and lists recursively.
    """
    if isinstance(obj, dict):
        # Convert {(37.5, -122): val} -> {"37.5,-122": val}
        return {
            f"{k[0]},{k[1]}" if isinstance(k, tuple) else k: _make_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    return obj

def save_state(
    last_processed_window: datetime,
    dr_count: int,
    engine_state: Dict[str, Any],
    extra: Optional[Dict] = None
) -> bool:
    """
    Atomic save of system state including Vector Flow snapshots.
    Deep-converts all coordinate tuples to strings for JSON compatibility.
    """
    try:
        state = {
            'schema_version': STATE_SCHEMA_VERSION,
            'last_processed_window': last_processed_window.isoformat(),
            'dr_count': dr_count,
            **engine_state,
            'saved_at': datetime.now(timezone.utc).isoformat()
        }

        if extra:
            state.update(extra)

        # Deep serialize to handle all nested tuple keys
        clean_state = _make_serializable(state)

        # Atomic write: write to .tmp then rename to prevent corruption during crashes
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = STATE_FILE.with_suffix('.tmp')

        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(clean_state, f, indent=2)

        tmp_path.replace(STATE_FILE)
        log.debug(f"State saved: {STATE_FILE}")
        return True

    except Exception as e:
        log.error(f"State save failed: {e}")
        return False

def _restore_tuples(obj):
    """
    Restores coordinate strings back into Python tuples.
    Handles nested dictionaries recursively.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, str) and "," in k:
                try:
                    # Convert "37.5,-122" -> (37.5, -122.0)
                    coords = tuple(map(float, k.split(',')))
                    new_dict[coords] = _restore_tuples(v)
                    continue
                except (ValueError, TypeError):
                    pass
            new_dict[k] = _restore_tuples(v)
        return new_dict
    elif isinstance(obj, list):
        return [_restore_tuples(i) for i in obj]
    return obj

def load_state() -> Optional[Dict]:
    """
    Load system state from disk and parse temporal fields.
    Restores all coordinate strings back to tuple keys.
    """
    if not STATE_FILE.exists():
        log.info("No existing state file found. Starting fresh.")
        return None

    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Restore coordinate tuples from strings
        state = _restore_tuples(raw)

        # Parse datetime fields with timezone awareness
        if 'last_processed_window' in state:
            state['last_processed_window'] = datetime.fromisoformat(state['last_processed_window'])
            if state['last_processed_window'].tzinfo is None:
                state['last_processed_window'] = state['last_processed_window'].replace(tzinfo=timezone.utc)

        if 'saved_at' in state:
            state['saved_at'] = datetime.fromisoformat(state['saved_at'])
            if state['saved_at'].tzinfo is None:
                state['saved_at'] = state['saved_at'].replace(tzinfo=timezone.utc)

        log.info(f"State loaded: version={state.get('schema_version')}, "
                 f"last_window={state.get('last_processed_window')}")

        return state

    except Exception as e:
        log.error(f"State load failed: {e}")
        return None

def get_neighbor_snapshot(state: Optional[Dict], lat: float, lon: float, direction: str) -> Dict[str, float]:
    """
    Retrieves the count, u, and v of a neighbor from the PREVIOUS window (T-1).
    Used to calculate the advection of congestion between grid cells.
    """
    if state is None or 'lag_buffer' not in state or not state['lag_buffer']:
        return {'count': 0.0, 'u': 0.0, 'v': 0.0}
    
    # Use the most recent snapshot in the buffer (T-1)
    # The lag_buffer is expected to be a list of dicts where keys are "lat,lon" strings
    prev_snap = state['lag_buffer'][-1]
    
    # Calculate neighbor coordinates based on the canonical grid resolution
    # 
    d_lat, d_lon = 0.0, 0.0
    if 'N' in direction: d_lat = GRID_RES
    if 'S' in direction: d_lat = -GRID_RES
    if 'E' in direction: d_lon = GRID_RES
    if 'W' in direction: d_lon = -GRID_RES
    
    nb_key = f"{lat + d_lat:.4f},{lon + d_lon:.4f}"
    
    # Returns the physical flow components (u, v) and aircraft density
    return prev_snap.get(nb_key, {'count': 0.0, 'u': 0.0, 'v': 0.0})

def prune_stale_ema(state: Dict, ttl_minutes: int = 15) -> Dict:
    """
    Evict EMA entries if the state file itself is older than the TTL.
    Ensures that "ghost" radar blobs don't persist after long downtimes.
    """
    if 'saved_at' not in state:
        return state
    
    now = datetime.now(timezone.utc)
    age = (now - state['saved_at']).total_seconds() / 60
    
    if age > ttl_minutes:
        log.warning(f"State TTL reached ({age:.1f}m > {ttl_minutes}m). Clearing EMA caches.")
        state['ema_state'] = {}
        state['ema_state_current'] = {}
    
    return state

# --- Helper Accessors ---

def get_last_processed_window(state: Optional[Dict]) -> Optional[datetime]:
    return state.get('last_processed_window') if state else None

def get_dr_count(state: Optional[Dict]) -> int:
    return state.get('dr_count', 0) if state else 0