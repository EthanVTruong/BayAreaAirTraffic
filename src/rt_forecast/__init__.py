"""
RT Forecast - Real-Time Airspace Congestion Forecasting
=============================================
Description: Production-grade, fault-tolerant Python service for Bay Area
             airspace congestion prediction. Uses Hybrid Advection-Delta
             architecture combining physics-based flow modeling with ML.
Author: RT Forecast Team
Version: 3.5.0

Modules:
    config  - Global configuration constants
    fetch   - OpenSky API fetcher with dead-reckoning failsafe
    preproc - Grid preprocessing with vector flow extraction
    infer   - ML inference engine (Hybrid Advection-Delta)
    state   - Atomic state persistence for warm restarts
    ui      - Folium map renderer (ForeFlight aesthetic)
    health  - Heartbeat and observability
    replay  - Sample data replay for testing

Usage:
    python -m rt_forecast --serve  # Continuous production loop
    python -m rt_forecast --once   # Single iteration
"""

__version__ = '3.5.0'
__author__ = 'RT Forecast Team'

from .config import FEATURE_HASH, FEATURE_COLS
from .fetch import fetch_states, get_dr_count
from .preproc import process_grid, filter_to_core
from .infer import InferenceEngine, ModelContractError
from .state import save_state, load_state
from .ui import render_forecast, compute_airport_scores
from .health import write_heartbeat, check_health

__all__ = [
    'fetch_states',
    'get_dr_count',
    'process_grid',
    'filter_to_core',
    'InferenceEngine',
    'ModelContractError',
    'save_state',
    'load_state',
    'render_forecast',
    'compute_airport_scores',
    'write_heartbeat',
    'check_health',
    'FEATURE_HASH',
    'FEATURE_COLS'
]