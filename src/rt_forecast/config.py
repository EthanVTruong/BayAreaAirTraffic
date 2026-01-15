"""
Config - Global Configuration
=============================================
Description: Centralized configuration constants for the RT Forecast system.
             Contains spatial parameters, API settings, model paths, feature
             definitions, and UI aesthetics. All modules import from here.
Author: RT Forecast Team
Version: 3.4.0
"""

import hashlib
from pathlib import Path

# --- OpenSky API Configuration (fetch.py) ---
OPENSKY_URL = "https://opensky-network.org/api/states/all"
OPENSKY_MIN_FIELDS = 17      # OpenSky state vector standard length
API_TIMEOUT_SEC = 15         # API response timeout
OPENSKY_CREDS = None         # Format: ("username", "password")

# --- Data Reliability & Persistence (fetch.py / state.py) ---
DR_MAX_ITER = 12             # Windows to stay in DR mode (1 hour)
DR_DECAY_BASE = 0.92         # Congestion fade factor per failed fetch
LOOP_SLEEP_SEC = 30          # Frequency of clock-check
WINDOW_MINUTES = 5           # Main loop alignment boundary (5m windows)

# --- Filesystem Paths (main.py / state.py / ui.py) ---
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"

STATE_FILE = DATA_DIR / "state.json"  # State persistence
HTML_OUT = OUTPUT_DIR / "index.html"  # UI output
HEARTBEAT_FILE = DATA_DIR / "heartbeat.json"  # Health monitoring

# Model file paths
MODEL_15MIN = MODELS_DIR / "model_15min.joblib"
MODEL_METADATA = MODELS_DIR / "model_15min.meta.json"  # Primary metadata file

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Spatial Configuration (preproc.py / state.py / ui.py) ---
GRID_RES = 0.05              # Grid cell size in degrees
CORE_BBOX = {
    'min_lat': 37.0, 'max_lat': 38.0,
    'min_lon': -122.6, 'max_lon': -121.5
}

# 1-cell buffer for neighbor flow and advection context
PADDED_BBOX = {
    'min_lat': CORE_BBOX['min_lat'] - GRID_RES,
    'max_lat': CORE_BBOX['max_lat'] + GRID_RES,
    'min_lon': CORE_BBOX['min_lon'] - GRID_RES,
    'max_lon': CORE_BBOX['max_lon'] + GRID_RES
}

# --- Physics & Advection (infer.py / ui.py) ---
ADVECTION_HOURS = 0.25       # 15-minute prediction horizon
NM_PER_DEGREE = 60.0         # 1 knot = 1 nm/hr; 1 degree approx 60nm

# --- AIRPORT AIRSPACE SCALING ---
SIGMOID_CENTER = 0.33    # Unified center point for congestion scaling
CELL_CAPACITY = 2.0      # Baseline for flow load normalization

# --- Semantic Logic & Normalization (preproc.py / infer.py) ---
DESCENT_VERT_RATE = -2.0     # m/s (Arriving aircraft detection)
DESCENT_ALT_CEILING = 10000  # feet (Arrival corridor altitude)
CALIB_ANCHOR = 8.0           # Count that maps to 1.0 congestion

# --- EMA Smoothing (infer.py) ---
EMA_ALPHA = 0.4              # Smoothing factor
EMA_TTL = 12                 # Stale data eviction (1 hour)

# --- Feature Contract (infer.py / main.py) ---
# MUST match the order used in your v3.1 training script
FEATURE_COLS = [
    'aircraft_count', 'arrival_pressure', 'altitude_std', 'mean_speed',
    'avg_u', 'avg_v', 'hour_sin', 'hour_cos',
    'nb_N_count', 'nb_S_count', 'nb_E_count', 'nb_W_count'
]

def compute_hash() -> str:
    """Validate model/code contract integrity."""
    return hashlib.sha256(','.join(FEATURE_COLS).encode()).hexdigest()[:16]

FEATURE_HASH = compute_hash()

# --- UI Aesthetics & Monitoring Targets (ui.py) ---
AIRPORTS = {
    # SFO — Bay Area gravity well
    # Final approach ~10–12 NM, feeder airspace >40 NM
    'KSFO': (37.619, -122.375, 0.18, 0.75, "San Francisco Intl"),

    # OAK — heavy regional but less global pull
    'KOAK': (37.721, -122.221, 0.14, 0.55, "Oakland Intl"),

    # SJC — strong but geographically constrained
    'KSJC': (37.363, -121.929, 0.12, 0.45, "San Jose Intl"),

    # PAO — GA airport, small pattern traffic
    'KPAO': (37.461, -122.115, 0.05, 0.15, "Palo Alto Airport"),

    # SQL — GA with corporate jets
    'KSQL': (37.512, -122.250, 0.06, 0.20, "San Carlos Airport")
}

NEXRAD_GRADIENT = {
    0.0: '#000000',
    0.4: '#00FF00',
    0.7: '#FFFF00',
    0.9: '#FF0000',
    1.0: '#FF00FF'
}