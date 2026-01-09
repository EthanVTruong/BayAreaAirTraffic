#!/usr/bin/env python3
"""
Bay Area Live Aircraft Congestion Forecast
=============================================================
Production real-time airspace congestion prediction.

Pipeline: fetch_data() -> process_grid() -> apply_inference() -> render_ui()

Features: Calibrated scoring, delta features, dead-reckoning failsafe,
          ATM discrete thresholds, UTC 5-min alignment, bootstrap startup.
"""

import os, sys, json, time, signal, logging, math, warnings
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BBOX = {'lamin': 37.0, 'lamax': 38.0, 'lomin': -122.6, 'lomax': -121.5}
GRID_RES = 0.05
WINDOW_MIN = 5
FEATURE_COLS = ['lat_bin', 'lon_bin', 'aircraft_count', 'arrival_pressure',
                'altitude_std', 'mean_speed', 'lag_1', 'lag_2', 'hour_sin', 'hour_cos']

# Arrival detection
ARRIVAL_ALT = (1500, 10000)  # feet
ARRIVAL_VERT = -2.0  # m/s

# Calibration
CALIB_ANCHOR = 8.0  # aircraft count = 1.0 congestion

# Dead-reckoning
DR_MAX_ITER = 1
DR_DECAY = 0.80

# ATM thresholds
ATM_THRESH = {'nominal': 0.4, 'monitor': 0.7}

# Airports
AIRPORTS = {
    'KSFO': {'lat': 37.6213, 'lon': -122.3790, 'radius': 0.15},
    'KOAK': {'lat': 37.7213, 'lon': -122.2208, 'radius': 0.12},
    'KSJC': {'lat': 37.3639, 'lon': -121.9289, 'radius': 0.12}
}

# Paths
MODEL_PATH = "outputs/model_30min.joblib"
STATE_PATH = "outputs/last_state.json"
OUTPUT_HTML = "outputs/live_forecast.html"
API_URL = "https://opensky-network.org/api/states/all"

# Logging
logging.basicConfig(format='%(asctime)s | %(levelname)-5s | %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%SZ', level=logging.INFO)
logging.Formatter.converter = time.gmtime
log = logging.getLogger('dst')

# =============================================================================
# GLOBAL STATE
# =============================================================================

MODEL = None
LAG_BUFFER = []  # Rolling buffer of last 3 cell snapshots
DR_STATE = {'last_states': None, 'last_time': None, 'count': 0}
PREV_SCORES = {}  # For trend calculation
RUNNING = True
BOOTSTRAP_DONE = False

# =============================================================================
# DATA FETCHING WITH DEAD-RECKONING
# =============================================================================

def fetch_data() -> Tuple[Optional[List[Dict]], str]:
    """Fetch aircraft states with dead-reckoning failsafe. Returns (states, status)."""
    global DR_STATE
    
    try:
        r = requests.get(API_URL, params=BBOX, timeout=25)
        r.raise_for_status()
        data = r.json()
        
        if not data or 'states' not in data or not data['states']:
            return _handle_dr("Empty response")
        
        states = []
        for sv in data.get('states', []):
            if sv[8] or sv[5] is None or sv[6] is None:  # on_ground or no position
                continue
            alt_m = sv[7] if sv[7] is not None else sv[13]
            states.append({
                'icao24': sv[0], 'lat': float(sv[6]), 'lon': float(sv[5]),
                'altitude_ft': (alt_m * 3.28084) if alt_m else 0.0,
                'velocity': sv[9] or 0.0, 'true_track': sv[10] or 0.0,
                'vertrate': sv[11] or 0.0
            })
        
        if not states:
            return _handle_dr("Zero aircraft")
        
        DR_STATE = {'last_states': states, 'last_time': datetime.now(timezone.utc), 'count': 0}
        log.info(f"API OK | aircraft={len(states)}")
        return states, 'OK'
        
    except Exception as e:
        return _handle_dr(str(e)[:40])

def _handle_dr(reason: str) -> Tuple[Optional[List[Dict]], str]:
    """Handle API failure with position extrapolation."""
    global DR_STATE
    log.warning(f"API fail: {reason}")
    
    if DR_STATE['last_states'] and DR_STATE['count'] < DR_MAX_ITER:
        DR_STATE['count'] += 1
        elapsed = min((datetime.now(timezone.utc) - DR_STATE['last_time']).total_seconds(), 300)
        
        extrapolated = []
        for s in DR_STATE['last_states']:
            ns = s.copy()
            if s['velocity'] > 0:
                track_rad = math.radians(s['true_track'])
                dx = s['velocity'] * elapsed * math.sin(track_rad)
                dy = s['velocity'] * elapsed * math.cos(track_rad)
                ns['lat'] = s['lat'] + dy / 111320
                ns['lon'] = s['lon'] + dx / (111320 * math.cos(math.radians(s['lat'])))
            if BBOX['lamin'] <= ns['lat'] <= BBOX['lamax'] and BBOX['lomin'] <= ns['lon'] <= BBOX['lomax']:
                extrapolated.append(ns)
        
        log.warning(f"DEAD-RECKONING | iter={DR_STATE['count']} | ac={len(extrapolated)}")
        return extrapolated, 'DR'
    
    log.error("DATA_LOST: DR exhausted")
    return None, 'FAILED'

# =============================================================================
# GRID PROCESSING
# =============================================================================

def process_grid(states: List[Dict], window_time: datetime) -> pd.DataFrame:
    """Bin aircraft to grid cells and aggregate metrics."""
    if not states:
        return pd.DataFrame()
    
    df = pd.DataFrame(states)
    
    # Deterministic binning
    df['lat_bin'] = np.round(np.floor((df['lat'] - BBOX['lamin']) / GRID_RES) * GRID_RES + BBOX['lamin'], 4)
    df['lon_bin'] = np.round(np.floor((df['lon'] - BBOX['lomin']) / GRID_RES) * GRID_RES + BBOX['lomin'], 4)
    
    # Filter to bounds
    mask = (df['lat_bin'] >= BBOX['lamin']) & (df['lat_bin'] < BBOX['lamax']) & \
           (df['lon_bin'] >= BBOX['lomin']) & (df['lon_bin'] < BBOX['lomax'])
    df = df[mask]
    
    if df.empty:
        return pd.DataFrame()
    
    # Detect arrivals
    is_arr = df['altitude_ft'].between(*ARRIVAL_ALT) & (df['vertrate'] < ARRIVAL_VERT)
    df['arr_icao'] = np.where(is_arr, df['icao24'], np.nan)
    
    # Aggregate
    agg = df.groupby(['lat_bin', 'lon_bin']).agg({
        'icao24': 'nunique', 'arr_icao': 'nunique',
        'velocity': 'mean', 'altitude_ft': 'std'
    }).reset_index()
    agg.columns = ['lat_bin', 'lon_bin', 'aircraft_count', 'arrival_pressure', 'mean_speed', 'altitude_std']
    agg = agg.fillna(0)
    
    return agg

# =============================================================================
# INFERENCE WITH CALIBRATION
# =============================================================================

def apply_inference(agg_df: pd.DataFrame, window_time: datetime, api_status: str) -> pd.DataFrame:
    """Engineer features and generate calibrated predictions."""
    global LAG_BUFFER, MODEL
    
    if agg_df.empty or MODEL is None:
        return agg_df
    
    df = agg_df.copy()
    
    # Hour encoding
    hour = window_time.hour + window_time.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Lag features and deltas
    prev = LAG_BUFFER[-1] if LAG_BUFFER else {}
    prev2 = LAG_BUFFER[-2] if len(LAG_BUFFER) >= 2 else {}
    
    def get_lags(row):
        key = (round(row['lat_bin'], 4), round(row['lon_bin'], 4))
        l1 = prev.get(key, {}).get('count', 0)
        l2 = prev2.get(key, {}).get('count', 0)
        return l1, l2, row['aircraft_count'] - l1, row['arrival_pressure'] - prev.get(key, {}).get('pressure', 0)
    
    lags = df.apply(get_lags, axis=1)
    df['lag_1'] = lags.apply(lambda x: x[0])
    df['lag_2'] = lags.apply(lambda x: x[1])
    df['count_delta'] = lags.apply(lambda x: x[2])
    df['pressure_delta'] = lags.apply(lambda x: x[3])
    
    # Update buffer
    cell_snap = {(round(r['lat_bin'], 4), round(r['lon_bin'], 4)): 
                 {'count': r['aircraft_count'], 'pressure': r['arrival_pressure']} 
                 for _, r in df.iterrows()}
    LAG_BUFFER.append(cell_snap)
    if len(LAG_BUFFER) > 3:
        LAG_BUFFER = LAG_BUFFER[-3:]
    
    # Predict
    X = df[FEATURE_COLS].values
    raw = MODEL.predict(X)
    
    # Calibrate: sigmoid scaling with density anchor
    density = df['aircraft_count'].values / CALIB_ANCHOR
    combined = 0.6 * raw + 0.4 * np.clip(density, 0, 1)
    calibrated = 1 / (1 + np.exp(-6 * (combined - 0.4)))
    
    # DR confidence decay
    if api_status == 'DR':
        calibrated *= DR_DECAY
    
    df['predicted_congestion'] = np.clip(calibrated, 0, 1)
    return df

# =============================================================================
# UI RENDERING
# =============================================================================

def render_ui(pred_df: pd.DataFrame, metrics: Dict, states: List[Dict]) -> None:
    """Generate ForeFlight-style heatmap with HUD and airport intel."""
    global PREV_SCORES
    
    m = folium.Map(location=[37.5, -122.1], zoom_start=9, tiles='CartoDB dark_matter')
    
    # Heatmap
    if not pred_df.empty:
        heat_data = [[r['lat_bin'] + GRID_RES/2, r['lon_bin'] + GRID_RES/2, r['predicted_congestion']] 
                     for _, r in pred_df.iterrows() if r['predicted_congestion'] >= 0.05]
        if heat_data:
            HeatMap(heat_data, min_opacity=0.2, max_opacity=0.9, radius=28, blur=18,
                    gradient={0.0: 'rgba(0,0,0,0)', 0.39: 'rgba(0,200,100,0.2)',
                              0.40: 'rgba(255,200,0,0.5)', 0.69: 'rgba(255,140,0,0.7)',
                              0.70: 'rgba(220,50,0,0.8)', 1.0: 'rgba(150,0,0,1.0)'}).add_to(m)
    
    # Airport markers with intel
    for icao, info in AIRPORTS.items():
        # Compute sector intel
        if states:
            demand = sum(1 for s in states if abs(s['lat']-info['lat']) <= info['radius'] 
                        and abs(s['lon']-info['lon']) <= info['radius'])
        else:
            demand = 0
        
        if not pred_df.empty:
            sector = pred_df[(abs(pred_df['lat_bin']-info['lat']) <= info['radius']) & 
                            (abs(pred_df['lon_bin']-info['lon']) <= info['radius'])]
            score = sector['predicted_congestion'].mean() if not sector.empty else 0
        else:
            score = 0
        
        # Trend
        prev = PREV_SCORES.get(icao, score)
        delta = score - prev
        PREV_SCORES[icao] = score
        trend = 'RISING' if delta > 0.05 else 'FALLING' if delta < -0.05 else 'STABLE'
        trend_color = '#FF6666' if trend == 'RISING' else '#66FF66' if trend == 'FALLING' else '#AAA'
        
        # Status
        if score >= ATM_THRESH['monitor']:
            status, sc = ('FLOW CONTROL', '#FF4444') if score >= 0.7 else ('MONITOR', '#FFB800')
        else:
            status, sc = 'NOMINAL', '#00CC66'
        
        tooltip = f"""<div style="font-family:monospace;font-size:11px">
            <b style="color:#00BFFF">{icao}</b><br>
            Forecast: <b style="color:{sc}">{score:.2f}</b><br>
            Trend: <span style="color:{trend_color}">{trend}</span><br>
            Demand: <b>{demand} AC</b><br>
            <span style="color:{sc}">{status}</span></div>"""
        
        folium.Marker([info['lat'], info['lon']], 
                      icon=folium.DivIcon(html=f'<div style="color:#00BFFF;font-weight:bold;text-shadow:0 0 3px black">* {icao}</div>'),
                      tooltip=folium.Tooltip(tooltip, sticky=True)).add_to(m)
    
    # HUD
    ts = metrics['timestamp'].strftime('%Y-%m-%d %H:%M UTC')
    api_c = '#00CC66' if metrics['api'] == 'OK' else '#FFB800' if metrics['api'] == 'DR' else '#FF4444'
    buf_c = '#00CC66' if len(LAG_BUFFER) >= 2 else '#FFB800' if LAG_BUFFER else '#FF6666'
    buf_s = 'HOT' if len(LAG_BUFFER) >= 2 else 'WARM' if LAG_BUFFER else 'COLD'
    max_c = metrics['max_cong']
    cong_c = '#FF4444' if max_c >= 0.7 else '#FFB800' if max_c >= 0.4 else '#00CC66'
    
    hud = f'''<div style="position:fixed;top:10px;right:10px;z-index:9999;background:rgba(0,0,0,0.9);
        padding:12px;border-radius:8px;font-family:monospace;font-size:11px;color:#FFF;min-width:220px;border:1px solid #333">
        <div style="font-size:13px;font-weight:bold;color:#00BFFF;margin-bottom:10px">BAY AREA DST v2.2</div>
        <div>API: <b style="color:{api_c}">{metrics['api']}</b></div>
        <div>Buffer: <b style="color:{buf_c}">{buf_s}</b></div>
        <div style="border-top:1px solid #444;margin:8px 0;padding-top:8px">
        <div>Time: {ts}</div>
        <div>Aircraft: <b>{metrics['ac']}</b></div>
        <div>Cells: <b>{metrics['cells']}</b></div>
        <div>Max Cong: <b style="color:{cong_c}">{max_c:.3f}</b></div>
        <div>Latency: {metrics['latency']:.2f}s</div></div>
        <div style="font-size:9px;color:#555;margin-top:8px;text-align:center">30-min Forecast | {GRID_RES} deg</div></div>'''
    
    legend = '''<div style="position:fixed;bottom:30px;right:10px;z-index:9998;background:rgba(0,0,0,0.85);
        padding:10px;border-radius:6px;font-family:monospace;font-size:10px;color:#FFF;border:1px solid #333">
        <div style="font-weight:bold;color:#888;margin-bottom:6px">CONGESTION</div>
        <div><span style="display:inline-block;width:16px;height:10px;background:rgba(0,200,100,0.3);margin-right:6px"></span>0.0-0.4 NOMINAL</div>
        <div><span style="display:inline-block;width:16px;height:10px;background:rgba(255,180,0,0.7);margin-right:6px"></span>0.4-0.7 MONITOR</div>
        <div><span style="display:inline-block;width:16px;height:10px;background:rgba(200,0,0,0.9);margin-right:6px"></span>0.7-1.0 FLOW CTRL</div></div>'''
    
    m.get_root().html.add_child(folium.Element(hud + legend))
    
    # Save with UTF-8
    os.makedirs(os.path.dirname(OUTPUT_HTML) or '.', exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(m._repr_html_())
    log.info(f"Map saved: {OUTPUT_HTML}")

# =============================================================================
# STATE PERSISTENCE
# =============================================================================

def save_state():
    """Atomic save of lag buffer for warm restart."""
    if not LAG_BUFFER:
        return
    try:
        state = {'cells': [{'lat': k[0], 'lon': k[1], **v} for k, v in LAG_BUFFER[-1].items()],
                 'timestamp': datetime.now(timezone.utc).isoformat(), 'version': '2.2.0'}
        tmp = STATE_PATH + '.tmp'
        os.makedirs(os.path.dirname(STATE_PATH) or '.', exist_ok=True)
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)
    except Exception as e:
        log.warning(f"State save failed: {e}")

def load_state():
    """Load previous state for warm start."""
    global LAG_BUFFER
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('cells'):
                LAG_BUFFER.append({(round(c['lat'], 4), round(c['lon'], 4)): 
                                   {'count': c.get('count', 0), 'pressure': c.get('pressure', 0)} 
                                   for c in data['cells']})
                log.info(f"State loaded: {len(LAG_BUFFER[-1])} cells")
        except Exception as e:
            log.warning(f"State load failed: {e}")

# =============================================================================
# MAIN LOOP
# =============================================================================

def get_window_time() -> datetime:
    """Get UTC timestamp aligned to 5-minute floor."""
    now = datetime.now(timezone.utc)
    return now.replace(minute=(now.minute // WINDOW_MIN) * WINDOW_MIN, second=0, microsecond=0)

def wait_next_window():
    """Sleep until next 5-minute boundary."""
    now = datetime.now(timezone.utc)
    next_min = ((now.minute // WINDOW_MIN) + 1) * WINDOW_MIN
    if next_min >= 60:
        target = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        target = now.replace(minute=next_min, second=0, microsecond=0)
    sleep_sec = (target - now).total_seconds()
    if sleep_sec > 0:
        log.info(f"Waiting {sleep_sec:.0f}s for {target.strftime('%H:%M')} UTC")
        time.sleep(sleep_sec)

def shutdown_handler(sig, frame):
    """Graceful shutdown with state save."""
    global RUNNING
    log.info("Shutdown signal - saving state...")
    RUNNING = False
    save_state()
    log.info("Shutdown complete.")
    sys.exit(0)

def run_iteration() -> Dict:
    """Execute single forecast iteration."""
    start = time.time()
    window_time = get_window_time()
    log.info(f"{'='*50}")
    log.info(f"ITERATION | window={window_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    
    # Pipeline
    states, api_status = fetch_data()
    
    if states is None:
        metrics = {'timestamp': window_time, 'api': api_status, 'ac': 0, 'cells': 0, 
                   'max_cong': 0, 'latency': time.time() - start}
        render_ui(pd.DataFrame(), metrics, [])
        log.info(f"API: {api_status} | AC: 0 | Cells: 0 | Latency: {metrics['latency']:.2f}s")
        return metrics
    
    agg_df = process_grid(states, window_time)
    pred_df = apply_inference(agg_df, window_time, api_status)
    
    max_cong = pred_df['predicted_congestion'].max() if not pred_df.empty else 0
    metrics = {'timestamp': window_time, 'api': api_status, 'ac': len(states), 
               'cells': len(agg_df), 'max_cong': max_cong, 'latency': time.time() - start}
    
    render_ui(pred_df, metrics, states)
    save_state()
    
    buf = 'HOT' if len(LAG_BUFFER) >= 2 else 'WARM' if LAG_BUFFER else 'COLD'
    log.info(f"API: {api_status} | AC: {len(states)} | Cells: {len(agg_df)} | "
             f"Buffer: {buf} | MaxCong: {max_cong:.3f} | Latency: {metrics['latency']:.2f}s")
    return metrics

def main():
    """Entry point with bootstrap and time-aligned loop."""
    global MODEL, RUNNING, BOOTSTRAP_DONE
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    log.info("=" * 50)
    log.info("BAY AREA DST v2.2 - OPERATIONAL")
    log.info(f"Model: {MODEL_PATH} | Output: {OUTPUT_HTML}")
    log.info("=" * 50)
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    MODEL = joblib.load(MODEL_PATH)
    log.info("Model loaded")
    
    # Load state for warm start
    load_state()
    
    while RUNNING:
        try:
            if BOOTSTRAP_DONE:
                wait_next_window()
            else:
                log.info("BOOTSTRAP: Immediate initial forecast...")
            
            run_iteration()
            
            if not BOOTSTRAP_DONE:
                BOOTSTRAP_DONE = True
                log.info("BOOTSTRAP COMPLETE: Now aligned to UTC windows")
                
        except Exception as e:
            log.error(f"Error: {e}")
    
    log.info("Engine stopped.")

if __name__ == "__main__":
    main()