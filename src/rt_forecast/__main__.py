"""
Main - CLI Entry Point
=============================================
Description: Production forecast loop with strict 5-minute time alignment.
             Supports live mode (OpenSky API), replay mode (Parquet files),
             and synthetic scenario generation. Includes first-run bootstrap
             for immediate feedback and graceful shutdown handling.
Author: RT Forecast Team
Version: 3.2.2

Modes:
    --serve     Continuous production loop (default)
    --once      Single iteration for testing
    --replay    Playback from historical/synthetic data
    --synthetic Generate stress test scenarios
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import pandas as pd
import numpy as np

from .config import (
    WINDOW_MINUTES, LOOP_SLEEP_SEC, MODELS_DIR,
    DATA_DIR, OUTPUT_DIR, FEATURE_HASH, FEATURE_COLS
)
from .fetch import fetch_states, get_dr_count, set_dr_state
from .preproc import process_grid, filter_to_core
from .infer import InferenceEngine, ModelContractError
from .state import save_state, load_state, get_last_processed_window, get_dr_count as get_saved_dr_count
from .ui import render_forecast, compute_airport_scores
from .health import write_heartbeat

# Configure logging to UTC
logging.basicConfig(
    format='%(asctime)s | %(levelname)-5s | %(name)s | %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ',
    level=logging.INFO
)
logging.Formatter.converter = time.gmtime
log = logging.getLogger('rt_forecast')

# Global system state
_running = True
_engine: InferenceEngine = None
_replay_engine = None
_data_source_fn: Optional[Callable] = None
_iteration_count = 0


def get_current_window() -> datetime:
    """Calculate current 5-minute window floor."""
    now = datetime.now(timezone.utc)
    floored_minute = (now.minute // WINDOW_MINUTES) * WINDOW_MINUTES
    return now.replace(minute=floored_minute, second=0, microsecond=0)


def run_iteration(window_time: datetime) -> bool:
    """
    Execute single forecast iteration for a specific window.
    Pipeline: Fetch -> Preproc -> Infer(Vector Flow) -> Render -> Save
    """
    global _engine, _data_source_fn, _replay_engine, _iteration_count
    start_time = time.time()
    _iteration_count += 1
    
    log.info("=" * 50)
    log.info(f"ITERATION | window={window_time.isoformat()}")
    
    # 1. Fetch ADS-B state vectors (live or replay)
    if _data_source_fn:
        states, status = _data_source_fn()
        dr_count = 0
    else:
        states, status = fetch_states()
        dr_count = get_dr_count()
    
    if states is None:
        log.warning(f"Data Source Unavailable (status={status})")
        return False
    
    # 2. Preprocess into spatial grid
    agg_df = process_grid(states, window_time)
    if agg_df.empty:
        log.warning("Empty grid - No aircraft in core BBOX")
        return False
    
    # 3. Vector Flow Inference
    predictions = _engine.predict(agg_df, window_time, status, dr_count)
    
    # 4. Extract metrics for HUD and Heartbeat
    df_cur = predictions.get('current', pd.DataFrame())
    df_15m = predictions.get('15min', pd.DataFrame())
    
    vis_cur = filter_to_core(df_cur)
    vis_15m = filter_to_core(df_15m) if not df_15m.empty else pd.DataFrame()
    
    max_cur = vis_cur['current_congestion'].max() if not vis_cur.empty and 'current_congestion' in vis_cur.columns else 0
    max_15m = vis_15m['predicted_15min'].max() if not vis_15m.empty and 'predicted_15min' in vis_15m.columns else 0
    cell_count = len(vis_cur)
    
    # 5. Render Interactive Map & Airport Intel
    airport_scores = compute_airport_scores(predictions)
    metrics = {
        'window_time': window_time,
        'status': status,
        'aircraft': len(states),
        'cells': cell_count,
        'max_current': max_cur,
        'max_15m': max_15m,
        'latency': time.time() - start_time,
        'ema_cells': len(_engine.ema_state),
        'dr_count': dr_count,
        'iteration': _iteration_count
    }
    
    render_forecast(predictions, states, metrics, airport_scores)
    
    # 6. Persistence & Health Monitoring
    save_state(
        last_processed_window=window_time,
        dr_count=dr_count,
        engine_state=_engine.get_state()
    )
    write_heartbeat(**metrics)
    
    log.info(f"COMPLETE | status={status} | aircraft={len(states)} | cells={cell_count} | "
             f"peak_cur={max_cur:.3f} | peak_15m={max_15m:.3f} | time={metrics['latency']:.2f}s")
    
    return True


def serve_loop():
    """Continuous production loop with First-Run Bootstrap and Replay support."""
    global _running, _replay_engine
    
    # Restore persistence
    state = load_state()
    last_processed = get_last_processed_window(state)
    if state and _engine:
        _engine.restore_state(state)
        set_dr_state(get_saved_dr_count(state))
    
    is_replay = _replay_engine is not None
    
    if is_replay:
        log.info(f"REPLAY MODE | windows={len(_replay_engine.windows)} | speed={_replay_engine.speed_multiplier}x")
        
        while _running:
            try:
                window_time = _replay_engine.get_current_window()
                success = run_iteration(window_time)
                
                if not success and not _replay_engine.loop:
                    log.info("Replay complete (no loop)")
                    break
                
                # Sleep based on replay speed
                if _replay_engine.speed_multiplier >= 100:
                    time.sleep(0.1)
                else:
                    sleep_time = max(0.1, min(300, (WINDOW_MINUTES * 60) / _replay_engine.speed_multiplier))
                    time.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Replay error: {e}")
                time.sleep(1)
    else:
        log.info(f"SERVE MODE | last_window={last_processed}")
        log.info(f"Wake interval: {LOOP_SLEEP_SEC}s")

        first_run_triggered = False

        while _running:
            try:
                current_window = get_current_window()
                
                # BOOTSTRAP: Run immediately on startup
                if not first_run_triggered:
                    log.info("BOOTSTRAP | Executing immediate first-run iteration...")
                    if run_iteration(current_window):
                        last_processed = current_window
                    first_run_triggered = True
                    continue

                # Alignment Logic
                if last_processed and current_window <= last_processed:
                    next_window = current_window + timedelta(minutes=WINDOW_MINUTES)
                    wait_sec = (next_window - datetime.now(timezone.utc)).total_seconds()
                    
                    if int(wait_sec) % 60 == 0:
                        log.info(f"WAIT | Next window in {int(wait_sec)}s ({next_window.strftime('%H:%M')} UTC)")
                    
                    time.sleep(LOOP_SLEEP_SEC)
                    continue
                
                if run_iteration(current_window):
                    last_processed = current_window
                
                time.sleep(LOOP_SLEEP_SEC)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Loop Critical Error: {e}")
                time.sleep(LOOP_SLEEP_SEC)
    
    log.info("Serve loop terminated")


def setup_replay_mode(args) -> bool:
    """Configure replay mode. Returns True if should exit immediately."""
    global _replay_engine, _data_source_fn
    
    from .replay import (
        ReplayEngine,
        generate_heavy_congestion_scenario,
        generate_airport_stress_test,
        generate_multi_airport_rush,
        list_scenarios
    )
    
    scenarios_dir = DATA_DIR / 'scenarios'
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    # List scenarios
    if args.list_scenarios:
        scenarios = list_scenarios(scenarios_dir)
        log.info("=" * 50)
        log.info("AVAILABLE REPLAY SCENARIOS")
        log.info("=" * 50)
        if not scenarios:
            log.info("  No scenarios found. Generate with --synthetic")
        for s in scenarios:
            log.info(f"  {s['name']}: {s.get('stats', {}).get('total_records', '?')} records")
        return True
    
    # Generate synthetic scenarios
    if args.synthetic:
        log.info("Generating heavy congestion scenario...")
        args.replay = str(generate_heavy_congestion_scenario(
            scenarios_dir / 'heavy_congestion.parquet',
            duration_hours=2.0, peak_aircraft=120
        ))
    
    if args.synthetic_stress:
        log.info(f"Generating {args.stress_airport} stress test...")
        args.replay = str(generate_airport_stress_test(
            scenarios_dir / f'{args.stress_airport.lower()}_stress.parquet',
            airport=args.stress_airport, duration_hours=1.0, arrival_rate=50
        ))
    
    if args.synthetic_rush:
        log.info("Generating multi-airport rush scenario...")
        args.replay = str(generate_multi_airport_rush(
            scenarios_dir / 'multi_airport_rush.parquet',
            duration_hours=1.5
        ))
    
    # Setup replay engine
    if args.replay:
        replay_path = Path(args.replay)
        if not replay_path.exists():
            log.error(f"Replay file not found: {replay_path}")
            return True
        
        _replay_engine = ReplayEngine(
            replay_path,
            speed_multiplier=args.replay_speed,
            loop=not args.no_loop
        )
        _data_source_fn = _replay_engine.fetch_states
        
        log.info(f"REPLAY: {replay_path.name} | speed={args.replay_speed}x | loop={not args.no_loop}")
    
    return False


def create_bootstrap_models():
    """Create dummy 40-feature models to allow system startup without training."""
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    import json
    
    num_features = len(FEATURE_COLS)
    log.info(f"BOOTSTRAP: Generating synthetic {num_features}-feature model...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / 'model_15min.joblib'
    meta_path = MODELS_DIR / 'model_15min.meta.json'
    
    if not model_path.exists():
        model = RandomForestRegressor(n_estimators=1, max_depth=2)
        X_dummy = np.random.rand(5, num_features)
        y_dummy = np.random.rand(5)
        model.fit(X_dummy, y_dummy)
        joblib.dump(model, model_path)
    
    if not meta_path.exists():
        meta = {
            'feature_hash': FEATURE_HASH,
            'horizon_minutes': 15,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'bootstrap': True
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


def shutdown_handler(sig, frame):
    global _running
    log.info("Shutdown signal received. Cleaning up...")
    _running = False


def main():
    global _engine, _replay_engine
    
    parser = argparse.ArgumentParser(
        description='RT Forecast v3.2.2 - Vector Flow Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rt_forecast --serve                    # Live production mode
  python -m rt_forecast --synthetic --serve        # Heavy congestion simulation
  python -m rt_forecast --replay FILE --speed 10   # Replay at 10x speed
  python -m rt_forecast --list-scenarios           # Show available scenarios
        """
    )
    
    # Core modes
    parser.add_argument('--once', action='store_true', help='Run single iteration')
    parser.add_argument('--serve', action='store_true', help='Run continuous loop')
    parser.add_argument('--bootstrap', action='store_true', help='Force create dummy models')
    
    # Replay mode
    parser.add_argument('--replay', type=str, metavar='FILE', help='Replay from Parquet file')
    parser.add_argument('--replay-speed', type=float, default=1.0, help='Playback speed (default: 1.0)')
    parser.add_argument('--no-loop', action='store_true', help='Stop when replay ends')
    parser.add_argument('--list-scenarios', action='store_true', help='List available scenarios')
    
    # Synthetic data generation
    parser.add_argument('--synthetic', action='store_true', help='Generate heavy congestion scenario')
    parser.add_argument('--synthetic-stress', action='store_true', help='Generate airport stress test')
    parser.add_argument('--stress-airport', type=str, default='KSFO', choices=['KSFO', 'KOAK', 'KSJC'])
    parser.add_argument('--synthetic-rush', action='store_true', help='Generate multi-airport rush')
    
    args = parser.parse_args()
    
    # Default to --serve
    if not any([args.once, args.serve, args.bootstrap, args.list_scenarios]):
        args.serve = True
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Workspace Setup
    for d in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Setup replay mode if requested
    if args.replay or args.synthetic or args.synthetic_stress or args.synthetic_rush or args.list_scenarios:
        if setup_replay_mode(args):
            return 0
    
    mode_str = "REPLAY" if _replay_engine else "LIVE"
    
    log.info("=" * 50)
    log.info(f"RT FORECAST v3.2.2 - Vector Flow Engine")
    log.info(f"Data Source: {mode_str}")
    log.info(f"Contract Hash: {FEATURE_HASH}")
    log.info("=" * 50)
    
    if args.bootstrap:
        create_bootstrap_models()
        return 0
    
    # Ensure model exists before starting
    if not (MODELS_DIR / 'model_15min.joblib').exists():
        create_bootstrap_models()
    
    _engine = InferenceEngine()
    if not _engine.load_models():
        log.error("Failed to initialize Inference Engine.")
        return 1
    
    if args.once:
        window = _replay_engine.get_current_window() if _replay_engine else get_current_window()
        run_iteration(window)
    else:
        serve_loop()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())