"""
Server - FastAPI Web Server
=============================================
Description: Production web server that runs the forecast loop in a background
             thread and serves the generated map visualization at the root URL.
             Designed for deployment on Render, Railway, or similar platforms.
Author: RT Forecast Team
Version: 1.0.0

Endpoints:
    GET /         - Serves the forecast map (index.html)
    GET /health   - Health check for container orchestration
    GET /status   - Detailed status with heartbeat data

Usage:
    python -m uvicorn server:app
"""

import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

# Configure logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)-5s | %(name)s | %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ',
    level=logging.INFO
)
logging.Formatter.converter = time.gmtime
log = logging.getLogger('server')

# Import config to get HTML_OUT path
from src.rt_forecast.config import HTML_OUT

app = FastAPI(
    title="RT Forecast",
    description="Bay Area Airspace Congestion Forecasting",
    version="3.2.0"
)

# Background thread state
_forecast_thread: threading.Thread = None
_running = True


def run_forecast_loop():
    """
    Background thread that runs the forecast loop.
    Handles errors gracefully and retries after 60 seconds.
    """
    global _running

    # Import here to avoid circular imports and ensure proper initialization
    from src.rt_forecast.__main__ import (
        serve_loop,
        InferenceEngine,
        create_bootstrap_models,
        get_current_window,
        run_iteration
    )
    from src.rt_forecast.config import MODELS_DIR, DATA_DIR, OUTPUT_DIR
    from src.rt_forecast.state import load_state, get_last_processed_window
    from src.rt_forecast.fetch import set_dr_state
    from src.rt_forecast.state import get_dr_count as get_saved_dr_count

    log.info("Forecast background thread starting...")

    while _running:
        try:
            # Ensure directories exist
            for d in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
                d.mkdir(parents=True, exist_ok=True)

            # Ensure model exists
            if not (MODELS_DIR / 'model_15min.joblib').exists():
                log.info("Creating bootstrap models...")
                create_bootstrap_models()

            # Initialize inference engine
            engine = InferenceEngine()
            if not engine.load_models():
                log.error("Failed to load models. Retrying in 60s...")
                time.sleep(60)
                continue

            # Restore state
            state = load_state()
            last_processed = get_last_processed_window(state)
            if state and engine:
                engine.restore_state(state)
                set_dr_state(get_saved_dr_count(state))

            log.info(f"Forecast loop initialized. Last window: {last_processed}")

            # Import and set the global engine in __main__
            import src.rt_forecast.__main__ as main_module
            main_module._engine = engine
            main_module._running = True

            # Run the serve loop (this blocks until shutdown or error)
            serve_loop()

        except Exception as e:
            log.error(f"Forecast loop error: {e}. Retrying in 60s...")
            time.sleep(60)

    log.info("Forecast background thread stopped")


@app.on_event("startup")
async def startup_event():
    """Start the forecast loop in a background thread on server startup."""
    global _forecast_thread

    log.info("Starting forecast background thread...")
    _forecast_thread = threading.Thread(target=run_forecast_loop, daemon=True)
    _forecast_thread.start()
    log.info("Server ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Signal the background thread to stop on server shutdown."""
    global _running
    log.info("Shutting down forecast thread...")
    _running = False

    # Also signal the inner loop to stop
    try:
        import src.rt_forecast.__main__ as main_module
        main_module._running = False
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
async def serve_map():
    """Serve the generated forecast map HTML."""
    if HTML_OUT.exists():
        return HTMLResponse(content=HTML_OUT.read_text(encoding='utf-8'))
    else:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html>
            <head><title>RT Forecast - Loading</title></head>
            <body style="background:#1a1a2e;color:white;font-family:sans-serif;
                         display:flex;justify-content:center;align-items:center;
                         height:100vh;margin:0;">
                <div style="text-align:center;">
                    <h1>RT Forecast</h1>
                    <p>Initializing forecast engine...</p>
                    <p style="color:#888;">The map will appear shortly. Refresh in 30 seconds.</p>
                </div>
            </body>
            </html>
            """,
            status_code=200
        )


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    from src.rt_forecast.health import check_health

    health = check_health()
    status_code = 200 if health.get('healthy', False) else 503

    return Response(
        content='{"status": "ok"}' if health.get('healthy') else '{"status": "degraded"}',
        media_type="application/json",
        status_code=status_code
    )


@app.get("/status")
async def detailed_status():
    """Detailed status endpoint with heartbeat data."""
    from src.rt_forecast.health import check_health
    from src.rt_forecast.config import HEARTBEAT_FILE
    import json

    health = check_health()

    status = {
        "healthy": health.get("healthy", False),
        "reason": health.get("reason"),
        "forecast_status": health.get("status"),
        "iteration": health.get("iter"),
        "dr_count": health.get("dr"),
        "map_exists": HTML_OUT.exists()
    }

    # Include heartbeat if available
    if HEARTBEAT_FILE.exists():
        try:
            with open(HEARTBEAT_FILE) as f:
                status["heartbeat"] = json.load(f)
        except Exception:
            pass

    return status
