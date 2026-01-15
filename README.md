# RT Forecast - Real-Time Aviation Congestion Forecasting

A production-grade, fault-tolerant Python service for predicting airspace congestion in the San Francisco Bay Area. Features a ForeFlight-inspired visualization with Sanctum Glassmorphism aesthetics.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## Project Overview

RT Forecast is a real-time aviation congestion prediction system that:

- **Ingests** live ADS-B data from the OpenSky Network API every 5 minutes
- **Predicts** congestion 15 minutes into the future using a Hybrid Advection-Delta ML model
- **Visualizes** results on an interactive Folium map with dual-layer radar display
- **Deploys** seamlessly to cloud platforms (Render, Railway, Docker)

### Visual Aesthetic

The UI combines **ForeFlight's** aviation-grade clarity with **Sanctum.so's** glassmorphism design language:
- Dark mode with NEXRAD-style gradient (Green → Yellow → Red → Magenta)
- Glass-effect panels with backdrop blur
- Responsive HUD with real-time metrics

---

## Architecture & Logic

### Data Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenSky API   │───▶│  Grid Binning   │───▶│  ML Inference   │───▶│ Folium Render   │
│   (ADS-B Data)  │    │  (0.05° cells)  │    │  (Hybrid Model) │    │  (index.html)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │
        ▼                      ▼                      ▼                      ▼
   Aircraft states      Spatial aggregation    Current + Forecast     Interactive map
   (lat, lon, vel,      with flow vectors      congestion scores      with layer toggle
    altitude, track)    (U/V components)       (0.0 - 1.0)            and airport markers
```

### The Hybrid Advection-Delta Model

The forecasting engine combines two complementary approaches:

#### 1. Kinematic Advection (Physics-Based)
Predicts **WHERE** traffic moves using velocity vectors:
- Decomposes aircraft velocity into U (eastward) and V (northward) components
- Calculates flow compression: `(inflow - outflow) / cell_capacity`
- Projects positions forward using flat-earth approximation

#### 2. ML Delta Prediction (Learned Patterns)
Predicts **HOW MUCH** congestion intensity changes:
- RandomForestRegressor trained on historical patterns
- Features: aircraft count, arrival pressure, neighbor flow, temporal signals
- Outputs delta (change) in congestion, not absolute values

#### Why "Spatially Blind"?

The model deliberately **excludes lat/lon coordinates** from features. This prevents:
- Memorizing the map (overfitting to specific locations)
- Geographic bias toward historically busy areas

Instead, it learns **flow dynamics** that generalize across the airspace.

### Feature Engineering

```python
FEATURE_COLS = [
    'aircraft_count',      # Density in cell
    'arrival_pressure',    # Descending aircraft count
    'altitude_std',        # Vertical spread (holding patterns)
    'mean_speed',          # Average velocity
    'avg_u', 'avg_v',      # Flow vector components
    'hour_sin', 'hour_cos', # Temporal encoding (rush hour detection)
    'nb_N_count', 'nb_S_count', 'nb_E_count', 'nb_W_count'  # Neighbor pressure
]
```

---

## Visualization System

### Dual-Layer Display

The map features two toggleable layers with distinct purposes:

| Layer | Purpose | Normalization | Visual Character |
|-------|---------|---------------|------------------|
| **Live Radar** | Current density | Global (absolute) | Sharp, high-contrast |
| **15-min Forecast** | Future prediction | Self-relative | Diffuse, probabilistic |

### Self-Relative Normalization (Key Concept)

The forecast layer uses **layer-local normalization** so predictions are always visible:

```
Forecast intensity = value / max(forecast_values)
```

This means:
- **Red** = Relative peak (highest predicted congestion in this frame)
- **Green** = Relative low (not necessarily zero congestion)

This design prevents forecasts from being invisibly faint when current traffic is heavy.

### NEXRAD-Style Gradient

```python
NEXRAD_GRADIENT = {
    0.0: '#000000',  # Black (empty)
    0.4: '#00FF00',  # Green (nominal)
    0.7: '#FFFF00',  # Yellow (monitor)
    0.9: '#FF0000',  # Red (congested)
    1.0: '#FF00FF'   # Magenta (saturated)
}
```

---

## Installation & Usage

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/rt-forecast.git
cd rt-forecast

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Start the server
python server.py
# or: uvicorn server:app --host 0.0.0.0 --port 8000

# Access the map
open http://localhost:8000
```

### CLI Options (Direct Module)

```bash
# Continuous production loop
python -m src.rt_forecast --serve

# Single iteration (for testing)
python -m src.rt_forecast --once

# Replay from synthetic data
python -m src.rt_forecast --synthetic --serve

# Generate stress test scenario
python -m src.rt_forecast --synthetic-stress --stress-airport KSFO --serve
```

### Docker Deployment

```bash
# Build image
docker build -t rt-forecast .

# Run container
docker run -p 8000:8000 rt-forecast
```

### Cloud Deployment (Render/Railway)

1. Push to GitHub
2. Connect repository to Render/Railway
3. The `Dockerfile` and `requirements.txt` are auto-detected
4. The `$PORT` environment variable is automatically injected

---

## Project Structure

```
rt-forecast/
├── server.py                 # FastAPI web server (entry point)
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── .gitignore               # Git exclusions
├── .dockerignore            # Docker build exclusions
│
├── src/
│   ├── __init__.py          # Package marker
│   ├── acquire_data.py      # [Utility] Process raw CSVs
│   ├── pull_data.py         # [Utility] Download OpenSky data
│   ├── congestion_model.py  # [Utility] Train ML model
│   │
│   └── rt_forecast/         # Core production package
│       ├── __init__.py      # Package exports
│       ├── __main__.py      # CLI entry point
│       ├── config.py        # Global configuration
│       ├── fetch.py         # OpenSky API + dead-reckoning
│       ├── preproc.py       # Grid aggregation
│       ├── infer.py         # ML inference engine
│       ├── state.py         # State persistence
│       ├── ui.py            # Folium map renderer
│       ├── health.py        # Heartbeat & monitoring
│       └── replay.py        # Synthetic data replay
│
├── data/                    # Runtime data (gitignored)
│   ├── state.json           # Persisted system state
│   └── heartbeat.json       # Health monitoring
│
├── models/                  # Trained models (gitignored)
│   └── model_15min.joblib   # RandomForest model
│
├── output/                  # Generated output (gitignored)
│   └── index.html           # Rendered forecast map
│
└── _archive/                # Archived/deprecated files
```

---

## Developer Notes

### Background Worker Thread

The `server.py` runs the forecast loop in a **daemon thread** that:
- Fetches data every 5 minutes (aligned to clock boundaries)
- Regenerates `output/index.html` on each iteration
- Handles API failures gracefully with dead-reckoning
- Retries on errors after 60 seconds

This design ensures the FastAPI server remains responsive while data refreshes in the background.

### Dead-Reckoning Failsafe

When the OpenSky API fails, the system:
1. Extrapolates aircraft positions using last known velocity/heading
2. Applies decay factor (0.92x per iteration) to congestion scores
3. Continues for up to 12 iterations (1 hour) before marking data stale

### Health Monitoring

The `/health` endpoint returns:
- `200 OK` if heartbeat is fresh (<10 minutes)
- `503 Service Unavailable` if stale or errored

Use this for container orchestration health checks.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the forecast map (HTML) |
| `/health` | GET | Health check (`{"status": "ok"}`) |
| `/status` | GET | Detailed status with heartbeat data |

---

## Configuration Reference

Key constants in `src/rt_forecast/config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `WINDOW_MINUTES` | 5 | Forecast update interval |
| `GRID_RES` | 0.05° | Spatial grid resolution (~5.5 km) |
| `DR_MAX_ITER` | 12 | Max dead-reckoning iterations |
| `SIGMOID_CENTER` | 0.33 | Congestion scaling midpoint |
| `EMA_ALPHA` | 0.4 | Temporal smoothing factor |

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- **OpenSky Network** for providing free ADS-B data
- **Folium** for Python mapping capabilities
- **ForeFlight** for UI/UX inspiration
- **Sanctum.so** for glassmorphism design principles
