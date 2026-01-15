"""
Health - Observability & Heartbeat
=============================================
Description: Writes structured heartbeat JSON for external monitoring systems.
             Provides health check endpoints for container orchestration and
             staleness detection (unhealthy if >10 minutes since last update).
Author: RT Forecast Team
Version: 3.1.0
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from .config import HEARTBEAT_FILE

log = logging.getLogger('rt_forecast.health')


def write_heartbeat(
    window_time: datetime,
    status: str,
    aircraft: int,
    cells: int,
    max_current: float,
    max_15m: float,
    latency: float,
    iteration: int,
    ema_cells: int,
    dr_count: int,
    error: Optional[str] = None
) -> bool:
    """Write structured heartbeat JSON for external monitoring."""
    try:
        hb = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'window': window_time.isoformat(),
            'status': status,
            'aircraft': aircraft,
            'cells': cells,
            'max_current': round(max_current, 4),
            'max_15m': round(max_15m, 4),
            'latency': round(latency, 3),
            'iter': iteration,
            'ema': ema_cells,
            'dr': dr_count,
            'error': error,
            'v': '3.1.0'
        }
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HEARTBEAT_FILE, 'w', encoding='utf-8') as f:
            json.dump(hb, f, indent=2)
        return True
    except Exception as e:
        log.error(f"Heartbeat failed: {e}")
        return False


def check_health() -> Dict:
    """Quick health check from last heartbeat (unhealthy if >10min stale)."""
    if not HEARTBEAT_FILE.exists():
        return {'healthy': False, 'reason': 'No heartbeat'}
    
    try:
        with open(HEARTBEAT_FILE, 'r') as f:
            hb = json.load(f)
        
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(hb['ts'])).total_seconds()
        healthy = age < 600
        
        return {
            'healthy': healthy,
            'reason': None if healthy else f'Stale ({age:.0f}s)',
            'status': hb.get('status'),
            'iter': hb.get('iter'),
            'dr': hb.get('dr')
        }
    except Exception as e:
        return {'healthy': False, 'reason': str(e)}