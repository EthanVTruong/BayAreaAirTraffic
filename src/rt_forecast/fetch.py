"""
Fetch - OpenSky API Data Fetcher
=============================================
Description: Fetches live aircraft states from the OpenSky Network API with
             robust error handling. Implements dead-reckoning failsafe for
             API failures, schema guard for drift protection, and flat-earth
             position extrapolation for graceful degradation.
Author: RT Forecast Team
Version: 3.2.0
"""

import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from .config import (
    OPENSKY_URL, OPENSKY_MIN_FIELDS, API_TIMEOUT_SEC,
    PADDED_BBOX, DR_MAX_ITER
)

log = logging.getLogger('rt_forecast.fetch')

# Module-level DR state
_dr_state: Dict = {
    'last_states': None,
    'last_time': None,
    'count': 0
}


def fetch_states() -> Tuple[Optional[List[Dict]], str]:
    """
    Fetch aircraft states from OpenSky API with failsafe dead-reckoning.
    
    Returns:
        Tuple of (states_list, status) where status is:
        - "LIVE": Fresh data from API
        - "DR": Dead-reckoned from last known state
        - "EMPTY": No data available (DR exhausted or no prior state)
    """
    global _dr_state
    
    try:
        params = {
            'lamin': PADDED_BBOX['min_lat'],
            'lamax': PADDED_BBOX['max_lat'],
            'lomin': PADDED_BBOX['min_lon'],
            'lomax': PADDED_BBOX['max_lon']
        }
        
        resp = requests.get(OPENSKY_URL, params=params, timeout=API_TIMEOUT_SEC)
        
        # Handle rate limiting
        if resp.status_code == 429:
            log.warning("API rate limited (429)")
            return _handle_dr("Rate limited")
        
        resp.raise_for_status()
        data = resp.json()
        
        if not data or 'states' not in data or not data['states']:
            return _handle_dr("Empty API response")
        
        states = _parse_states(data['states'])
        
        if not states:
            return _handle_dr("Zero valid aircraft after parsing")
        
        # Success - reset DR state
        _dr_state = {
            'last_states': states,
            'last_time': datetime.now(timezone.utc),
            'count': 0
        }
        
        log.info(f"LIVE | aircraft={len(states)}")
        return states, 'LIVE'
        
    except requests.exceptions.Timeout:
        return _handle_dr("API timeout")
    except requests.exceptions.RequestException as e:
        return _handle_dr(f"Request error: {str(e)[:30]}")
    except Exception as e:
        return _handle_dr(f"Unexpected error: {str(e)[:30]}")


def _parse_states(raw_states: List) -> List[Dict]:
    """
    Parse raw OpenSky state vectors with schema guard.
    
    Schema guard: Skip any row with fewer than OPENSKY_MIN_FIELDS to protect
    against API schema drift.
    """
    states = []
    schema_errors = 0
    
    for sv in raw_states:
        # SCHEMA GUARD: Protect against API drift
        if len(sv) < OPENSKY_MIN_FIELDS:
            schema_errors += 1
            continue
        
        # Skip grounded or position-less aircraft
        on_ground = sv[8]
        lon = sv[5]
        lat = sv[6]
        
        if on_ground or lon is None or lat is None:
            continue
        
        # Extract altitude (baro or geo)
        alt_m = sv[7] if sv[7] is not None else sv[13]
        
        states.append({
            'icao24': sv[0],
            'lat': float(lat),
            'lon': float(lon),
            'altitude_ft': (alt_m * 3.28084) if alt_m else 0.0,
            'velocity': sv[9] or 0.0,
            'true_track': sv[10] or 0.0,
            'vert_rate': sv[11] or 0.0
        })
    
    if schema_errors > 0:
        log.warning(f"SCHEMA GUARD: Skipped {schema_errors} malformed vectors")
    
    return states


def _handle_dr(reason: str) -> Tuple[Optional[List[Dict]], str]:
    """
    Handle API failure with flat-earth position extrapolation.
    
    Dead-Reckoning Math (Flat-Earth Approximation):
        Δlat = (velocity * time * cos(track)) / 111320
        Δlon = (velocity * time * sin(track)) / (111320 * cos(lat))
    
    Constraint: Max DR_MAX_ITER iterations before returning EMPTY.
    """
    global _dr_state
    
    log.warning(f"API fail: {reason}")
    
    if _dr_state['last_states'] is None:
        log.error("DR unavailable: No prior state")
        return None, 'EMPTY'
    
    if _dr_state['count'] >= DR_MAX_ITER:
        log.error(f"DR exhausted after {DR_MAX_ITER} iteration(s)")
        return None, 'EMPTY'
    
    # Increment DR counter
    _dr_state['count'] += 1
    
    # Calculate elapsed time (cap at 5 minutes to prevent ghost planes)
    now = datetime.now(timezone.utc)
    elapsed_sec = min((now - _dr_state['last_time']).total_seconds(), 300)
    
    extrapolated = []
    
    for s in _dr_state['last_states']:
        new_state = s.copy()
        
        if s['velocity'] > 0 and s['true_track'] is not None:
            # Flat-earth extrapolation
            track_rad = math.radians(s['true_track'])
            
            # Δlat = (v * t * cos(track)) / 111320
            delta_lat = (s['velocity'] * elapsed_sec * math.cos(track_rad)) / 111320
            
            # Δlon = (v * t * sin(track)) / (111320 * cos(lat))
            lat_rad = math.radians(s['lat'])
            delta_lon = (s['velocity'] * elapsed_sec * math.sin(track_rad)) / (111320 * math.cos(lat_rad))
            
            new_state['lat'] = s['lat'] + delta_lat
            new_state['lon'] = s['lon'] + delta_lon
        
        # Keep only aircraft still in padded bbox
        if (PADDED_BBOX['min_lat'] <= new_state['lat'] <= PADDED_BBOX['max_lat'] and
            PADDED_BBOX['min_lon'] <= new_state['lon'] <= PADDED_BBOX['max_lon']):
            extrapolated.append(new_state)
    
    log.warning(f"DR | iter={_dr_state['count']} | aircraft={len(extrapolated)} | elapsed={elapsed_sec:.0f}s")
    
    if not extrapolated:
        return None, 'EMPTY'
    
    return extrapolated, 'DR'


def get_dr_count() -> int:
    """Get current dead-reckoning iteration count."""
    return _dr_state['count']


def set_dr_state(count: int, last_time: Optional[datetime] = None):
    """Restore DR state from persistence (for warm restart)."""
    global _dr_state
    _dr_state['count'] = count
    if last_time:
        _dr_state['last_time'] = last_time