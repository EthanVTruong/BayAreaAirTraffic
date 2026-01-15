"""
Replay - Sample Data Replay Engine
=============================================
Description: Enables running the RT Forecast system on historical or synthetic
             ADS-B data instead of live OpenSky API. Supports time-aligned window
             iteration, configurable playback speed, and synthetic scenario
             generation for stress testing.
Author: RT Forecast Team
Version: 3.0.0

Use Cases:
    - Heavy-traffic simulation and stress testing
    - UI/UX validation without live data
    - Model performance benchmarking
    - Airport congestion scenario replay

Usage:
    python -m rt_forecast --replay data/scenarios/heavy_congestion.parquet
    python -m rt_forecast --synthetic --serve
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator

import numpy as np
import pandas as pd

from .config import PADDED_BBOX, CORE_BBOX, WINDOW_MINUTES

log = logging.getLogger('rt_forecast.replay')

# =============================================================================
# REPLAY ENGINE
# =============================================================================

class ReplayEngine:
    """
    Replays ADS-B data from Parquet files, simulating live fetch_states() calls.
    
    Features:
    - Time-aligned window iteration
    - Configurable playback speed
    - Loop/single-pass modes
    - Status simulation (always returns 'LIVE' or 'REPLAY')
    """
    
    def __init__(
        self,
        data_path: Path,
        speed_multiplier: float = 1.0,
        loop: bool = True,
        start_window: Optional[datetime] = None
    ):
        """
        Initialize replay engine.
        
        Args:
            data_path: Path to Parquet file with ADS-B data
            speed_multiplier: Playback speed (1.0 = real-time, 10.0 = 10x faster)
            loop: Whether to loop back to start when data exhausted
            start_window: Optional starting window (default: first in data)
        """
        self.data_path = Path(data_path)
        self.speed_multiplier = speed_multiplier
        self.loop = loop
        self.start_window = start_window
        
        # Load and prepare data
        self._load_data()
        
        # Playback state
        self.current_idx = 0
        self.iteration_count = 0
        
        log.info(f"ReplayEngine initialized: {len(self.windows)} windows available")
    
    def _load_data(self):
        """Load and index data by 5-minute windows."""
        log.info(f"Loading replay data: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Replay data not found: {self.data_path}")
        
        self.df = pd.read_parquet(self.data_path)
        
        # Ensure timestamp column
        if 'timestamp' not in self.df.columns:
            raise ValueError("Replay data must have 'timestamp' column")
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Create window column (floor to 5-minute boundary)
        self.df['window'] = self.df['timestamp'].dt.floor(f'{WINDOW_MINUTES}min')
        
        # Get unique windows in order
        self.windows = sorted(self.df['window'].unique())
        
        if not self.windows:
            raise ValueError("No windows found in replay data")
        
        # Build window -> states lookup
        self.window_data: Dict[datetime, List[Dict]] = {}
        for window in self.windows:
            window_df = self.df[self.df['window'] == window]
            self.window_data[window] = self._df_to_states(window_df)
        
        log.info(f"Loaded {len(self.df)} records across {len(self.windows)} windows")
        log.info(f"Time range: {self.windows[0]} to {self.windows[-1]}")
        
        # Set start position
        if self.start_window and self.start_window in self.window_data:
            self.current_idx = self.windows.index(self.start_window)
    
    def _df_to_states(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame rows to state dictionaries."""
        states = []
        for _, row in df.iterrows():
            states.append({
                'icao24': row.get('icao24', f'SIM{np.random.randint(10000):05d}'),
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'altitude_ft': float(row.get('altitude_ft', row.get('altitude', 10000))),
                'velocity': float(row.get('velocity', 200)),
                'true_track': float(row.get('true_track', row.get('heading', 0))),
                'vert_rate': float(row.get('vert_rate', 0))
            })
        return states
    
    def fetch_states(self) -> Tuple[Optional[List[Dict]], str]:
        """
        Fetch next window of states (simulates live fetch_states()).
        
        Returns:
            Tuple of (states_list, status)
            - status is 'REPLAY' for replayed data
        """
        if self.current_idx >= len(self.windows):
            if self.loop:
                self.current_idx = 0
                log.info("REPLAY: Looping back to start")
            else:
                log.warning("REPLAY: Data exhausted")
                return None, 'EMPTY'
        
        window = self.windows[self.current_idx]
        states = self.window_data[window]
        
        self.current_idx += 1
        self.iteration_count += 1
        
        log.info(f"REPLAY | window={window} | aircraft={len(states)} | iter={self.iteration_count}")
        
        return states, 'REPLAY'
    
    def get_current_window(self) -> datetime:
        """Get the current window being replayed (for time display)."""
        if self.current_idx > 0 and self.current_idx <= len(self.windows):
            return self.windows[self.current_idx - 1]
        elif self.windows:
            return self.windows[0]
        return datetime.now(timezone.utc)
    
    def get_dr_count(self) -> int:
        """Replay never uses dead-reckoning."""
        return 0
    
    def reset(self):
        """Reset to beginning of data."""
        self.current_idx = 0
        self.iteration_count = 0


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_heavy_congestion_scenario(
    output_path: Path,
    duration_hours: float = 2.0,
    peak_aircraft: int = 120,
    seed: int = 42
) -> Path:
    """
    Generate synthetic heavy congestion scenario for testing.
    
    Simulates:
    - Rush hour with 3x normal traffic
    - Multiple simultaneous approaches to SFO/OAK
    - Holding patterns (aircraft circling)
    - Ground stops causing traffic backup
    - Weather deviation routes
    
    Args:
        output_path: Where to save the Parquet file
        duration_hours: Scenario duration
        peak_aircraft: Maximum simultaneous aircraft
        seed: Random seed for reproducibility
    
    Returns:
        Path to generated Parquet file
    """
    np.random.seed(seed)
    log.info(f"Generating heavy congestion scenario: {duration_hours}h, peak={peak_aircraft}")
    
    records = []
    start_time = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)  # 4 PM rush
    n_windows = int(duration_hours * 60 / WINDOW_MINUTES)
    
    # Airport locations
    airports = {
        'KSFO': (37.6213, -122.3790),
        'KOAK': (37.7213, -122.2208),
        'KSJC': (37.3626, -121.9290),
    }
    
    for w in range(n_windows):
        window_time = start_time + timedelta(minutes=w * WINDOW_MINUTES)
        
        # Traffic builds to peak at 30min, stays high, then slowly decreases
        progress = w / n_windows
        if progress < 0.25:
            traffic_factor = 0.5 + 2.0 * (progress / 0.25)  # Ramp up
        elif progress < 0.75:
            traffic_factor = 2.5  # Peak congestion
        else:
            traffic_factor = 2.5 - 1.5 * ((progress - 0.75) / 0.25)  # Wind down
        
        n_aircraft = int(peak_aircraft * traffic_factor / 2.5)
        n_aircraft = max(20, min(peak_aircraft, n_aircraft))
        
        for i in range(n_aircraft):
            # Distribute aircraft types for heavy congestion
            aircraft_type = np.random.choice(
                ['sfo_arrival', 'oak_arrival', 'sjc_arrival', 
                 'holding', 'departure', 'overflight', 'go_around'],
                p=[0.25, 0.20, 0.10, 0.15, 0.15, 0.10, 0.05]
            )
            
            if aircraft_type == 'sfo_arrival':
                # Heavy SFO arrivals from multiple directions
                approach = np.random.choice(['north', 'south', 'east'])
                if approach == 'north':
                    lat = np.random.uniform(37.75, 37.95)
                    lon = np.random.uniform(-122.5, -122.3)
                    heading = np.random.uniform(170, 190)
                elif approach == 'south':
                    lat = np.random.uniform(37.35, 37.55)
                    lon = np.random.uniform(-122.45, -122.25)
                    heading = np.random.uniform(350, 370) % 360
                else:
                    lat = np.random.uniform(37.55, 37.70)
                    lon = np.random.uniform(-122.0, -121.8)
                    heading = np.random.uniform(260, 280)
                
                altitude = np.random.uniform(2000, 7000)
                velocity = np.random.uniform(160, 220)
                vert_rate = np.random.uniform(-10, -3)
                
            elif aircraft_type == 'oak_arrival':
                lat = np.random.uniform(37.65, 37.85)
                lon = np.random.uniform(-122.35, -122.1)
                heading = np.random.uniform(140, 220)
                altitude = np.random.uniform(2500, 6000)
                velocity = np.random.uniform(150, 200)
                vert_rate = np.random.uniform(-8, -2)
                
            elif aircraft_type == 'sjc_arrival':
                lat = np.random.uniform(37.25, 37.45)
                lon = np.random.uniform(-122.1, -121.85)
                heading = np.random.uniform(0, 360)
                altitude = np.random.uniform(3000, 8000)
                velocity = np.random.uniform(170, 230)
                vert_rate = np.random.uniform(-6, -2)
                
            elif aircraft_type == 'holding':
                # Holding patterns - aircraft circling waiting for clearance
                hold_center = np.random.choice(list(airports.keys()))
                center_lat, center_lon = airports[hold_center]
                
                # Offset from airport
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0.1, 0.25)  # degrees
                lat = center_lat + radius * np.cos(angle)
                lon = center_lon + radius * np.sin(angle)
                
                # Circling heading (changes with position in hold)
                heading = (np.degrees(angle) + 90) % 360
                altitude = np.random.uniform(4000, 8000)
                velocity = np.random.uniform(180, 220)
                vert_rate = np.random.uniform(-1, 1)  # Level flight
                
            elif aircraft_type == 'departure':
                # Departures climbing out
                dep_apt = np.random.choice(list(airports.keys()))
                base_lat, base_lon = airports[dep_apt]
                
                heading = np.random.uniform(0, 360)
                dist = np.random.uniform(0.05, 0.2)
                lat = base_lat + dist * np.cos(np.radians(heading))
                lon = base_lon + dist * np.sin(np.radians(heading))
                altitude = np.random.uniform(3000, 15000)
                velocity = np.random.uniform(200, 300)
                vert_rate = np.random.uniform(8, 20)
                
            elif aircraft_type == 'overflight':
                # High altitude overflights
                lat = np.random.uniform(CORE_BBOX['min_lat'], CORE_BBOX['max_lat'])
                lon = np.random.uniform(CORE_BBOX['min_lon'], CORE_BBOX['max_lon'])
                heading = np.random.uniform(0, 360)
                altitude = np.random.uniform(25000, 40000)
                velocity = np.random.uniform(400, 500)
                vert_rate = np.random.uniform(-2, 2)
                
            else:  # go_around
                # Missed approaches - climbing back out
                apt = np.random.choice(['KSFO', 'KOAK'])
                base_lat, base_lon = airports[apt]
                lat = base_lat + np.random.uniform(-0.05, 0.05)
                lon = base_lon + np.random.uniform(-0.05, 0.05)
                heading = np.random.uniform(0, 360)
                altitude = np.random.uniform(1500, 4000)
                velocity = np.random.uniform(180, 240)
                vert_rate = np.random.uniform(10, 25)
            
            records.append({
                'icao24': f'{aircraft_type[:2].upper()}{w:04d}{i:03d}',
                'lat': lat,
                'lon': lon,
                'altitude_ft': altitude,
                'velocity': velocity,
                'true_track': heading,
                'vert_rate': vert_rate,
                'timestamp': window_time,
                'scenario': 'heavy_congestion',
                'aircraft_type': aircraft_type
            })
    
    df = pd.DataFrame(records)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    
    log.info(f"Generated {len(df)} aircraft states across {n_windows} windows")
    log.info(f"Saved to: {output_path}")
    
    # Generate summary stats
    stats = {
        'total_records': len(df),
        'n_windows': n_windows,
        'duration_hours': duration_hours,
        'peak_aircraft': peak_aircraft,
        'time_range': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max())
        },
        'aircraft_types': df['aircraft_type'].value_counts().to_dict(),
        'avg_per_window': len(df) / n_windows
    }
    
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return output_path


def generate_airport_stress_test(
    output_path: Path,
    airport: str = 'KSFO',
    duration_hours: float = 1.0,
    arrival_rate: int = 40,  # arrivals per hour
    seed: int = 123
) -> Path:
    """
    Generate concentrated arrival stress test for a single airport.
    
    Simulates maximum arrival rate with:
    - Sequential ILS approaches
    - Parallel runway operations
    - Missed approaches
    - Departure conflicts
    """
    np.random.seed(seed)
    
    airport_coords = {
        'KSFO': (37.6213, -122.3790, 'San Francisco Intl'),
        'KOAK': (37.7213, -122.2208, 'Oakland Intl'),
        'KSJC': (37.3626, -121.9290, 'San Jose Intl'),
    }
    
    if airport not in airport_coords:
        raise ValueError(f"Unknown airport: {airport}")
    
    apt_lat, apt_lon, apt_name = airport_coords[airport]
    
    log.info(f"Generating {airport} stress test: {arrival_rate}/hr for {duration_hours}h")
    
    records = []
    start_time = datetime(2025, 1, 15, 17, 0, 0, tzinfo=timezone.utc)
    n_windows = int(duration_hours * 60 / WINDOW_MINUTES)
    
    for w in range(n_windows):
        window_time = start_time + timedelta(minutes=w * WINDOW_MINUTES)
        
        # Calculate aircraft for this window
        arrivals_per_window = int(arrival_rate * WINDOW_MINUTES / 60)
        arrivals_per_window = max(5, arrivals_per_window + np.random.randint(-2, 3))
        
        # Generate arrival stream
        for i in range(arrivals_per_window):
            # Distance from airport (closer = more congested feel)
            distance = np.random.uniform(0.02, 0.35)  # degrees
            
            # Approach angle (mostly aligned with runway)
            if airport == 'KSFO':
                # RWY 28L/R approaches from the east
                approach_heading = np.random.uniform(270, 290) if np.random.random() > 0.3 else np.random.uniform(80, 100)
            else:
                approach_heading = np.random.uniform(0, 360)
            
            # Position based on approach heading (aircraft coming FROM opposite direction)
            inbound_heading = (approach_heading + 180) % 360
            lat = apt_lat + distance * np.cos(np.radians(inbound_heading))
            lon = apt_lon + distance * np.sin(np.radians(inbound_heading))
            
            # Altitude decreases with distance (3° glide slope approximation)
            altitude = max(500, 3000 + distance * 15000)
            
            # Speed decreases closer to airport
            velocity = 250 - distance * 200
            velocity = max(140, min(280, velocity))
            
            records.append({
                'icao24': f'ARR{w:04d}{i:03d}',
                'lat': lat,
                'lon': lon,
                'altitude_ft': altitude,
                'velocity': velocity,
                'true_track': approach_heading,
                'vert_rate': np.random.uniform(-8, -3),
                'timestamp': window_time,
                'scenario': f'{airport}_stress',
                'aircraft_type': 'arrival'
            })
        
        # Add some departures for conflict
        n_departures = np.random.randint(2, 5)
        for i in range(n_departures):
            heading = np.random.uniform(0, 360)
            distance = np.random.uniform(0.03, 0.15)
            
            records.append({
                'icao24': f'DEP{w:04d}{i:03d}',
                'lat': apt_lat + distance * np.cos(np.radians(heading)),
                'lon': apt_lon + distance * np.sin(np.radians(heading)),
                'altitude_ft': np.random.uniform(2000, 10000),
                'velocity': np.random.uniform(200, 280),
                'true_track': heading,
                'vert_rate': np.random.uniform(10, 20),
                'timestamp': window_time,
                'scenario': f'{airport}_stress',
                'aircraft_type': 'departure'
            })
    
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    log.info(f"Generated {len(df)} records for {airport} stress test")
    
    return output_path


def generate_multi_airport_rush(
    output_path: Path,
    duration_hours: float = 1.5,
    seed: int = 456
) -> Path:
    """
    Generate simultaneous rush hour at all Bay Area airports.
    
    Maximum realistic traffic scenario with:
    - SFO at capacity (60+ arrivals/hr)
    - OAK heavy arrivals
    - SJC moderate traffic
    - All GA airports active
    - Multiple holding patterns
    """
    np.random.seed(seed)
    log.info("Generating multi-airport rush hour scenario")
    
    records = []
    start_time = datetime(2025, 1, 15, 17, 30, 0, tzinfo=timezone.utc)
    n_windows = int(duration_hours * 60 / WINDOW_MINUTES)
    
    airports = {
        'KSFO': {'coords': (37.6213, -122.3790), 'rate': 60, 'type': 'major'},
        'KOAK': {'coords': (37.7213, -122.2208), 'rate': 35, 'type': 'major'},
        'KSJC': {'coords': (37.3626, -121.9290), 'rate': 25, 'type': 'major'},
        'KPAO': {'coords': (37.4611, -122.1150), 'rate': 15, 'type': 'ga'},
        'KSQL': {'coords': (37.5119, -122.2490), 'rate': 12, 'type': 'ga'},
        'KHWD': {'coords': (37.6592, -122.1217), 'rate': 10, 'type': 'ga'},
    }
    
    for w in range(n_windows):
        window_time = start_time + timedelta(minutes=w * WINDOW_MINUTES)
        
        for apt_code, apt_info in airports.items():
            apt_lat, apt_lon = apt_info['coords']
            base_rate = apt_info['rate']
            
            # Aircraft per window
            n_aircraft = int(base_rate * WINDOW_MINUTES / 60)
            n_aircraft += np.random.randint(-2, 3)
            n_aircraft = max(2, n_aircraft)
            
            for i in range(n_aircraft):
                # Mix of arrivals and departures
                is_arrival = np.random.random() < 0.6
                
                distance = np.random.uniform(0.02, 0.3)
                heading = np.random.uniform(0, 360)
                
                lat = apt_lat + distance * np.cos(np.radians(heading))
                lon = apt_lon + distance * np.sin(np.radians(heading))
                
                if is_arrival:
                    altitude = max(1000, 2000 + distance * 20000)
                    velocity = 200 - distance * 100
                    vert_rate = np.random.uniform(-10, -2)
                    track = (heading + 180) % 360  # Inbound
                else:
                    altitude = np.random.uniform(2000, 12000)
                    velocity = np.random.uniform(180, 280)
                    vert_rate = np.random.uniform(5, 18)
                    track = heading  # Outbound
                
                records.append({
                    'icao24': f'{apt_code[1:3]}{w:04d}{i:03d}',
                    'lat': lat,
                    'lon': lon,
                    'altitude_ft': altitude,
                    'velocity': max(100, min(350, velocity)),
                    'true_track': track,
                    'vert_rate': vert_rate,
                    'timestamp': window_time,
                    'scenario': 'multi_airport_rush',
                    'airport': apt_code,
                    'aircraft_type': 'arrival' if is_arrival else 'departure'
                })
        
        # Add overflights
        n_overflights = np.random.randint(5, 12)
        for i in range(n_overflights):
            records.append({
                'icao24': f'OVR{w:04d}{i:03d}',
                'lat': np.random.uniform(CORE_BBOX['min_lat'], CORE_BBOX['max_lat']),
                'lon': np.random.uniform(CORE_BBOX['min_lon'], CORE_BBOX['max_lon']),
                'altitude_ft': np.random.uniform(28000, 42000),
                'velocity': np.random.uniform(420, 520),
                'true_track': np.random.uniform(0, 360),
                'vert_rate': np.random.uniform(-2, 2),
                'timestamp': window_time,
                'scenario': 'multi_airport_rush',
                'airport': 'OVERFLIGHT',
                'aircraft_type': 'overflight'
            })
    
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    log.info(f"Generated {len(df)} records for multi-airport rush")
    
    return output_path


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_scenarios(scenarios_dir: Path) -> List[Dict]:
    """List available replay scenarios."""
    scenarios = []
    
    if not scenarios_dir.exists():
        return scenarios
    
    for f in scenarios_dir.glob('*.parquet'):
        stats_file = f.with_suffix('.stats.json')
        
        info = {
            'name': f.stem,
            'path': str(f),
            'size_mb': f.stat().st_size / 1024 / 1024
        }
        
        if stats_file.exists():
            with open(stats_file) as sf:
                info['stats'] = json.load(sf)
        
        scenarios.append(info)
    
    return scenarios