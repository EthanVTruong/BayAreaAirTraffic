"""
Acquire Data - ADS-B Processing Pipeline
=============================================
Description: Offline utility that processes local OpenSky State Vector CSVs
             into training-ready Parquet files. Filters by bounding box,
             altitude, and flight validity. Used for model training data prep.
Author: RT Forecast Team
Version: 1.0.0

Usage:
    python src/acquire_data.py --input data/raw/opensky --output data/processed

Note: This is a standalone utility, not part of the live forecast pipeline.
"""

import pandas as pd
import numpy as np
import os
import gc
import glob
import gzip
import argparse
from datetime import timedelta
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Bounding boxes (WGS84 degrees)
# Buffer BBOX: Larger area for initial filtering (captures approaches)
BUFFER_BBOX = {
    "west": -123.0,
    "south": 36.8,
    "east": -121.2,
    "north": 38.2,
}

# Core BBOX: Smaller area for penetration filter
CORE_BBOX = {
    "west": -122.6,
    "south": 37.0,
    "east": -121.5,
    "north": 38.0,
}

# Altitude filters (feet) - applied AFTER meters→feet conversion
ALT_MIN_FT = 100
ALT_MAX_FT = 40000

# Meters to feet conversion factor
M_TO_FT = 3.28084

# Temporal parameters
RESAMPLE_FREQ = "30s"           # 30-second pulse
GAP_THRESHOLD_SECONDS = 300     # 5 minutes = new island
OVERLAP_MINUTES = 10            # Cross-day stitching overlap

# Bay Area airports for spatial enrichment
AIRPORTS = {
    "KSFO": {"lat": 37.6213, "lon": -122.379},
    "KSJC": {"lat": 37.3626, "lon": -121.929},
    "KOAK": {"lat": 37.7213, "lon": -122.221},
    "KPAO": {"lat": 37.4611, "lon": -122.115},
    "KSQL": {"lat": 37.5119, "lon": -122.249},
}

# Output schema
OUTPUT_COLS = [
    "timestamp", "island_id", "icao24", "lat", "lon",
    "velocity", "heading", "vertrate", "altitude_ft",
    "nearest_airport", "distance_to_airport"
]


# =============================================================================
# SPATIAL UTILITIES
# =============================================================================

def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray,
                         lat2: float, lon2: float) -> np.ndarray:
    """Vectorized Haversine distance in nautical miles."""
    R = 3440.065  # Earth radius in NM
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon = np.radians(lon1 - lon2)
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add nearest_airport and distance_to_airport using vectorized Haversine."""
    distances = {}
    for code, apt in AIRPORTS.items():
        distances[code] = haversine_vectorized(
            df["lat"].values, df["lon"].values,
            apt["lat"], apt["lon"]
        )
    
    dist_df = pd.DataFrame(distances, index=df.index)
    df["nearest_airport"] = dist_df.idxmin(axis=1)
    df["distance_to_airport"] = dist_df.min(axis=1)
    
    return df


def point_in_bbox(lat: pd.Series, lon: pd.Series, bbox: dict) -> pd.Series:
    """Check if points are inside a bounding box."""
    return (
        (lat >= bbox["south"]) &
        (lat <= bbox["north"]) &
        (lon >= bbox["west"]) &
        (lon <= bbox["east"])
    )


# =============================================================================
# PHASE 1: LOAD & FILTER (Per-File, Memory Efficient)
# =============================================================================

def load_and_filter_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load a single CSV and immediately filter to Buffer BBOX.
    This discards global data early to save memory.
    """
    print(f"  Loading {os.path.basename(filepath)}...", end=" ", flush=True)

    try:
        # Check for Gzip magic number (0x1f 0x8b)
        with open(filepath, 'rb') as f:
            is_gzipped = f.read(2) == b'\x1f\x8b'

        # Explicitly set compression to handle the .csv.gz.csv extension confusion
        df = pd.read_csv(
            filepath,
            low_memory=False,
            compression='gzip' if is_gzipped else None
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    raw_count = len(df)
    
    # Standardize column names (OpenSky uses various formats)
    col_map = {
        "time": "time",
        "lat": "lat", "latitude": "lat",
        "lon": "lon", "longitude": "lon",
        "baroaltitude": "baroaltitude", "baro_altitude": "baroaltitude",
        "geoaltitude": "geoaltitude", "geo_altitude": "geoaltitude",
        "velocity": "velocity", "groundspeed": "velocity",
        "heading": "heading", "track": "heading",
        "vertrate": "vertrate", "vertical_rate": "vertrate",
        "icao24": "icao24",
        "callsign": "callsign",
        "onground": "onground", "on_ground": "onground",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # --- IMMEDIATE: Filter to Buffer BBOX (discard global data) ---
    if "lat" not in df.columns or "lon" not in df.columns:
        print("ERROR: Missing lat/lon columns")
        return None
    
    df = df[point_in_bbox(df["lat"], df["lon"], BUFFER_BBOX)]
    after_bbox = len(df)
    
    if len(df) == 0:
        print(f"{raw_count:,} → 0 (no data in bbox)")
        return None
    
    # --- Convert altitude M → FT (MUST be first before any altitude filter) ---
    if "geoaltitude" in df.columns and df["geoaltitude"].notna().any():
        df["altitude_ft"] = df["geoaltitude"] * M_TO_FT
    elif "baroaltitude" in df.columns:
        df["altitude_ft"] = df["baroaltitude"] * M_TO_FT
    else:
        df["altitude_ft"] = np.nan
    
    # --- Remove ground traffic ---
    if "onground" in df.columns:
        df = df[df["onground"] == False]
    after_ground = len(df)
    
    # --- Altitude filter (100ft to 40,000ft) - uses altitude_ft ---
    if "altitude_ft" in df.columns:
        df = df[df["altitude_ft"].between(ALT_MIN_FT, ALT_MAX_FT)]
    after_alt = len(df)
    
    # --- Create timestamp ---
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        print("ERROR: No time column")
        return None
    
    # --- Drop invalid records ---
    df = df.dropna(subset=["lat", "lon", "timestamp", "icao24"])
    final = len(df)
    
    print(f"{raw_count:,} → {final:,} (bbox:{after_bbox:,}, ground:{after_ground:,}, alt:{after_alt:,})")
    
    return df


# =============================================================================
# PHASE 2: CROSS-DAY STITCHING
# =============================================================================

def extract_overlap_data(df: pd.DataFrame, minutes: int = OVERLAP_MINUTES) -> pd.DataFrame:
    """Extract the last N minutes of data for cross-day stitching."""
    if len(df) == 0:
        return pd.DataFrame()
    
    cutoff = df["timestamp"].max() - timedelta(minutes=minutes)
    return df[df["timestamp"] >= cutoff].copy()


def stitch_with_previous(current_df: pd.DataFrame, 
                         overlap_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge overlap data from previous day into current day."""
    if overlap_df is None or len(overlap_df) == 0:
        return current_df
    
    if len(current_df) == 0:
        return current_df
    
    # Combine and deduplicate
    combined = pd.concat([overlap_df, current_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["icao24", "timestamp"], keep="last")
    
    return combined


# =============================================================================
# PHASE 3: ISLAND PROTOCOL
# =============================================================================

def assign_island_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign island_id to each contiguous trajectory segment.
    
    ISLAND PROTOCOL:
    - New island when icao24 changes
    - New island when time gap > 5 minutes within same aircraft
    
    Must be done on RAW timestamps BEFORE resampling.
    """
    # Sort by aircraft, then time
    df = df.sort_values(["icao24", "timestamp"]).reset_index(drop=True)
    
    # Detect boundaries
    df["prev_icao24"] = df["icao24"].shift(1)
    df["time_gap"] = df["timestamp"].diff()
    
    gap_threshold = timedelta(seconds=GAP_THRESHOLD_SECONDS)
    
    df["new_island"] = (
        (df["icao24"] != df["prev_icao24"]) |
        (df["time_gap"] > gap_threshold) |
        (df["time_gap"].isna())
    )
    
    df["island_id"] = df["new_island"].cumsum()
    
    # Cleanup
    df = df.drop(columns=["prev_icao24", "time_gap", "new_island"])
    
    return df


def resample_island(island_df: pd.DataFrame, island_id: int) -> pd.DataFrame:
    """
    Resample a single island to 30-second pulse.
    
    INTERPOLATION RULES:
    - LINEAR: lat, lon, altitude_ft, velocity (smooth physical quantities)
    - FORWARD-FILL: heading, vertrate (avoid angle/sign artifacts)
    
    CONSTRAINT: This function operates on ONE island only.
    """
    if len(island_df) < 2:
        island_df = island_df.copy()
        island_df["is_interpolated"] = False
        return island_df
    
    # Metadata (constant for this island)
    icao24 = island_df["icao24"].iloc[0]
    callsign = island_df["callsign"].iloc[0] if "callsign" in island_df.columns else None
    
    # Set timestamp as index
    island_df = island_df.set_index("timestamp").sort_index()
    
    # Target time range at 30-second intervals
    start = island_df.index.min().floor(RESAMPLE_FREQ)
    end = island_df.index.max().ceil(RESAMPLE_FREQ)
    target_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ, tz="UTC")
    
    if len(target_index) == 0:
        return pd.DataFrame()
    
    # Reindex to combined timestamps
    combined_index = island_df.index.union(target_index)
    resampled = island_df.reindex(combined_index)
    
    # Track interpolated vs original
    resampled["is_interpolated"] = ~resampled.index.isin(island_df.index)
    
    # LINEAR interpolation for position/speed
    for col in ["lat", "lon", "altitude_ft", "velocity"]:
        if col in resampled.columns:
            resampled[col] = resampled[col].interpolate(method="linear")
    
    # FORWARD-FILL for kinematic features (avoid artifacts)
    for col in ["heading", "vertrate"]:
        if col in resampled.columns:
            resampled[col] = resampled[col].ffill()
    
    # Constant metadata
    resampled["icao24"] = icao24
    if callsign:
        resampled["callsign"] = callsign
    resampled["island_id"] = island_id
    
    # Keep only 30-second intervals
    resampled = resampled.reindex(target_index)
    resampled = resampled.dropna(subset=["lat", "lon"])
    
    resampled.index.name = "timestamp"
    return resampled.reset_index()


def resample_all_islands(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Apply resampling to all islands. Returns DataFrame and stats."""
    print("  Resampling to 30-second pulse...")
    
    original_count = len(df)
    chunks = []
    
    islands = df.groupby("island_id")
    total = len(islands)
    
    for i, (island_id, island_df) in enumerate(islands):
        if (i + 1) % 500 == 0:
            print(f"    Island {i+1}/{total}...", end="\r")
        
        resampled = resample_island(island_df.copy(), island_id)
        if len(resampled) > 0:
            chunks.append(resampled)
    
    print(f"    Processed {total} islands" + " " * 20)
    
    if not chunks:
        return pd.DataFrame(), {}
    
    result = pd.concat(chunks, ignore_index=True)
    
    # Stats
    interpolated = result["is_interpolated"].sum() if "is_interpolated" in result.columns else 0
    stats = {
        "original_points": original_count,
        "resampled_points": len(result),
        "interpolated_points": interpolated,
        "interpolation_ratio": interpolated / max(len(result), 1),
    }
    
    return result, stats


# =============================================================================
# PHASE 4: CORE PENETRATION FILTER
# =============================================================================

def filter_core_penetration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only islands that have at least one point inside Core BBOX.
    This removes pure overflights that never enter the terminal area.
    """
    # Mark points in core
    df["in_core"] = point_in_bbox(df["lat"], df["lon"], CORE_BBOX)
    
    # Find islands with core penetration
    core_islands = df[df["in_core"]].groupby("island_id").size().index
    
    # Filter to those islands
    df = df[df["island_id"].isin(core_islands)]
    
    # Cleanup
    df = df.drop(columns=["in_core"])
    
    return df


# =============================================================================
# PHASE 5: QUALITY REPORT
# =============================================================================

def generate_quality_report(df: pd.DataFrame, stats: dict, output_dir: str) -> str:
    """Generate quality report with required metrics."""
    report_path = os.path.join(output_dir, "quality_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BAY AREA ADS-B DATA QUALITY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # --- 1. ISLAND COUNT ---
        f.write("1. ISLAND COUNT (Contiguous Trajectory Segments)\n")
        f.write("-" * 50 + "\n")
        n_islands = df["island_id"].nunique()
        n_aircraft = df["icao24"].nunique()
        
        f.write(f"Unique island_id:  {n_islands:,}\n")
        f.write(f"Unique icao24:     {n_aircraft:,}\n")
        f.write(f"Islands/aircraft:  {n_islands/max(n_aircraft,1):.2f}\n\n")
        
        # Island duration stats
        island_stats = df.groupby("island_id")["timestamp"].agg(["min", "max", "count"])
        island_stats["duration_min"] = (island_stats["max"] - island_stats["min"]).dt.total_seconds() / 60
        
        f.write("Island duration (minutes):\n")
        f.write(f"  Min:    {island_stats['duration_min'].min():.1f}\n")
        f.write(f"  Median: {island_stats['duration_min'].median():.1f}\n")
        f.write(f"  Mean:   {island_stats['duration_min'].mean():.1f}\n")
        f.write(f"  Max:    {island_stats['duration_min'].max():.1f}\n\n")
        
        # --- 2. INTERPOLATION RATIO ---
        f.write("2. INTERPOLATION RATIO\n")
        f.write("-" * 50 + "\n")
        f.write("Formula: interpolated_points / total_resampled_points\n\n")
        
        total = stats.get("resampled_points", len(df))
        interp = stats.get("interpolated_points", 0)
        
        if "is_interpolated" in df.columns:
            interp = df["is_interpolated"].sum()
            total = len(df)
        
        ratio = interp / max(total, 1)
        
        f.write(f"Original points:      {stats.get('original_points', 'N/A'):,}\n")
        f.write(f"Resampled points:     {total:,}\n")
        f.write(f"Interpolated points:  {interp:,}\n")
        f.write(f"Interpolation ratio:  {ratio*100:.1f}%\n\n")
        
        # --- 3. GAP ANALYSIS ---
        f.write("3. GAP ANALYSIS (Global gaps > 1 hour)\n")
        f.write("-" * 50 + "\n")
        
        unique_times = df["timestamp"].drop_duplicates().sort_values()
        
        if len(unique_times) > 1:
            time_gaps = unique_times.diff()
            large_gaps = time_gaps[time_gaps > timedelta(hours=1)]
            
            if len(large_gaps) == 0:
                f.write("✓ No gaps > 1 hour detected.\n\n")
            else:
                f.write(f"⚠ Found {len(large_gaps)} gaps > 1 hour:\n")
                gap_times = unique_times[large_gaps.index]
                
                for i, (idx, gap) in enumerate(large_gaps.items()):
                    if i >= 15:
                        f.write(f"  ... and {len(large_gaps) - 15} more\n")
                        break
                    gap_start = gap_times.loc[idx] - gap
                    gap_end = gap_times.loc[idx]
                    gap_hrs = gap.total_seconds() / 3600
                    f.write(f"  {gap_start.strftime('%Y-%m-%d %H:%M')} → ")
                    f.write(f"{gap_end.strftime('%Y-%m-%d %H:%M')} ({gap_hrs:.1f}h)\n")
                f.write("\n")
        
        # --- 4. DATASET SUMMARY ---
        f.write("4. DATASET SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total records:  {len(df):,}\n")
        f.write(f"Time start:     {df['timestamp'].min()}\n")
        f.write(f"Time end:       {df['timestamp'].max()}\n")
        duration = df["timestamp"].max() - df["timestamp"].min()
        f.write(f"Duration:       {duration.days} days\n\n")
        
        if "nearest_airport" in df.columns:
            f.write("By nearest airport:\n")
            for apt, count in df["nearest_airport"].value_counts().items():
                f.write(f"  {apt}: {count:>10,} ({100*count/len(df):5.1f}%)\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    return report_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(input_dir: str, output_dir: str):
    """
    Main acquisition pipeline.
    
    Processes CSVs in sorted order (chronological).
    Implements cross-day stitching and Island Protocol.
    """
    print("=" * 70)
    print("BAY AREA ADS-B ACQUISITION PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input:         {input_dir}")
    print(f"  Output:        {output_dir}")
    print(f"  Buffer BBOX:   [{BUFFER_BBOX['west']}, {BUFFER_BBOX['south']}, {BUFFER_BBOX['east']}, {BUFFER_BBOX['north']}]")
    print(f"  Core BBOX:     [{CORE_BBOX['west']}, {CORE_BBOX['south']}, {CORE_BBOX['east']}, {CORE_BBOX['north']}]")
    print(f"  Altitude:      {ALT_MIN_FT} - {ALT_MAX_FT} ft")
    print(f"  Gap threshold: {GAP_THRESHOLD_SECONDS}s")
    print(f"  Resample:      {RESAMPLE_FREQ}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find CSV files (sorted for chronological processing)
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv*")))
    
    if not csv_files:
        print(f"\nERROR: No CSV files found in {input_dir}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files")
    print("-" * 70)
    
    # Process files with cross-day stitching
    all_data = []
    prev_overlap = None
    total_original = 0
    
    for i, filepath in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] ", end="")
        
        # Load and filter
        df = load_and_filter_csv(filepath)
        
        if df is None or len(df) == 0:
            continue
        
        # Cross-day stitching
        df = stitch_with_previous(df, prev_overlap)
        
        # Save overlap for next file
        prev_overlap = extract_overlap_data(df)
        
        total_original += len(df)
        all_data.append(df)
        
        # Memory cleanup per file
        gc.collect()
    
    if not all_data:
        print("\nERROR: No data extracted from any files")
        return
    
    # Combine all data
    print("\n" + "-" * 70)
    print("Combining all files...")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Deduplicate (from stitching)
    combined = combined.drop_duplicates(subset=["icao24", "timestamp"], keep="last")
    print(f"  Combined: {len(combined):,} records")
    
    # Clear memory
    del all_data
    del prev_overlap
    gc.collect()
    
    # Phase 3: Assign island IDs
    print("\nAssigning island IDs...")
    combined = assign_island_ids(combined)
    n_islands_pre = combined["island_id"].nunique()
    print(f"  Islands (pre-filter): {n_islands_pre:,}")
    
    # Phase 4: Resample all islands
    print("\nResampling trajectories...")
    combined, interp_stats = resample_all_islands(combined)
    interp_stats["original_points"] = total_original
    
    if len(combined) == 0:
        print("ERROR: No data after resampling")
        return
    
    # Phase 5: Core penetration filter
    print("\nFiltering for core penetration...")
    pre_filter = len(combined)
    combined = filter_core_penetration(combined)
    print(f"  {pre_filter:,} → {len(combined):,} (kept islands entering core)")
    
    # Phase 6: Spatial enrichment
    print("\nAdding spatial features...")
    combined = add_spatial_features(combined)
    
    # Phase 7: Final cleanup
    print("\nFinal cleanup...")
    
    # Strict sort by (island_id, timestamp)
    combined = combined.sort_values(["island_id", "timestamp"]).reset_index(drop=True)
    
    # Reassign island_id to be contiguous after filtering
    island_map = {old: new for new, old in enumerate(combined["island_id"].unique(), 1)}
    combined["island_id"] = combined["island_id"].map(island_map)
    
    # Ensure output columns (add missing, drop extras)
    for col in OUTPUT_COLS:
        if col not in combined.columns:
            combined[col] = np.nan
    
    final_cols = [c for c in OUTPUT_COLS if c in combined.columns]
    extra_cols = ["is_interpolated"] if "is_interpolated" in combined.columns else []
    combined = combined[final_cols + extra_cols]
    
    # Save output
    output_path = os.path.join(output_dir, "bayarea_state_vectors.parquet")
    print(f"\nSaving: {output_path}")
    combined.to_parquet(output_path, compression="snappy", index=False)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")
    
    # Generate quality report
    print("\nGenerating quality report...")
    report_path = generate_quality_report(combined, interp_stats, output_dir)
    print(f"  Report: {report_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Output:          {output_path}")
    print(f"Records:         {len(combined):,}")
    print(f"Islands:         {combined['island_id'].nunique():,}")
    print(f"Aircraft:        {combined['icao24'].nunique():,}")
    print(f"Time range:      {combined['timestamp'].min()}")
    print(f"                 {combined['timestamp'].max()}")
    print(f"Interp ratio:    {interp_stats.get('interpolation_ratio', 0)*100:.1f}%")
    
    # Cleanup
    del combined
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Bay Area ADS-B Acquisition Pipeline (CSV Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python acquire_data_traffic.py --input data/raw/opensky --output data/processed

The script processes all .csv and .csv.gz files in the input directory,
applying the Island Protocol for model-ready output.

Recommended: 14-30 days of data for meaningful model training.
        """
    )
    parser.add_argument("--input", required=True,
                        help="Directory containing OpenSky CSV files")
    parser.add_argument("--output", default="data/processed",
                        help="Output directory (default: data/processed)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        return
    
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()