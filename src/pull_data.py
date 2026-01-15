#!/usr/bin/env python3
"""
Pull Data - OpenSky Dataset Downloader
=============================================
Description: Downloads hourly state vector CSVs from OpenSky's public dataset
             repository. Extracts tar archives and organizes CSVs into a single
             output folder for training data preparation.
Author: RT Forecast Team
Version: 1.0.0

Usage:
    python src/pull_data.py --output data/raw/opensky

Note: Downloads ~22GB compressed data. This is a standalone utility for
      training data acquisition, not part of the live forecast pipeline.
"""

import os
import sys
import tarfile
import argparse
import requests
import shutil
import gzip
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base URL for OpenSky datasets
BASE_URL = "https://opensky-network.org/datasets/states"

# Target dates (7 days spread across summer 2019)
TARGET_DATES = [
    "2019-07-08",
    "2019-07-15",
    "2019-07-22",
    "2019-07-29",
    "2019-08-05",
    "2019-08-12",
    "2019-08-19",
]

# Hours to download (all 24)
HOURS = [f"{h:02d}" for h in range(24)]

# Download settings
CHUNK_SIZE = 8192 * 16  # 128KB chunks
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
MAX_WORKERS = 4  # Parallel downloads (be nice to their servers)


def diagnose_file(filepath: str) -> dict:
    """Diagnose a downloaded file's format."""
    info = {
        "path": filepath,
        "exists": os.path.exists(filepath),
        "size": 0,
        "format": "unknown",
        "magic_hex": "",
        "tar_filename": "",
        "tar_size": "",
    }
    
    if not info["exists"]:
        return info
    
    info["size"] = os.path.getsize(filepath)
    
    with open(filepath, 'rb') as f:
        header = f.read(600)
    
    info["magic_hex"] = header[:20].hex()
    
    # Identify format
    if header[:2] == b'\x1f\x8b':
        info["format"] = "gzip"
    elif header[257:262] == b'ustar':
        info["format"] = "tar (ustar)"
    elif header[:4] == b'PK\x03\x04':
        info["format"] = "zip"
    elif header[:2] == b'BZ':
        info["format"] = "bzip2"
    elif b'time,' in header or b'icao24,' in header:
        info["format"] = "csv (plain)"
    else:
        # Check if it looks like old-style tar (filename at offset 0)
        try:
            filename = header[0:100].rstrip(b'\x00').decode('utf-8', errors='ignore')
            if filename and ('csv' in filename or 'states' in filename):
                info["format"] = "tar (old POSIX)"
                info["tar_filename"] = filename
                # Parse file size from offset 124 (octal)
                size_str = header[124:136].rstrip(b'\x00').decode('ascii', errors='ignore').strip()
                if size_str:
                    info["tar_size"] = str(int(size_str, 8))
        except:
            pass
    
    return info


def run_diagnostic(temp_dir: str):
    """Run diagnostic on files in temp directory."""
    print("\n" + "=" * 70)
    print("FILE DIAGNOSTIC")
    print("=" * 70)
    
    files = [f for f in os.listdir(temp_dir) if f.endswith('.tar')]
    
    if not files:
        print(f"No .tar files found in {temp_dir}")
        return
    
    for filename in files[:5]:  # Check first 5
        filepath = os.path.join(temp_dir, filename)
        info = diagnose_file(filepath)
        print(f"\nFile: {filename}")
        print(f"  Size: {info['size']:,} bytes")
        print(f"  Format: {info['format']}")
        print(f"  Magic (hex): {info['magic_hex']}")
        if info.get('tar_filename'):
            print(f"  Tar filename: {info['tar_filename']}")
        if info.get('tar_size'):
            print(f"  Tar content size: {info['tar_size']} bytes")
        
        # Try to extract and report
        print(f"  Testing extraction...")
        test_dir = os.path.join(temp_dir, "test_extract")
        os.makedirs(test_dir, exist_ok=True)
        result = extract_tar(filepath, test_dir)
        if result:
            for r in result:
                size = os.path.getsize(r) if os.path.exists(r) else 0
                print(f"  → Extracted: {os.path.basename(r)} ({size:,} bytes)")
        else:
            print(f"  → Extraction failed")


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def build_url(date: str, hour: str) -> str:
    """
    Build download URL for a specific date and hour.
    
    URL pattern: https://opensky-network.org/datasets/states/.{date}/{hour}/states_{date}-{hour}.csv.tar
    """
    return f"{BASE_URL}/.{date}/{hour}/states_{date}-{hour}.csv.tar"


def download_file(url: str, output_path: str, retries: int = MAX_RETRIES) -> bool:
    """Download a file with retry logic and progress indication."""
    
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            return True
            
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt + 1}/{retries} after error: {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    FAILED after {retries} attempts: {e}")
                return False
    
    return False


def extract_tar(tar_path: str, extract_dir: str) -> list:
    """Extract CSV files from tar archive. Returns list of extracted files."""
    extracted = []
    
    # Method 1: Try Python's tarfile with various modes
    # OpenSky uses POSIX tar (may not have ustar magic)
    modes_to_try = ['r:', 'r:*', 'r:gz', 'r:bz2', 'r']
    
    for mode in modes_to_try:
        try:
            with tarfile.open(tar_path, mode) as tar:
                members = tar.getmembers()
                
                if not members:
                    continue
                
                for member in members:
                    # Skip directories
                    if not member.isfile():
                        continue
                    
                    # Get filename
                    csv_filename = os.path.basename(member.name)
                    if not csv_filename:
                        csv_filename = f"extracted_{os.path.basename(tar_path).replace('.tar', '')}"
                    
                    # Ensure .csv extension
                    if not csv_filename.endswith('.csv'):
                        csv_filename = csv_filename.replace('.csv.tar', '.csv')
                        if not csv_filename.endswith('.csv'):
                            csv_filename += '.csv'
                    
                    output_path = os.path.join(extract_dir, csv_filename)
                    
                    # Extract file content
                    try:
                        fileobj = tar.extractfile(member)
                        if fileobj is not None:
                            content = fileobj.read()
                            if len(content) > 0:
                                with open(output_path, 'wb') as out:
                                    out.write(content)
                                if os.path.getsize(output_path) > 0:
                                    extracted.append(output_path)
                                    print(f"    Extracted: {csv_filename} ({os.path.getsize(output_path):,} bytes)")
                    except Exception as e:
                        print(f"    extractfile failed: {e}")
                
                if extracted:
                    return extracted
                    
        except tarfile.TarError as e:
            # print(f"    tarfile mode {mode} failed: {e}")
            continue
        except Exception as e:
            # print(f"    tarfile mode {mode} exception: {e}")
            continue
    
    # Method 2: Manual tar extraction (for old POSIX tar without ustar)
    try:
        csv_filename = os.path.basename(tar_path).replace('.csv.tar', '.csv')
        if not csv_filename.endswith('.csv'):
            csv_filename += '.csv'
        output_path = os.path.join(extract_dir, csv_filename)
        
        with open(tar_path, 'rb') as f:
            # Read the tar header (512 bytes)
            header = f.read(512)
            
            if len(header) < 512:
                raise ValueError("File too small for tar header")
            
            # Parse tar header
            # Filename is at offset 0, 100 bytes
            # File size is at offset 124, 12 bytes (octal)
            name = header[0:100].rstrip(b'\x00').decode('utf-8', errors='ignore')
            size_str = header[124:136].rstrip(b'\x00').decode('ascii', errors='ignore').strip()
            
            if size_str:
                file_size = int(size_str, 8)  # Octal
            else:
                # If no size in header, read until null padding
                f.seek(512)
                content = f.read()
                # Find where null padding starts
                null_pos = content.find(b'\x00' * 512)
                if null_pos > 0:
                    file_size = null_pos
                else:
                    file_size = len(content)
                f.seek(512)
            
            # Read the file content
            content = f.read(file_size)
            
            if len(content) > 100:  # Sanity check
                with open(output_path, 'wb') as out:
                    out.write(content)
                
                if os.path.getsize(output_path) > 0:
                    extracted.append(output_path)
                    print(f"    Manual extracted: {csv_filename} ({os.path.getsize(output_path):,} bytes)")
                    return extracted
                    
    except Exception as e:
        print(f"    Manual tar extraction failed: {e}")
    
    # Method 3: Try as gzip
    try:
        csv_filename = os.path.basename(tar_path).replace('.csv.tar', '.csv')
        output_path = os.path.join(extract_dir, csv_filename)
        
        with gzip.open(tar_path, 'rb') as gz:
            content = gz.read()
            if len(content) > 0:
                with open(output_path, 'wb') as out:
                    out.write(content)
                extracted.append(output_path)
                return extracted
    except:
        pass
    
    # Method 4: Raw CSV detection
    try:
        with open(tar_path, 'rb') as f:
            content = f.read(1000)
        
        if b'time' in content or b'icao24' in content:
            csv_filename = os.path.basename(tar_path).replace('.tar', '')
            output_path = os.path.join(extract_dir, csv_filename)
            shutil.copy(tar_path, output_path)
            extracted.append(output_path)
            return extracted
    except:
        pass
    
    # If we got here, all methods failed - print diagnostic info
    try:
        with open(tar_path, 'rb') as f:
            header = f.read(600)
        print(f"    Extraction failed. First 100 bytes: {header[:100]}")
        print(f"    Bytes 257-262 (ustar): {header[257:262]}")
        print(f"    Bytes 0-100 (filename): {header[0:100]}")
    except:
        pass
    
    return extracted


def download_and_extract(date: str, hour: str, output_dir: str, temp_dir: str) -> dict:
    """Download a single hour's data and extract CSV."""
    
    url = build_url(date, hour)
    tar_filename = f"states_{date}-{hour}.csv.tar"
    tar_path = os.path.join(temp_dir, tar_filename)
    
    result = {
        "date": date,
        "hour": hour,
        "success": False,
        "csv_file": None,
        "error": None,
    }
    
    # Download
    if not download_file(url, tar_path):
        result["error"] = "Download failed"
        return result
    
    # Check file size and magic bytes for debugging
    try:
        file_size = os.path.getsize(tar_path)
        with open(tar_path, 'rb') as f:
            magic = f.read(300)  # Need 262+ bytes for tar magic
        
        # Identify file type by magic bytes
        file_type = "unknown"
        if magic[:2] == b'\x1f\x8b':
            file_type = "gzip"
        elif magic[257:262] == b'ustar':
            file_type = "tar (ustar)"
        elif magic[:2] == b'BZ':
            file_type = "bzip2"
        elif magic[:4] == b'PK\x03\x04':
            file_type = "zip"
        elif b'time' in magic and b'icao24' in magic:
            file_type = "csv (plain)"
        else:
            # Check for old POSIX tar (filename at offset 0)
            try:
                filename = magic[0:100].rstrip(b'\x00').decode('utf-8', errors='ignore')
                if filename and ('csv' in filename or 'states' in filename):
                    file_type = "tar (POSIX)"
            except:
                pass
            
    except Exception as e:
        result["error"] = f"File check failed: {e}"
        return result
    
    # Extract
    extracted = extract_tar(tar_path, output_dir)
    
    if extracted:
        result["success"] = True
        result["csv_file"] = extracted[0]
        
        # Clean up tar file on success
        try:
            os.remove(tar_path)
        except:
            pass
    else:
        result["error"] = f"Extraction failed (type={file_type}, size={file_size})"
        # Keep tar file for diagnostic - don't delete
    
    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_downloader(output_dir: str, parallel: bool = True, keep_tar: bool = False):
    """Main download pipeline."""
    
    print("=" * 70)
    print("OPENSKY STATE VECTOR DOWNLOADER")
    print("=" * 70)
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, ".temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Build task list
    tasks = []
    for date in TARGET_DATES:
        for hour in HOURS:
            tasks.append((date, hour))
    
    total_tasks = len(tasks)
    print(f"\nConfiguration:")
    print(f"  Output:     {output_dir}")
    print(f"  Dates:      {len(TARGET_DATES)} days")
    print(f"  Hours:      24 per day")
    print(f"  Total:      {total_tasks} files (~{total_tasks * 130}MB compressed)")
    print(f"  Parallel:   {MAX_WORKERS if parallel else 1} workers")
    
    print(f"\nTarget dates:")
    for d in TARGET_DATES:
        print(f"  - {d}")
    
    print("\n" + "-" * 70)
    print("Starting downloads...")
    print("-" * 70)
    
    # Track results
    completed = 0
    successful = 0
    failed = []
    
    start_time = time.time()
    
    if parallel:
        # Parallel downloads
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_and_extract, date, hour, output_dir, temp_dir): (date, hour)
                for date, hour in tasks
            }
            
            for future in as_completed(futures):
                date, hour = futures[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result["success"]:
                        successful += 1
                        print(f"[{completed}/{total_tasks}] ✓ {date} {hour}:00")
                    else:
                        failed.append((date, hour, result["error"]))
                        print(f"[{completed}/{total_tasks}] ✗ {date} {hour}:00 - {result['error']}")
                except Exception as e:
                    failed.append((date, hour, str(e)))
                    print(f"[{completed}/{total_tasks}] ✗ {date} {hour}:00 - {e}")
    else:
        # Sequential downloads (for debugging)
        for date, hour in tasks:
            completed += 1
            print(f"[{completed}/{total_tasks}] Downloading {date} {hour}:00...", end=" ", flush=True)
            
            result = download_and_extract(date, hour, output_dir, temp_dir)
            
            if result["success"]:
                successful += 1
                print("✓")
            else:
                failed.append((date, hour, result["error"]))
                print(f"✗ {result['error']}")
    
    # Cleanup temp directory (keep if failures or --keep-tar)
    if not keep_tar and not failed:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
    else:
        print(f"\nTemp files kept in: {temp_dir}")
        print("Run diagnostic: python download_opensky.py --diagnose " + temp_dir)
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}/{total_tasks}")
    print(f"Failed:     {len(failed)}/{total_tasks}")
    print(f"Time:       {elapsed/60:.1f} minutes")
    print(f"Output:     {output_dir}")
    
    if failed:
        print(f"\nFailed downloads:")
        for date, hour, error in failed[:10]:
            print(f"  - {date} {hour}:00: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # List output files
    csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
    print(f"\nCSV files in output: {len(csv_files)}")
    
    return successful == total_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenSky state vector data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Downloads 7 days of hourly state vector data from OpenSky's public repository.

Dates: 2019-07-08, 2019-07-15, 2019-07-22, 2019-07-29, 2019-08-05, 2019-08-12, 2019-08-19
Total: 168 files (~22GB compressed, ~50GB+ extracted)

Example:
    python download_opensky.py --output data/raw/opensky
    
If extraction fails, run diagnostic:
    python download_opensky.py --diagnose data/raw/opensky/.temp
        """
    )
    parser.add_argument("--output", default="data/raw/opensky",
                        help="Output directory for CSV files")
    parser.add_argument("--sequential", action="store_true",
                        help="Download sequentially instead of parallel (slower but easier to debug)")
    parser.add_argument("--diagnose", metavar="DIR",
                        help="Run diagnostic on downloaded tar files in specified directory")
    parser.add_argument("--keep-tar", action="store_true",
                        help="Keep tar files after extraction (for debugging)")
    
    args = parser.parse_args()
    
    if args.diagnose:
        run_diagnostic(args.diagnose)
        return
    
    success = run_downloader(args.output, parallel=not args.sequential, keep_tar=args.keep_tar)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()