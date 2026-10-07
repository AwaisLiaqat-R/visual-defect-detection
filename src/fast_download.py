"""
Parallel multi-threaded downloader for Kaggle datasets.
Uses HTTP Range requests across 8 parallel threads to achieve maximum download speed.
"""
import os
import sys
import time
import zipfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import kagglehub
from src.config import KAGGLE_DATASET_ID, RAW_DATA_DIR
from src.utils import setup_logger

logger = setup_logger("fast_downloader")

def get_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    return session

def download_range(url: str, start: int, end: int, part_file: Path, chunk_id: int):
    """Downloads a specific byte range to a part file."""
    headers = {"Range": f"bytes={start}-{end}"}
    session = get_session()
    with session.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(part_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return chunk_id

def parallel_download_dataset(dataset_id: str = KAGGLE_DATASET_ID, num_threads: int = 8) -> Path:
    """
    Downloads dataset using parallel HTTP chunk streams and extracts archive.
    """
    logger.info(f"Resolving download URL for {dataset_id}...")
    dataset_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"

    session = get_session()
    head_resp = session.get(dataset_url, stream=True, allow_redirects=True, timeout=30)
    final_url = head_resp.url
    total_size = int(head_resp.headers.get("Content-Length", 0))
    head_resp.close()
    
    logger.info(f"Resolved URL: {final_url[:80]}... Total size: {total_size / (1024*1024):.2f} MB")

    cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "vadimzavadskyi" / "kolektorsdd2-ksdd2" / "1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_file = cache_dir.parent / "1.archive"

    # Check if already extracted
    extracted_check = list(cache_dir.rglob("*.png")) + list(cache_dir.rglob("*.jpg"))
    if len(extracted_check) > 50:
        logger.info(f"Dataset already extracted at: {cache_dir} ({len(extracted_check)} images)")
        return cache_dir

    if total_size <= 0:
        logger.warning("Could not determine Content-Length; falling back to kagglehub standard download...")
        return Path(kagglehub.dataset_download(dataset_id))

    chunk_size = total_size // num_threads
    temp_dir = cache_dir.parent / "temp_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ranges = []
    for i in range(num_threads):
        start = i * chunk_size
        end = total_size - 1 if i == num_threads - 1 else (start + chunk_size - 1)
        part_path = temp_dir / f"part_{i}.tmp"
        ranges.append((start, end, part_path, i))

    logger.info(f"Downloading {total_size / (1024*1024):.2f} MB across {num_threads} parallel threads...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(download_range, final_url, start, end, part_path, i): i
            for start, end, part_path, i in ranges
        }
        for future in as_completed(futures):
            part_id = futures[future]
            try:
                future.result()
                logger.info(f"✓ Chunk {part_id + 1}/{num_threads} completed.")
            except Exception as e:
                logger.error(f"Error in chunk {part_id}: {e}")
                raise

    elapsed = time.time() - t0
    speed_mbps = (total_size / (1024*1024)) / max(elapsed, 0.1)
    logger.info(f"Parallel download finished in {elapsed:.1f}s ({speed_mbps:.2f} MB/s)!")

    # Assemble archive
    logger.info(f"Merging chunks into {archive_file}...")
    with open(archive_file, "wb") as outfile:
        for i in range(num_threads):
            part_path = temp_dir / f"part_{i}.tmp"
            with open(part_path, "rb") as infile:
                shutil.copyfileobj(infile, outfile, length=4*1024*1024)
            part_path.unlink()
    try:
        temp_dir.rmdir()
    except Exception:
        pass

    # Extract archive
    logger.info(f"Extracting archive to {cache_dir}...")
    with zipfile.ZipFile(archive_file, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)

    logger.info(f"Extraction complete. Dataset ready at {cache_dir}")
    return cache_dir

if __name__ == "__main__":
    parallel_download_dataset()
