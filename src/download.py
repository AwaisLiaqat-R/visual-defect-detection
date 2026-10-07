"""
Resilient automatic resume downloader for KaggleHub.
Loops and resumes dataset download upon any network timeout/disconnect until 100% complete.
"""
import os
import sys
import time
from pathlib import Path
import kagglehub

from src.config import KAGGLE_DATASET_ID
from src.utils import setup_logger

logger = setup_logger("downloader")

def download_with_resilience(dataset_id: str = KAGGLE_DATASET_ID, max_attempts: int = 50) -> Path:
    logger.info(f"Starting resilient KaggleHub download for '{dataset_id}'...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"[Attempt {attempt}/{max_attempts}] Connecting to Kaggle...")
            path = kagglehub.dataset_download(dataset_id)
            path = Path(path)
            logger.info(f"★★★ Dataset successfully downloaded and extracted to: {path} ★★★")
            return path
        except Exception as e:
            logger.warning(f"[Attempt {attempt}] Connection interrupted: {e}")
            logger.info("Waiting 3 seconds before resuming download...")
            time.sleep(3)
            
    raise RuntimeError(f"Could not complete download after {max_attempts} attempts.")

if __name__ == "__main__":
    download_with_resilience()
