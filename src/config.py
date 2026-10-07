"""
Configuration management for Defect Detection Pipeline.
Centralizes all paths, hyperparameters, dataset configs, and deployment settings.
"""
from pathlib import Path
import os
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, ARTIFACT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Kaggle Dataset Identifier
KAGGLE_DATASET_ID = "vadimzavadskyi/kolektorsdd2-ksdd2"

# Reproducibility
RANDOM_SEED = 42

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # 0 for safe multiprocessing on Windows

# Image Preprocessing Configuration
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Dataset Splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 4

# Class Labels
CLASS_NAMES = ["normal", "defective"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Model paths
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
BASELINE_MODEL_PATH = CHECKPOINT_DIR / "baseline_model.pth"
METADATA_PATH = CHECKPOINT_DIR / "model_metadata.json"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_IMAGE_SIZE_MB = 10
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

# Business Metric Weights (Cost of FN vs FP in manufacturing)
COST_FALSE_NEGATIVE = 10.0  # Missing a defect leads to defective product shipped
COST_FALSE_POSITIVE = 1.0   # False alarm requires manual reinspection
BETA_F_SCORE = 2.0          # F2-score weights recall higher than precision
