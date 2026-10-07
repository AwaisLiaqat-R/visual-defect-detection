"""
Dataset downloading, auditing, YOLO/mask annotation parsing, stratified splitting,
and PyTorch Dataset & Transform pipelines.
"""
import os
import sys
import glob
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    KAGGLE_DATASET_ID, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    RANDOM_SEED, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE, NUM_WORKERS,
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
)
from src.utils import setup_logger

logger = setup_logger("dataset")

def download_dataset(dataset_id: str = KAGGLE_DATASET_ID) -> Path:
    """
    Downloads the dataset from KaggleHub and returns the local cache path.
    """
    logger.info(f"Downloading Kaggle dataset: {dataset_id}...")
    import kagglehub
    download_path = kagglehub.dataset_download(dataset_id)
    download_path = Path(download_path)
    logger.info(f"Dataset successfully downloaded to: {download_path}")
    return download_path

def audit_raw_dataset(raw_dir: Path) -> Dict[str, Any]:
    """
    Recursively audits raw dataset directory, counting files, formats, annotations, and sizes.
    """
    logger.info(f"Auditing raw dataset directory: {raw_dir}")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    all_files = list(raw_dir.rglob("*"))
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in image_extensions]
    text_files = [f for f in all_files if f.is_file() and f.suffix.lower() == ".txt"]
    gt_mask_files = [f for f in all_files if f.is_file() and ("_gt" in f.stem.lower() or "_mask" in f.stem.lower())]

    audit_summary = {
        "raw_directory": str(raw_dir),
        "total_files": len(all_files),
        "total_images": len(image_files),
        "total_txt_annotations": len(text_files),
        "total_gt_masks": len(gt_mask_files),
        "subdirectories": [str(p.relative_to(raw_dir)) for p in raw_dir.iterdir() if p.is_dir()],
    }

    # Sample dimensions inspection
    sample_sizes = []
    for img_path in image_files[:min(50, len(image_files))]:
        try:
            with Image.open(img_path) as img:
                sample_sizes.append(img.size)
        except Exception as e:
            logger.warning(f"Could not open image {img_path}: {e}")

    if sample_sizes:
        widths, heights = zip(*sample_sizes)
        audit_summary["sample_resolutions"] = {
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "sample_count": len(sample_sizes)
        }

    logger.info(f"Audit Complete: Found {len(image_files)} images, {len(text_files)} txt files, {len(gt_mask_files)} masks.")
    return audit_summary

def convert_annotations_to_labels(raw_dir: Path, output_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Converts diverse dataset annotations (YOLO format .txt, segmentation masks _GT.png, or folder labels)
    into a standardized classification DataFrame:
        - 0: normal
        - 1: defective
    """
    logger.info(f"Parsing annotations and labeling dataset in {raw_dir}...")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Exclude mask images from being treated as primary samples
    def is_mask_file(p: Path) -> bool:
        stem = p.stem.lower()
        return stem.endswith("_gt") or stem.endswith("_mask") or "_label" in stem

    raw_images = [
        f for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions and not is_mask_file(f)
    ]

    logger.info(f"Found {len(raw_images)} primary product images.")

    records = []
    for img_path in raw_images:
        label = 0 # Default: Normal
        annotation_source = "default_normal"

        # Check 1: Folder name heuristic
        parts = [p.lower() for p in img_path.parts]
        if any(keyword in parts for keyword in ["defective", "defect", "ng", "positive", "bad", "anomaly"]):
            label = 1
            annotation_source = "directory_name_defect"
        elif any(keyword in parts for keyword in ["normal", "ok", "good", "negative", "clean"]):
            label = 0
            annotation_source = "directory_name_normal"

        # Check 2: YOLO annotation .txt file in same directory or parallel 'labels' directory
        txt_candidate_same = img_path.with_suffix(".txt")
        # parallel labels dir candidate
        txt_candidate_labels = None
        if "images" in parts:
            idx = parts.index("images")
            labels_parts = list(img_path.parts)
            labels_parts[idx] = "labels"
            txt_candidate_labels = Path(*labels_parts).with_suffix(".txt")

        target_txt = None
        if txt_candidate_same.exists():
            target_txt = txt_candidate_same
        elif txt_candidate_labels and txt_candidate_labels.exists():
            target_txt = txt_candidate_labels

        if target_txt and target_txt.exists():
            try:
                with open(target_txt, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                if len(lines) > 0:
                    label = 1
                    annotation_source = f"yolo_txt_boxes({len(lines)})"
                else:
                    label = 0
                    annotation_source = "yolo_txt_empty"
            except Exception as e:
                logger.warning(f"Error reading annotation {target_txt}: {e}")

        # Check 3: Ground Truth Mask (_GT.png or _gt.png)
        gt_mask_candidate = img_path.parent / f"{img_path.stem}_GT.png"
        if not gt_mask_candidate.exists():
            gt_mask_candidate = img_path.parent / f"{img_path.stem}_gt.png"
        if not gt_mask_candidate.exists():
            gt_mask_candidate = img_path.parent / f"{img_path.stem}_GT.PNG"

        if gt_mask_candidate.exists():
            try:
                with Image.open(gt_mask_candidate) as mask_img:
                    mask_np = np.array(mask_img)
                    if np.max(mask_np) > 0:
                        label = 1
                        annotation_source = f"gt_mask_defect(max_val={np.max(mask_np)})"
                    else:
                        label = 0
                        annotation_source = "gt_mask_empty"
            except Exception as e:
                logger.warning(f"Error reading GT mask {gt_mask_candidate}: {e}")

        records.append({
            "image_path": str(img_path.resolve()),
            "relative_path": str(img_path.relative_to(raw_dir)),
            "filename": img_path.name,
            "label": int(label),
            "class_name": IDX_TO_CLASS[int(label)],
            "annotation_source": annotation_source
        })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} total items. Class distribution:\n{df['class_name'].value_counts()}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved manifest to {output_csv}")

    return df

def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    """
    Creates stratified Train / Validation / Test splits with guaranteed class preservation.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    logger.info(f"Creating stratified splits ({train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%)...")

    # If dataset has natural train/test partition from original directory structure, we can also inspect it
    indices = np.arange(len(df))
    labels = df["label"].values

    # Check if a class has < 2 samples
    class_counts = df["label"].value_counts()
    if any(class_counts < 2):
        raise ValueError(f"Each class must have at least 2 samples for stratified split. Counts: {class_counts.to_dict()}")

    # First split: train vs temp (val + test)
    temp_size = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_size,
        random_state=random_seed,
        stratify=labels
    )

    # Second split: val vs test
    val_prop_of_temp = val_ratio / temp_size
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_prop_of_temp,
        random_state=random_seed,
        stratify=temp_labels
    )

    df_copy = df.copy()
    df_copy["split"] = "unassigned"
    df_copy.loc[train_idx, "split"] = "train"
    df_copy.loc[val_idx, "split"] = "val"
    df_copy.loc[test_idx, "split"] = "test"

    for split_name in ["train", "val", "test"]:
        sub_df = df_copy[df_copy["split"] == split_name]
        counts = sub_df["class_name"].value_counts().to_dict()
        logger.info(f"Split [{split_name.upper()}]: Total={len(sub_df)}, Classes={counts}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_copy.to_csv(output_csv, index=False)
        logger.info(f"Saved stratified manifest to {output_csv}")

    return df_copy

def get_transforms(image_size: int = IMAGE_SIZE, is_training: bool = False) -> transforms.Compose:
    """
    Constructs PyTorch vision transforms.
    Training pipeline includes domain-appropriate augmentations (rotations, flips, subtle color changes).
    Validation/Testing uses deterministic resizing and standard ImageNet normalization.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

class DefectDataset(Dataset):
    """
    Custom PyTorch Dataset for loading visual defect images and ground truth labels.
    """
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label, path
        except Exception as e:
            logger.error(f"Failed to read image at {path}: {e}")
            # Fallback tensor
            fallback = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
            return fallback, label, path

def get_dataloaders(
    manifest_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds PyTorch DataLoaders for train, val, and test splits with proper transforms and batching.
    """
    train_df = manifest_df[manifest_df["split"] == "train"]
    val_df = manifest_df[manifest_df["split"] == "val"]
    test_df = manifest_df[manifest_df["split"] == "test"]

    train_dataset = DefectDataset(train_df, transform=get_transforms(is_training=True))
    val_dataset = DefectDataset(val_df, transform=get_transforms(is_training=False))
    test_dataset = DefectDataset(test_df, transform=get_transforms(is_training=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
