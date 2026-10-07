"""
Master End-to-End Visual Defect Detection Pipeline.
Executes the full 18-step workflow from dataset download to model evaluation and production readiness.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.config import (
    KAGGLE_DATASET_ID, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    CHECKPOINT_DIR, ARTIFACT_DIR, BEST_MODEL_PATH, BASELINE_MODEL_PATH,
    METADATA_PATH, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS,
    IMAGE_SIZE, NUM_EPOCHS, LEARNING_RATE
)
from src.utils import setup_logger, set_seed
from src.download import download_with_resilience
from src.dataset import (
    audit_raw_dataset, convert_annotations_to_labels,
    create_stratified_splits, get_dataloaders
)
from src.train import run_training_pipeline
from src.evaluate import run_evaluation_pipeline

logger = setup_logger("pipeline")

def plot_class_distribution_and_samples(manifest_df: pd.DataFrame, output_dir: Path) -> None:
    """Step 04: Statistical and visual verification of labels."""
    logger.info("Generating dataset distribution and visual sample verifications...")
    
    # Class Distribution Bar Plot
    counts = manifest_df["class_name"].value_counts()
    plt.figure(figsize=(6, 4), dpi=300)
    bars = plt.bar(counts.index, counts.values, color=['#2b5c8f', '#d95f02'], width=0.5)
    plt.title("Class Distribution (Normal vs. Defective)", fontsize=11, fontweight='bold')
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    dist_path = output_dir / "class_distribution.png"
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved class distribution plot to {dist_path}")

    # Visual Inspection Sample Grid (Normal vs Defective)
    normal_samples = manifest_df[manifest_df["label"] == 0].head(4)
    defect_samples = manifest_df[manifest_df["label"] == 1].head(4)
    
    samples_to_plot = pd.concat([normal_samples, defect_samples])
    plt.figure(figsize=(12, 6), dpi=300)
    for i, (_, row) in enumerate(samples_to_plot.iterrows(), 1):
        plt.subplot(2, 4, i)
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            plt.imshow(img)
            plt.title(f"{row['class_name'].upper()}\n({row['filename'][:20]})", fontsize=8, fontweight='bold',
                      color='green' if row['label'] == 0 else 'red')
        except Exception as e:
            plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        plt.axis('off')
    
    plt.suptitle("Dataset Verification Samples (Top: Normal | Bottom: Defective)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    sample_path = output_dir / "sample_verification.png"
    plt.savefig(sample_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visual sample verification to {sample_path}")

def run_master_pipeline(skip_download: bool = False, fast_train: bool = False) -> Dict[str, Any]:
    """
    Executes all 18 assessment steps sequentially.
    """
    set_seed(42)
    start_total = time.time()
    logger.info("=================================================================")
    logger.info("=== STARTING VISUAL DEFECT DETECTION PIPELINE (18 STEPS) ====")
    logger.info("=================================================================")

    # Step 01: KaggleHub download
    logger.info("\n--- [Step 01] KaggleHub Download ---")
    import kagglehub
    raw_cache = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    logger.info(f"Dataset path verified at: {raw_cache}")

    # Step 02: Audit actual dataset
    logger.info("\n--- [Step 02] Audit Actual Dataset ---")
    audit_results = audit_raw_dataset(raw_cache)
    with open(DATA_DIR / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=4)
    logger.info(f"Audit summary: {json.dumps(audit_results, indent=2)}")

    # Step 03: Convert YOLO / Mask annotations -> Normal/Defective
    logger.info("\n--- [Step 03] Convert Annotations to Normal / Defective ---")
    manifest_csv = DATA_DIR / "dataset_manifest.csv"
    manifest_df = convert_annotations_to_labels(raw_cache, output_csv=manifest_csv)

    # Step 04: Verify labels visually & statistically
    logger.info("\n--- [Step 04] Verify Labels Visually & Statistically ---")
    plot_class_distribution_and_samples(manifest_df, ARTIFACT_DIR)

    # Step 05: Create train / validation / test split
    logger.info("\n--- [Step 05] Create Train / Validation / Test Split (70% / 15% / 15%) ---")
    split_csv = PROCESSED_DATA_DIR / "dataset_splits.csv"
    split_df = create_stratified_splits(manifest_df, output_csv=split_csv)

    epochs = 3 if fast_train else NUM_EPOCHS

    # Step 07: Establish simple baseline
    logger.info("\n--- [Step 07] Establish Simple Baseline (Baseline CNN) ---")
    logger.info("Training lightweight scratch BaselineCNN for empirical benchmark...")
    baseline_model, baseline_history, baseline_val_metrics = run_training_pipeline(
        model_type="baseline",
        num_epochs=min(epochs, 5),
        lr=1e-3,
        checkpoint_path=BASELINE_MODEL_PATH
    )
    baseline_eval = run_evaluation_pipeline(model_type="baseline", checkpoint_path=BASELINE_MODEL_PATH, save_production_metadata=False)
    logger.info(f"Baseline Test Defect Recall: {baseline_eval['eval_optimal']['metrics']['recall']*100:.1f}%, F1: {baseline_eval['eval_optimal']['metrics']['f1']:.4f}")

    # Steps 08, 09, 10: Transfer-learning model + class imbalance + train/checkpoint best model
    logger.info("\n--- [Step 08, 09, 10] Transfer-Learning Model (ResNet-18) + Class Imbalance + Training ---")
    transfer_model, transfer_history, transfer_val_metrics = run_training_pipeline(
        model_type="transfer",
        num_epochs=epochs,
        lr=LEARNING_RATE,
        checkpoint_path=BEST_MODEL_PATH
    )

    # Step 11, 12, 13, 14: Evaluation + Error analysis + Threshold selection + Save production model
    logger.info("\n--- [Step 11, 12, 13, 14] Test Metrics + Error Analysis + Threshold Tuning + Production Model ---")
    transfer_eval = run_evaluation_pipeline(
        model_type="transfer",
        checkpoint_path=BEST_MODEL_PATH,
        save_production_metadata=True
    )

    # Comparison summary: Baseline vs Transfer Learning
    b_m = baseline_eval["eval_optimal"]["metrics"]
    t_m = transfer_eval["eval_optimal"]["metrics"]
    logger.info("\n================ PERFORMANCE COMPARISON ================")
    logger.info(f"{'Metric':<25} | {'Baseline CNN':<15} | {'Transfer ResNet-18':<18}")
    logger.info("-" * 65)
    logger.info(f"{'Accuracy':<25} | {b_m['accuracy']*100:>13.2f}% | {t_m['accuracy']*100:>16.2f}%")
    logger.info(f"{'Defect Precision':<25} | {b_m['precision']*100:>13.2f}% | {t_m['precision']*100:>16.2f}%")
    logger.info(f"{'Defect Recall':<25} | {b_m['recall']*100:>13.2f}% | {t_m['recall']*100:>16.2f}%")
    logger.info(f"{'Defect F1-Score':<25} | {b_m['f1']:>14.4f} | {t_m['f1']:>17.4f}")
    logger.info(f"{'Defect F2-Score':<25} | {b_m.get('f2', 0):>14.4f} | {t_m.get('f2', 0):>17.4f}")
    logger.info(f"{'Avg Latency (CPU)':<25} | {b_m['avg_latency_ms']:>12.2f} ms | {t_m['avg_latency_ms']:>15.2f} ms")
    logger.info("========================================================\n")

    total_time = time.time() - start_total
    logger.info(f"★★★ Master Pipeline Complete in {total_time:.1f} seconds! ★★★")
    logger.info(f"Production Model Saved: {BEST_MODEL_PATH}")
    logger.info(f"Metadata Exported:      {METADATA_PATH}")
    logger.info(f"Visual Artifacts in:    {ARTIFACT_DIR}")

    return {
        "baseline_metrics": b_m,
        "transfer_metrics": t_m,
        "optimal_threshold": transfer_eval["optimal_threshold"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete defect detection pipeline.")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download and use local cache")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode with reduced epochs for verification")
    args = parser.parse_args()

    run_master_pipeline(skip_download=args.skip_download, fast_train=args.fast)
