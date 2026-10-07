"""
Evaluation and Error Analysis Pipeline.
Computes test metrics (Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix),
analyzes False Positives vs False Negatives, runs latency benchmarking,
and exports production metadata.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    DEVICE, BEST_MODEL_PATH, BASELINE_MODEL_PATH, METADATA_PATH,
    ARTIFACT_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, CLASS_TO_IDX,
    IDX_TO_CLASS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
)
from src.utils import (
    setup_logger, calculate_metrics, plot_confusion_matrix,
    plot_roc_pr_curves, plot_error_analysis
)
from src.dataset import DefectDataset, get_transforms
from src.models import build_model
from src.threshold_optimizer import find_optimal_threshold

logger = setup_logger("evaluate")

def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = DEVICE,
    decision_threshold: float = 0.5
) -> Dict[str, Any]:
    """Runs inference on test split and calculates comprehensive evaluation metrics."""
    model.eval()
    all_targets = []
    all_probs = []
    all_image_paths = []

    start_bench = time.time()
    total_samples = 0

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_targets.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_image_paths.extend(paths)
            total_samples += labels.size(0)

    total_infer_time = time.time() - start_bench
    avg_latency_ms = (total_infer_time / max(total_samples, 1)) * 1000.0
    fps = total_samples / max(total_infer_time, 1e-6)

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    y_pred = (y_probs >= decision_threshold).astype(int)

    metrics = calculate_metrics(y_true, y_pred, y_probs)
    metrics["avg_latency_ms"] = float(avg_latency_ms)
    metrics["throughput_fps"] = float(fps)
    metrics["decision_threshold"] = float(decision_threshold)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Error analysis sample extraction
    records = []
    false_positives = []
    false_negatives = []

    for i in range(len(y_true)):
        item = {
            "image_path": all_image_paths[i],
            "filename": Path(all_image_paths[i]).name,
            "true_label": int(y_true[i]),
            "true_class": IDX_TO_CLASS[int(y_true[i])],
            "pred_label": int(y_pred[i]),
            "pred_class": IDX_TO_CLASS[int(y_pred[i])],
            "prob_defective": float(y_probs[i]),
            "prob_normal": float(1.0 - y_probs[i])
        }
        records.append(item)

        if y_true[i] == 0 and y_pred[i] == 1:
            false_positives.append(item)
        elif y_true[i] == 1 and y_pred[i] == 0:
            false_negatives.append(item)

    # Sort false alarms and missed defects by confidence margin
    false_positives.sort(key=lambda x: x["prob_defective"], reverse=True)
    false_negatives.sort(key=lambda x: x["prob_defective"])

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "records": records,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def run_evaluation_pipeline(
    model_type: str = "transfer",
    checkpoint_path: Optional[Path] = None,
    save_production_metadata: bool = True
) -> Dict[str, Any]:
    """Complete evaluation and diagnostics workflow."""
    manifest_path = PROCESSED_DATA_DIR / "dataset_splits.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset splits at {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    test_df = manifest_df[manifest_df["split"] == "test"]
    val_df = manifest_df[manifest_df["split"] == "val"]

    test_dataset = DefectDataset(test_df, transform=get_transforms(is_training=False))
    val_dataset = DefectDataset(val_df, transform=get_transforms(is_training=False))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH if model_type == "transfer" else BASELINE_MODEL_PATH

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    logger.info(f"Loading checkpoint {checkpoint_path} for evaluation on {DEVICE}...")
    model = build_model(model_type=model_type, num_classes=2, pretrained=False).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Step 1: Sweep thresholds on Validation set to avoid test leakage
    logger.info("Evaluating on Validation set for threshold optimization...")
    val_eval = evaluate_test_set(model, val_loader, device=DEVICE, decision_threshold=0.50)
    thresh_result = find_optimal_threshold(
        val_eval["y_true"],
        val_eval["y_probs"],
        output_plot_path=ARTIFACT_DIR / "threshold_tradeoffs.png"
    )
    optimal_threshold = thresh_result["recommended_threshold"]
    logger.info(f"★ Selected Optimal Operating Threshold: {optimal_threshold:.2f}")

    # Step 2: Unbiased Evaluation on Test set using both default (0.50) and optimal threshold
    logger.info("Evaluating on Unseen Test Set with standard threshold (0.50)...")
    eval_default = evaluate_test_set(model, test_loader, device=DEVICE, decision_threshold=0.50)
    
    logger.info(f"Evaluating on Unseen Test Set with Optimal Threshold ({optimal_threshold:.2f})...")
    eval_optimal = evaluate_test_set(model, test_loader, device=DEVICE, decision_threshold=optimal_threshold)

    m = eval_optimal["metrics"]
    logger.info("================ TEST EVALUATION RESULTS ================")
    logger.info(f"Accuracy:         {m['accuracy']*100:.2f}%")
    logger.info(f"Defect Precision: {m['precision']*100:.2f}%")
    logger.info(f"Defect Recall:    {m['recall']*100:.2f}%")
    logger.info(f"Defect F1-Score:  {m['f1']:.4f}")
    logger.info(f"Defect F2-Score:  {m.get('f2', 0.0):.4f}")
    if "roc_auc" in m:
        logger.info(f"ROC-AUC:          {m['roc_auc']:.4f}")
        logger.info(f"PR-AUC:           {m['pr_auc']:.4f}")
    logger.info(f"Avg CPU Latency:  {m['avg_latency_ms']:.2f} ms / image ({m['throughput_fps']:.1f} FPS)")
    logger.info(f"False Positives (Alarms): {len(eval_optimal['false_positives'])}")
    logger.info(f"False Negatives (Missed): {len(eval_optimal['false_negatives'])}")
    logger.info("=========================================================")

    # Generate Visual Artifacts
    cm_path = ARTIFACT_DIR / f"{model_type}_confusion_matrix.png"
    plot_confusion_matrix(eval_optimal["confusion_matrix"], CLASS_NAMES, cm_path, title=f"Confusion Matrix ({model_type.upper()} @ Threshold {optimal_threshold:.2f})")

    curves_path = ARTIFACT_DIR / f"{model_type}_roc_pr_curves.png"
    plot_roc_pr_curves(eval_optimal["y_true"], eval_optimal["y_probs"], curves_path)

    error_path = ARTIFACT_DIR / f"{model_type}_error_analysis.png"
    plot_error_analysis(eval_optimal["false_positives"], eval_optimal["false_negatives"], CLASS_NAMES, error_path)

    # Save detailed test predictions CSV
    test_df_export = pd.DataFrame(eval_optimal["records"])
    test_df_export.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    logger.info(f"Saved test predictions table to {ARTIFACT_DIR / 'test_predictions.csv'}")

    # Save production model metadata JSON
    if save_production_metadata and model_type == "transfer":
        metadata = {
            "model_name": "ResNet18-Transfer-Defect-Classifier",
            "model_type": model_type,
            "architecture": "ResNet-18 ImageNet Pretrained Backbone + Fine-Tuned Head",
            "num_classes": 2,
            "classes": CLASS_NAMES,
            "class_to_idx": CLASS_TO_IDX,
            "image_size": IMAGE_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "operating_threshold": float(optimal_threshold),
            "test_metrics": eval_optimal["metrics"],
            "confusion_matrix": eval_optimal["confusion_matrix"].tolist(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Production model metadata saved to {METADATA_PATH}")

    return {
        "eval_default": eval_default,
        "eval_optimal": eval_optimal,
        "optimal_threshold": optimal_threshold
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate defect detection model.")
    parser.add_argument("--model", type=str, default="transfer", choices=["baseline", "transfer"])
    args = parser.parse_args()

    run_evaluation_pipeline(model_type=args.model)
