"""
Utility functions for reproducibility, metrics calculation, and publication-quality visualization.
"""
import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution and server environments
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from PIL import Image
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

def setup_logger(name: str = "DefectDetection", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized console and optional file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

logger = setup_logger("utils")

def set_seed(seed: int = 42) -> None:
    """Enforces determinism across Python, NumPy, PyTorch CPU and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Reproducibility seed set to {seed}")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculates precision, recall, F1, accuracy, and optional AUC metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    # Calculate F2-score (recall weighted 2x higher than precision for defect detection)
    p = metrics["precision"]
    r = metrics["recall"]
    if (4 * p + r) > 0:
        metrics["f2"] = float((1 + 4) * (p * r) / (4 * p + r))
    else:
        metrics["f2"] = 0.0

    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_probs))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_probs))
        except Exception as e:
            logger.warning(f"Could not compute AUC scores: {e}")
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
    return metrics

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path, title: str = "Defect Detection Confusion Matrix") -> None:
    """Generates and saves an annotated confusion matrix plot."""
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Class", fontsize=10, labelpad=8)
    plt.ylabel("Actual Ground Truth", fontsize=10, labelpad=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")

def plot_training_history(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plots training and validation loss and F1-score across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5), dpi=300)
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-o', label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], 'r--s', label="Val Loss", linewidth=2)
    plt.title("Loss vs. Epochs", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # F1 score subplot
    plt.subplot(1, 2, 2)
    if "val_f1" in history:
        plt.plot(epochs, history["val_f1"], 'g-^', label="Val Defect F1", linewidth=2)
    if "val_recall" in history:
        plt.plot(epochs, history["val_recall"], 'm--d', label="Val Defect Recall", linewidth=2)
    plt.title("Validation Metrics vs. Epochs", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training history curves to {output_path}")

def plot_roc_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, output_path: Path) -> None:
    """Plots ROC Curve and Precision-Recall Curve side by side."""
    if len(np.unique(y_true)) <= 1:
        logger.warning("Only 1 class present in y_true, skipping ROC/PR curves.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(12, 5), dpi=300)

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC)", fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right")

    # PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals, color="purple", lw=2, label=f"PR curve (AP = {pr_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC/PR curves to {output_path}")

def plot_error_analysis(
    false_positives: List[Dict[str, Any]],
    false_negatives: List[Dict[str, Any]],
    class_names: List[str],
    output_path: Path,
    max_images_per_type: int = 4
) -> None:
    """Generates visual analysis grid for False Positives (False Alarms) and False Negatives (Missed Defects)."""
    total_fp = len(false_positives)
    total_fn = len(false_negatives)
    num_fp = min(total_fp, max_images_per_type)
    num_fn = min(total_fn, max_images_per_type)

    if num_fp == 0 and num_fn == 0:
        logger.info("No errors to visualize.")
        return

    cols = max(num_fp, num_fn, 1)
    rows = (1 if num_fp > 0 else 0) + (1 if num_fn > 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), dpi=300, squeeze=False)
    current_row = 0

    if num_fp > 0:
        for col_idx in range(cols):
            ax = axes[current_row][col_idx]
            if col_idx < num_fp:
                item = false_positives[col_idx]
                try:
                    img = Image.open(item["image_path"]).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(
                        f"False Positive (False Alarm)\nTrue: {class_names[item['true_label']]} | Pred: {class_names[item['pred_label']]}\nDefect Prob: {item['prob_defective']:.2f}",
                        fontsize=8, color='red'
                    )
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading image:\n{e}", ha='center', va='center', fontsize=7)
            ax.axis('off')
        current_row += 1

    if num_fn > 0:
        for col_idx in range(cols):
            ax = axes[current_row][col_idx]
            if col_idx < num_fn:
                item = false_negatives[col_idx]
                try:
                    img = Image.open(item["image_path"]).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(
                        f"False Negative (Missed Defect!)\nTrue: {class_names[item['true_label']]} | Pred: {class_names[item['pred_label']]}\nDefect Prob: {item['prob_defective']:.2f}",
                        fontsize=8, color='darkred', fontweight='bold'
                    )
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading image:\n{e}", ha='center', va='center', fontsize=7)
            ax.axis('off')

    plt.suptitle(f"Error Analysis: {total_fp} False Positives, {total_fn} False Negatives", fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved error analysis visualization to {output_path}")
