"""
Operating threshold optimization for industrial defect detection.
Finds optimal decision threshold balancing False Positives vs False Negatives based on business costs and F_beta metrics.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import (
    CHECKPOINT_DIR, ARTIFACT_DIR, COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE, BETA_F_SCORE
)
from src.utils import setup_logger

logger = setup_logger("threshold_optimizer")

def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    beta: float = BETA_F_SCORE,
    output_plot_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Sweeps decision threshold from 0.01 to 0.99 and evaluates:
    1. Business Cost: (cost_fn * FN) + (cost_fp * FP)
    2. F_beta Score (e.g. F2-score prioritizing recall)
    3. F1 Score
    4. Precision vs Recall trade-offs
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        
        tp = int(np.sum((y_true == 1) & (preds == 1)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        fn = int(np.sum((y_true == 1) & (preds == 0)))
        tn = int(np.sum((y_true == 0) & (preds == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # F_beta
        beta_sq = beta ** 2
        f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall) if (beta_sq * precision + recall) > 0 else 0.0

        # Manufacturing cost
        total_cost = (cost_fn * fn) + (cost_fp * fp)

        results.append({
            "threshold": float(t),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            f"f_{beta:.0f}": float(f_beta),
            "business_cost": float(total_cost)
        })

    df_res = pd.DataFrame(results)

    # 1. Best by Cost Minimization
    min_cost_idx = df_res["business_cost"].idxmin()
    best_cost_row = df_res.loc[min_cost_idx]

    # 2. Best by F_beta
    best_fbeta_idx = df_res[f"f_{beta:.0f}"].idxmax()
    best_fbeta_row = df_res.loc[best_fbeta_idx]

    # 3. Standard default (0.50)
    idx_50 = (df_res["threshold"] - 0.50).abs().idxmin()
    default_row = df_res.loc[idx_50]

    recommended_threshold = float(best_fbeta_row["threshold"])

    logger.info("=== Decision Threshold Trade-off Analysis ===")
    logger.info(f"Default (0.50)      -> Rec: {default_row['recall']*100:.1f}%, Prec: {default_row['precision']*100:.1f}%, FP: {int(default_row['fp'])}, FN: {int(default_row['fn'])}, Cost: ${default_row['business_cost']:.1f}")
    logger.info(f"Cost-Optimal ({best_cost_row['threshold']:.2f}) -> Rec: {best_cost_row['recall']*100:.1f}%, Prec: {best_cost_row['precision']*100:.1f}%, FP: {int(best_cost_row['fp'])}, FN: {int(best_cost_row['fn'])}, Cost: ${best_cost_row['business_cost']:.1f}")
    logger.info(f"F{beta:.0f}-Optimal ({best_fbeta_row['threshold']:.2f})   -> Rec: {best_fbeta_row['recall']*100:.1f}%, Prec: {best_fbeta_row['precision']*100:.1f}%, FP: {int(best_fbeta_row['fp'])}, FN: {int(best_fbeta_row['fn'])}, Cost: ${best_fbeta_row['business_cost']:.1f}")

    if output_plot_path:
        plot_threshold_curves(df_res, recommended_threshold, beta, output_plot_path)

    return {
        "recommended_threshold": recommended_threshold,
        "cost_optimal_threshold": float(best_cost_row["threshold"]),
        "f_beta_optimal_threshold": float(best_fbeta_row["threshold"]),
        "cost_optimal_metrics": best_cost_row.to_dict(),
        "f_beta_optimal_metrics": best_fbeta_row.to_dict(),
        "default_metrics": default_row.to_dict(),
        "sweep_dataframe": df_res
    }

def plot_threshold_curves(df_res: pd.DataFrame, chosen_threshold: float, beta: float, output_path: Path) -> None:
    """Plots threshold vs metrics and threshold vs cost curves."""
    plt.figure(figsize=(13, 5), dpi=300)

    # Subplot 1: Metrics vs Threshold
    plt.subplot(1, 2, 1)
    plt.plot(df_res["threshold"], df_res["precision"], 'b-', label="Precision", linewidth=2)
    plt.plot(df_res["threshold"], df_res["recall"], 'r-', label="Recall (Defect Capture)", linewidth=2)
    plt.plot(df_res["threshold"], df_res["f1"], 'g--', label="F1-Score", linewidth=1.5)
    plt.plot(df_res["threshold"], df_res[f"f_{beta:.0f}"], 'm-.', label=f"F{beta:.0f}-Score", linewidth=2)
    plt.axvline(chosen_threshold, color='black', linestyle=':', label=f"Operating Threshold ({chosen_threshold:.2f})")
    plt.title("Metrics vs. Operating Threshold", fontweight='bold')
    plt.xlabel("Decision Threshold (Defective Class)")
    plt.ylabel("Score")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="best")

    # Subplot 2: Manufacturing Business Cost vs Threshold
    plt.subplot(1, 2, 2)
    plt.plot(df_res["threshold"], df_res["business_cost"], 'purple', linewidth=2.5, label="Total Cost ($)")
    plt.axvline(chosen_threshold, color='black', linestyle=':', label=f"Selected Threshold ({chosen_threshold:.2f})")
    plt.title("Manufacturing Cost vs. Threshold", fontweight='bold')
    plt.xlabel("Decision Threshold (Defective Class)")
    plt.ylabel("Total Expected Cost ($)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="best")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved threshold optimization plot to {output_path}")
