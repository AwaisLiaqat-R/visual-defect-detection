"""
Training engine with class-imbalance weighting, early stopping,
checkpointing, and multi-metric validation.
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    CHECKPOINT_DIR, BEST_MODEL_PATH, BASELINE_MODEL_PATH,
    EARLY_STOPPING_PATIENCE, PROCESSED_DATA_DIR, RANDOM_SEED,
    CLASS_NAMES
)
from src.utils import setup_logger, set_seed, calculate_metrics, plot_training_history
from src.dataset import get_dataloaders
from src.models import build_model

logger = setup_logger("train")

def compute_class_weights(train_df: pd.DataFrame, device: str = DEVICE) -> torch.Tensor:
    """
    Computes inverse frequency class weights from training set only to prevent leakage:
        weight_c = N / (num_classes * count_c)
    """
    class_counts = train_df["label"].value_counts().sort_index().values
    num_samples = len(train_df)
    num_classes = len(class_counts)
    
    weights = num_samples / (num_classes * class_counts.astype(np.float32))
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    
    tensor_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Computed Class Weights: Normal={tensor_weights[0]:.3f}, Defective={tensor_weights[1]:.3f}")
    return tensor_weights

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc

def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates the model on a DataLoader without updating weights."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = calculate_metrics(y_true, y_pred, y_probs)
    return epoch_loss, metrics, y_true, y_pred, y_probs

def run_training_pipeline(
    model_type: str = "transfer",
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_path: Optional[Path] = None
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, float]]:
    """
    Full training pipeline including data loading, class weighting,
    loss optimization, validation checkpointing, and metric history tracking.
    """
    set_seed(RANDOM_SEED)
    manifest_path = PROCESSED_DATA_DIR / "dataset_splits.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Please run dataset preparation first.")

    manifest_df = pd.read_csv(manifest_path)
    train_df = manifest_df[manifest_df["split"] == "train"]

    train_loader, val_loader, test_loader = get_dataloaders(manifest_df)
    logger.info(f"DataLoaders initialized: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")

    # Instantiate model
    model = build_model(model_type=model_type, num_classes=2, pretrained=True).to(DEVICE)
    logger.info(f"Model ({model_type}) initialized on {DEVICE}")

    # Loss with training class weights
    class_weights = compute_class_weights(train_df, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH if model_type == "transfer" else BASELINE_MODEL_PATH

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": []
    }

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    epochs_no_improve = 0

    start_time = time.time()
    logger.info(f"--- Starting Training ({model_type}) for {num_epochs} Epochs ---")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics, _, _, _ = evaluate_loader(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        logger.info(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Rec: {val_metrics['recall']*100:.1f}% | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Save best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            
            torch.save({
                "epoch": epoch,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "class_weights": class_weights.cpu().tolist()
            }, checkpoint_path)
            logger.info(f" ★ Best model checkpoint saved to {checkpoint_path} (Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.4f}, Best Val F1: {best_val_f1:.4f}")

    # Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, history, checkpoint["val_metrics"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train defect classification model.")
    parser.add_argument("--model", type=str, default="transfer", choices=["baseline", "transfer"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    run_training_pipeline(model_type=args.model, num_epochs=args.epochs, lr=args.lr)
