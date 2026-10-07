"""
Unit tests for Model Architectures, Transforms, Loss Weights, and Metrics Calculation.
"""
import pytest
import numpy as np
import torch
import pandas as pd

from src.models import BaselineCNN, TransferDefectClassifier, build_model
from src.dataset import get_transforms
from src.utils import calculate_metrics, set_seed
from src.train import compute_class_weights
from src.config import IMAGE_SIZE

def test_baseline_cnn_forward():
    """Verifies BaselineCNN handles batch input and outputs 2-class logits."""
    model = BaselineCNN(num_classes=2, in_channels=3)
    model.eval()
    dummy_input = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"

def test_transfer_classifier_forward():
    """Verifies TransferDefectClassifier handles batch input and outputs 2-class logits."""
    model = TransferDefectClassifier(num_classes=2, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"

def test_model_factory():
    """Verifies build_model instantiates appropriate models."""
    baseline = build_model(model_type="baseline", num_classes=2)
    assert isinstance(baseline, BaselineCNN)

    transfer = build_model(model_type="transfer", num_classes=2, pretrained=False)
    assert isinstance(transfer, TransferDefectClassifier)

    with pytest.raises(ValueError):
        build_model(model_type="invalid_type")

def test_transforms_shape_and_type():
    """Verifies training and eval transforms produce normalized float tensors."""
    from PIL import Image
    dummy_pil = Image.new("RGB", (300, 400), color=(128, 128, 128))
    
    train_tf = get_transforms(image_size=IMAGE_SIZE, is_training=True)
    eval_tf = get_transforms(image_size=IMAGE_SIZE, is_training=False)

    tensor_train = train_tf(dummy_pil)
    tensor_eval = eval_tf(dummy_pil)

    assert tensor_train.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor_eval.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor_train.dtype == torch.float32
    assert tensor_eval.dtype == torch.float32

def test_compute_class_weights():
    """Verifies class weight calculation balances inverse sample frequencies."""
    # 80 normal, 20 defective (4:1 imbalance)
    df = pd.DataFrame({
        "label": [0] * 80 + [1] * 20
    })
    weights = compute_class_weights(df, device="cpu")
    assert len(weights) == 2
    # Defective weight should be 4x higher than Normal weight
    assert weights[1] > weights[0]
    assert np.isclose((weights[1] / weights[0]).item(), 4.0, atol=1e-2)

def test_calculate_metrics():
    """Verifies precision, recall, F1, and F2 calculations against known ground truth."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0]) # 3 TP, 1 FN, 0 FP, 4 TN
    y_probs = np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.1, 0.95, 0.05])

    metrics = calculate_metrics(y_true, y_pred, y_probs)

    assert metrics["accuracy"] == 7 / 8
    assert metrics["precision"] == 1.0  # 3 / (3 + 0)
    assert metrics["recall"] == 0.75    # 3 / (3 + 1)
    assert np.isclose(metrics["f1"], (2 * 1.0 * 0.75) / (1.0 + 0.75))
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f2" in metrics
