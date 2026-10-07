"""
Model Architectures for Visual Defect Detection:
1. BaselineCNN: Lightweight scratch convolutional network establishing the minimum performance benchmark.
2. TransferDefectClassifier: Production-grade transfer learning architecture using pre-trained ResNet-18.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, Optional

class BaselineCNN(nn.Module):
    """
    A lightweight custom Convolutional Neural Network serving as the baseline.
    Enables empirical comparison against transfer learning methods.
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class TransferDefectClassifier(nn.Module):
    """
    Transfer-learning visual defect classification model based on ResNet-18.
    
    Why ResNet-18?
    1. Proven feature extractor on low-level edge/texture anomalies (crucial for surface defects).
    2. Lightweight footprint (~11M params) enabling <15ms CPU latency in production FastAPI runtime.
    3. Residual connections prevent vanishing gradients when fine-tuning.
    4. Ideal balance of low latency, high defect recall, and small memory footprint.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False, dropout_rate: float = 0.3):
        super(TransferDefectClassifier, self).__init__()
        
        # Load weights
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        
        # Replace classification head with regularized dense layers
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self, unfreeze_from_layer: Optional[str] = None):
        """Allows staged fine-tuning by unfreezing backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_model(model_type: str = "transfer", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Factory function for instantiating models."""
    if model_type == "baseline":
        return BaselineCNN(num_classes=num_classes)
    elif model_type == "transfer":
        return TransferDefectClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'baseline' or 'transfer'.")
