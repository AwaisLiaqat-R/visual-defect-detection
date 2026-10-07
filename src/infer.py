"""
Standalone production inference engine for defect detection.
Supports PIL images, file paths, and raw byte buffers with latency tracking and metadata enrichment.
"""
import io
import time
import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from src.config import (
    DEVICE, BEST_MODEL_PATH, METADATA_PATH,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, IDX_TO_CLASS
)
from src.models import build_model
from src.utils import setup_logger

logger = setup_logger("infer")

class DefectInferenceEngine:
    """
    Thread-safe, high-throughput inference engine for industrial defect classification.
    """
    def __init__(
        self,
        checkpoint_path: Path = BEST_MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
        device: str = DEVICE
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)

        # Load metadata if available
        self.operating_threshold = 0.50
        self.image_size = IMAGE_SIZE
        self.classes = CLASS_NAMES

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.operating_threshold = meta.get("operating_threshold", 0.50)
                self.classes = meta.get("classes", CLASS_NAMES)
                self.image_size = meta.get("image_size", IMAGE_SIZE)
                logger.info(f"Loaded model metadata: threshold={self.operating_threshold:.2f}, classes={self.classes}")
            except Exception as e:
                logger.warning(f"Could not load metadata from {self.metadata_path}: {e}")

        # Initialize transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Load model architecture and checkpoint weights
        self.model = build_model(model_type="transfer", num_classes=len(self.classes), pretrained=False)
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint weights from {self.checkpoint_path}")
        else:
            logger.warning(f"Checkpoint {self.checkpoint_path} not found. Running with uninitialized weights.")

        self.model.to(self.device)
        self.model.eval()

        # Warmup forward pass
        dummy_tensor = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)
        with torch.no_grad():
            _ = self.model(dummy_tensor)
        logger.info("Inference engine warm-up complete.")

    def _prepare_image(self, image_input: Union[str, Path, Image.Image, bytes]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Parses various input formats into a normalized PyTorch tensor and metadata."""
        filename = "in_memory"
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            filename = img_path.name
            img = Image.open(img_path)
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = img.size
        img_format = img.format if img.format else "RAW"
        img_rgb = img.convert("RGB")
        
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        meta = {
            "filename": filename,
            "format": img_format,
            "width": orig_w,
            "height": orig_h,
            "channels": 3
        }
        return tensor, meta

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes single image defect inference with precise latency measurement.
        """
        thresh = threshold if threshold is not None else self.operating_threshold
        t0 = time.perf_counter()

        tensor, img_meta = self._prepare_image(image_input)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        p_normal = float(probs[0])
        p_defective = float(probs[1])

        # Apply operating threshold
        is_defective = bool(p_defective >= thresh)
        pred_label = 1 if is_defective else 0
        pred_class = self.classes[pred_label]
        confidence = p_defective if is_defective else p_normal

        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "normal": round(p_normal, 4),
                "defective": round(p_defective, 4)
            },
            "threshold_used": round(thresh, 4),
            "is_defective": is_defective,
            "latency_ms": round(latency_ms, 2),
            "image_info": img_meta
        }

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, bytes]],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Processes multiple images in a high-speed batch."""
        t0 = time.perf_counter()
        predictions = [self.predict(inp, threshold=threshold) for inp in image_inputs]
        batch_latency = (time.perf_counter() - t0) * 1000.0

        defective_count = sum(1 for p in predictions if p["is_defective"])
        total = len(predictions)
        normal_count = total - defective_count
        defect_rate = (defective_count / max(total, 1)) * 100.0

        return {
            "total_images": total,
            "defective_count": defective_count,
            "normal_count": normal_count,
            "defect_rate": round(defect_rate, 2),
            "predictions": predictions,
            "total_batch_latency_ms": round(batch_latency, 2)
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run defect inference on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold override")
    args = parser.parse_args()

    engine = DefectInferenceEngine()
    res = engine.predict(args.image, threshold=args.threshold)
    print(json.dumps(res, indent=2))
