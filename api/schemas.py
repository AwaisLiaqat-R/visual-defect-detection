"""
Pydantic schemas for API request validation and structured responses.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ImageMetadata(BaseModel):
    filename: str = Field(..., description="Uploaded file name")
    format: str = Field(..., description="Image format (PNG, JPEG, etc.)")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    channels: int = Field(..., description="Number of color channels")

class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted class name ('normal' or 'defective')")
    predicted_label: int = Field(..., description="Numeric class index (0 for normal, 1 for defective)")
    confidence: float = Field(..., description="Probability of the predicted class [0.0 - 1.0]")
    probabilities: Dict[str, float] = Field(..., description="Probability breakdown across all classes")
    threshold_used: float = Field(..., description="Decision threshold applied for defective class")
    is_defective: bool = Field(..., description="Boolean flag indicating defect detection")
    latency_ms: float = Field(..., description="Inference runtime in milliseconds")
    image_info: ImageMetadata = Field(..., description="Extracted image dimensions and format")

class BatchPredictionResponse(BaseModel):
    total_images: int = Field(..., description="Total images processed in batch")
    defective_count: int = Field(..., description="Count of defective items detected")
    normal_count: int = Field(..., description="Count of normal items detected")
    defect_rate: float = Field(..., description="Defect rate percentage in this batch")
    predictions: List[PredictionResponse] = Field(..., description="Individual item predictions")
    total_batch_latency_ms: float = Field(..., description="Total processing time for the batch")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status ('healthy' or 'unhealthy')")
    model_loaded: bool = Field(..., description="Whether PyTorch model weights are in memory")
    model_type: str = Field(..., description="Model architecture type")
    device: str = Field(..., description="Inference execution device ('cpu' or 'cuda')")
    operating_threshold: float = Field(..., description="Current configured decision threshold")

class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    num_classes: int
    classes: List[str]
    image_size: int
    mean: List[float]
    std: List[float]
    decision_threshold: float
    test_metrics: Optional[Dict[str, float]] = None
    training_date: Optional[str] = None
