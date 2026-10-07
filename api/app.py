"""
Production-grade FastAPI Visual Defect Detection API.
Features robust input validation, structured JSON logging, error handling,
operating threshold configuration, latency tracking, and batch inference.
"""
import io
import time
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import torch

from src.config import (
    BEST_MODEL_PATH, METADATA_PATH, MAX_IMAGE_SIZE_MB,
    ALLOWED_IMAGE_TYPES, API_HOST, API_PORT, CLASS_NAMES
)
from src.infer import DefectInferenceEngine
from api.schemas import (
    PredictionResponse, BatchPredictionResponse, HealthResponse,
    ModelInfoResponse, ImageMetadata
)
from api.logger import api_logger

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-warms the deep learning model on application startup."""
    global engine
    api_logger.info("Initializing DefectInferenceEngine on startup...")
    try:
        engine = DefectInferenceEngine(
            checkpoint_path=BEST_MODEL_PATH,
            metadata_path=METADATA_PATH
        )
        api_logger.info(f"Engine ready. Operating threshold: {engine.operating_threshold:.2f}, Device: {engine.device}")
    except Exception as e:
        api_logger.error(f"Failed to initialize inference engine: {e}", exc_info=True)
    yield

app = FastAPI(
    title="Visual Defect Detection Production API",
    description="Automated AI Quality Control & Surface Defect Classification Service (KolektorSDD2)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for web integrations and frontend dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine and runtime metrics tracker
engine: Optional[DefectInferenceEngine] = None
METRICS_COUNTER = {
    "total_inferences": 0,
    "total_defects_detected": 0,
    "total_normal_detected": 0,
    "total_latency_ms": 0.0,
    "start_timestamp": time.time()
}

@app.get("/", tags=["General"])
async def root():
    """Service landing page with links to documentation and health status."""
    return {
        "service": "Visual Defect Detection API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "model_info_url": "/model-info",
        "predict_endpoint": "/predict"
    }

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health():
    """System health check verifying model state and device status."""
    is_loaded = engine is not None and engine.model is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_type="Transfer-ResNet18",
        device=str(engine.device) if engine else "unknown",
        operating_threshold=engine.operating_threshold if engine else 0.50
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Diagnostics"])
async def model_info():
    """Returns architecture metadata, class mapping, ImageNet normalization, and evaluation metrics."""
    if not engine or not METADATA_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not available. Ensure model has been evaluated."
        )
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return ModelInfoResponse(
        model_name=meta.get("model_name", "ResNet18-Transfer-Defect-Classifier"),
        architecture=meta.get("architecture", "ResNet18"),
        num_classes=meta.get("num_classes", 2),
        classes=meta.get("classes", CLASS_NAMES),
        image_size=meta.get("image_size", 224),
        mean=meta.get("mean", [0.485, 0.456, 0.406]),
        std=meta.get("std", [0.229, 0.224, 0.225]),
        decision_threshold=meta.get("operating_threshold", 0.50),
        test_metrics=meta.get("test_metrics"),
        training_date=meta.get("timestamp")
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_single(
    file: UploadFile = File(..., description="Product image file (JPEG/PNG/BMP)"),
    threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional decision threshold override for defect classification"
    )
):
    """
    Submits a single product image to the AI model for automated quality inspection.
    Returns defect classification, confidence score, class probabilities, and runtime latency.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized."
        )

    # 1. Validate MIME type
    content_type = file.content_type
    if not content_type or content_type.lower() not in ALLOWED_IMAGE_TYPES:
        api_logger.warning(f"Invalid MIME type received: {content_type} (Filename: {file.filename})")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed types: {list(ALLOWED_IMAGE_TYPES)}"
        )

    # 2. Read image buffer & validate file size
    try:
        contents = await file.read()
    except Exception as e:
        api_logger.error(f"Failed to read upload stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error reading uploaded file."
        )

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        api_logger.warning(f"File size exceeded limit: {size_mb:.2f} MB > {MAX_IMAGE_SIZE_MB} MB")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {MAX_IMAGE_SIZE_MB} MB."
        )

    # 3. Validate image integrity
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(contents)) # Re-open after verify
    except (UnidentifiedImageError, Exception) as e:
        api_logger.error(f"Image corruption check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file. Unable to decode image format."
        )

    # 4. Perform AI Defect Inference
    try:
        pred_dict = engine.predict(pil_img, threshold=threshold)
        pred_dict["image_info"]["filename"] = file.filename or "uploaded_image"
        
        # Update telemetry metrics
        METRICS_COUNTER["total_inferences"] += 1
        if pred_dict["is_defective"]:
            METRICS_COUNTER["total_defects_detected"] += 1
        else:
            METRICS_COUNTER["total_normal_detected"] += 1
        METRICS_COUNTER["total_latency_ms"] += pred_dict["latency_ms"]

        api_logger.info(
            f"Prediction: {pred_dict['predicted_class'].upper()} "
            f"(Confidence: {pred_dict['confidence']:.4f}, Latency: {pred_dict['latency_ms']:.2f}ms, File: {file.filename})"
        )

        return PredictionResponse(**pred_dict)
    except Exception as e:
        api_logger.error(f"Inference execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(e)}"
        )

@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="List of product image files"),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0)
):
    """
    Submits multiple images in a single batch request for high-throughput production line inspection.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided in batch request.")

    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum batch limit is 50 images per request.")

    start_batch = time.perf_counter()
    results = []

    for file in files:
        contents = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(contents))
            pred = engine.predict(pil_img, threshold=threshold)
            pred["image_info"]["filename"] = file.filename or "uploaded_image"
            results.append(PredictionResponse(**pred))
        except Exception as e:
            api_logger.warning(f"Failed processing {file.filename} in batch: {e}")

    total_batch_time = (time.perf_counter() - start_batch) * 1000.0
    defective_count = sum(1 for p in results if p.is_defective)
    total_valid = len(results)
    normal_count = total_valid - defective_count
    defect_rate = (defective_count / max(total_valid, 1)) * 100.0

    return BatchPredictionResponse(
        total_images=total_valid,
        defective_count=defective_count,
        normal_count=normal_count,
        defect_rate=round(defect_rate, 2),
        predictions=results,
        total_batch_latency_ms=round(total_batch_time, 2)
    )

@app.get("/metrics", tags=["Diagnostics"])
async def metrics():
    """Returns aggregate operational statistics, defect detection counts, and average latencies."""
    total = METRICS_COUNTER["total_inferences"]
    avg_latency = (METRICS_COUNTER["total_latency_ms"] / total) if total > 0 else 0.0
    defect_percentage = (METRICS_COUNTER["total_defects_detected"] / total * 100.0) if total > 0 else 0.0

    return {
        "uptime_seconds": round(time.time() - METRICS_COUNTER["start_timestamp"], 1),
        "total_inferences": total,
        "total_defects_detected": METRICS_COUNTER["total_defects_detected"],
        "total_normal_detected": METRICS_COUNTER["total_normal_detected"],
        "defect_rate_percentage": round(defect_percentage, 2),
        "average_latency_ms": round(avg_latency, 2)
    }
