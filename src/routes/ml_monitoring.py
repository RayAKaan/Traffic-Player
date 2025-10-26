# app/routes/ml_monitoring.py

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import psutil
import time

router = APIRouter()

@router.get("/ml/status")
async def get_ml_status():
    # ✅ Real-time system metrics
    cpu_usage = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory().percent
    gpu_usage = 78  # You can mock this or integrate with nvidia-smi later
    inference_speed = 130  # ms per frame (mocked or pulled from logs)

    # ✅ Model performance data (mocked for now)
    models = [
        {
            "name": "Vehicle Detection",
            "accuracy": 94.2,
            "confidence": 98.5,
            "status": "optimal",
            "lastTrained": "2 hours ago",
        },
        {
            "name": "Traffic Flow Prediction",
            "accuracy": 89.7,
            "confidence": 87.2,
            "status": "warning",
            "lastTrained": "6 hours ago",
        },
        {
            "name": "Emergency Recognition",
            "accuracy": 96.8,
            "confidence": 99.1,
            "status": "optimal",
            "lastTrained": "1 hour ago",
        },
        {
            "name": "Pattern Analysis",
            "accuracy": 91.3,
            "confidence": 92.7,
            "status": "training",
            "lastTrained": "ongoing",
        },
    ]

    return JSONResponse({
        "models": models,
        "system": {
            "cpu": cpu_usage,
            "gpu": gpu_usage,
            "memory": memory,
            "inference_speed": inference_speed
        }
    })
