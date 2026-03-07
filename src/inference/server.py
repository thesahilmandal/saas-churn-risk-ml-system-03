"""
API Gateway for the Inference Service.

Responsibilities:
- Provide a robust, asynchronous REST API using FastAPI.
- Manage the application lifespan (startup model loading).
- Expose the `/predict` endpoint for CSV batch inference.
- Expose the `/reload` endpoint for zero-downtime model updates.
- Offload observability writing to non-blocking background tasks.
"""

import io
import sys
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.exception import CustomerChurnException
from src.constants.pipeline_constants import REFERENCE_SCHEMA_PATH
from src.inference.data_validator import DataValidator
from src.inference.model_manager import ModelManager
from src.inference.prediction import PredictionEngine
from src.inference.schemas import PredictionResponse
from src.inference.telemetry import TelemetryLogger
from src.logging import logging

# ==========================================================
# LIFESPAN & INITIALIZATION
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage startup and shutdown events.
    Ensures the ML model is loaded into memory before accepting any requests.
    """
    try:
        logging.info("[API GATEWAY] Starting up Inference Service...")
        
        # 1. Initialize Singleton Model Manager and load the model from S3
        model_manager = ModelManager()
        model_manager.load_champion_model()
        
        logging.info("[API GATEWAY] Startup complete. Ready for traffic.")
        yield
        
    except Exception as e:
        logging.exception("[API GATEWAY] Fatal error during startup.")
        raise RuntimeError("Could not initialize Inference Service.") from e
    finally:
        logging.info("[API GATEWAY] Shutting down Inference Service...")

# Initialize FastAPI App
app = FastAPI(
    title="Customer Churn Inference API",
    description="Production ML service for predicting telecommunication customer churn.",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize stateless components
# Note: Ensure your 'config/schema.yaml' path matches your repository structure
SCHEMA_PATH = "config/schema.yaml"
validator = DataValidator(schema_file_path=REFERENCE_SCHEMA_PATH)
prediction_engine = PredictionEngine()
telemetry_logger = TelemetryLogger()


# ==========================================================
# HEALTH & MAINTENANCE ENDPOINTS
# ==========================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Kubernetes/Docker health check endpoint."""
    model_manager = ModelManager()
    version = model_manager.get_current_version()
    return {
        "status": "healthy",
        "loaded_model_version": version
    }

@app.post("/reload", tags=["System"])
async def reload_model():
    """
    Zero-downtime webhook to pull the latest champion model from S3.
    Triggered by CI/CD after the Training Pipeline completes.
    """
    try:
        logging.info("[API GATEWAY] Manual model reload triggered via API.")
        model_manager = ModelManager()
        model_manager.load_champion_model()
        
        return {
            "status": "success",
            "message": f"Successfully reloaded model version: {model_manager.get_current_version()}"
        }
    except Exception as e:
        logging.error(f"[API GATEWAY] Reload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload the model from S3.")


# ==========================================================
# INFERENCE ENDPOINT
# ==========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Generate Churn Predictions from a CSV batch."
)
async def predict_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Accepts a CSV file of customer data, validates it, and returns churn predictions.
    Invalid rows are rejected and sent to a Dead Letter Queue.
    """
    request_id = f"req-{uuid.uuid4()}"
    logging.info(f"[API GATEWAY] Received prediction request | Trace ID: {request_id}")

    # 1. File Validation
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        logging.error(f"[API GATEWAY] Failed to parse CSV: {e}")
        raise HTTPException(status_code=400, detail="Could not parse the uploaded CSV file.")

    total_rows = len(df)
    if total_rows == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    try:
        # 2. Gatekeeper: Validate Data
        valid_df, invalid_df, errors = validator.validate_batch(df)

        # 3. Core Engine: Generate Predictions
        predictions = prediction_engine.execute_batch_prediction(valid_df)

        # 4. Telemetry: Dispatch Background Tasks for MongoDB
        model_manager = ModelManager()
        current_version = model_manager.get_current_version()

        if not valid_df.empty:
            background_tasks.add_task(
                telemetry_logger.log_predictions,
                request_id=request_id,
                model_version=current_version,
                valid_df=valid_df,
                predictions=predictions
            )
        
        if not invalid_df.empty:
            background_tasks.add_task(
                telemetry_logger.log_dlq_errors,
                request_id=request_id,
                invalid_df=invalid_df,
                errors=errors
            )

        # 5. Formulate API Response using Pydantic Schema
        response = PredictionResponse(
            request_id=request_id,
            model_version=current_version,
            total_rows_received=total_rows,
            total_success=len(predictions),
            total_failed=len(errors),
            predictions=predictions,
            errors=errors
        )
        
        logging.info(f"[API GATEWAY] Request {request_id} processed successfully.")
        return response

    except CustomerChurnException as e:
        # Custom exceptions raised by our internal components
        logging.error(f"[API GATEWAY] Internal ML Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail="Internal ML processing error.")
    except Exception as e:
        # Catch-all for unexpected Python errors
        logging.error(f"[API GATEWAY] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected system error occurred.")