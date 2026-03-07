"""
Data Contracts for the Inference Service.

Responsibilities:
- Define strict Pydantic models for API responses.
- Autogenerate OpenAPI (Swagger) documentation.
- Ensure standard JSON serialization for telemetry and the end-user.
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    """Schema for a single successful prediction."""
    
    customerID: str = Field(
        ...,
        description="Unique identifier for the customer.",
        json_schema_extra={"example": "7590-VHVEG"}
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw probability score from the model (0.0 to 1.0)."
    )
    churn_decision: str = Field(
        ...,
        description="Final binary decision based on the threshold (e.g., 'Yes' or 'No').",
        json_schema_extra={"example": "Yes"}
    )


class ErrorDetail(BaseModel):
    """Schema for a single failed row (Dead Letter Queue entry)."""
    
    row_index: Optional[int] = Field(
        default=None,
        description="The line number or index of the failed row in the CSV."
    )
    customerID: Optional[str] = Field(
        default=None,
        description="Customer ID if it could be successfully extracted before failure."
    )
    error_reason: str = Field(
        ...,
        description="Specific reason why the row failed validation.",
        json_schema_extra={"example": "Missing required field: TotalCharges"}
    )


class PredictionResponse(BaseModel):
    """
    Master schema for the /predict endpoint response.
    Provides a comprehensive summary of the batch inference job.
    """
    
    # ConfigDict configures behavior and provides a full example for Swagger UI
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "request_id": "req-12345-abcde",
                "model_version": "v2",
                "timestamp_utc": "2026-03-07T12:00:00Z",
                "total_rows_received": 500,
                "total_success": 400,
                "total_failed": 100,
                "predictions": [
                    {
                        "customerID": "7590-VHVEG",
                        "churn_probability": 0.85,
                        "churn_decision": "Yes"
                    }
                ],
                "errors": [
                    {
                        "row_index": 42,
                        "customerID": None,
                        "error_reason": "Invalid dtype for SeniorCitizen: expected numeric"
                    }
                ]
            }
        }
    )

    request_id: str = Field(
        ...,
        description="Unique trace ID for the API request."
    )
    model_version: str = Field(
        ...,
        description="The version of the champion model used for inference (e.g., 'v1')."
    )
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of when the response was generated."
    )
    
    # Summary metrics
    total_rows_received: int = Field(..., ge=1, description="Total rows in the uploaded CSV.")
    total_success: int = Field(..., ge=0, description="Number of successfully predicted rows.")
    total_failed: int = Field(..., ge=0, description="Number of rows rejected by the validator.")

    # Payloads
    predictions: List[PredictionResult] = Field(
        default_factory=list,
        description="List of successful predictions."
    )
    errors: List[ErrorDetail] = Field(
        default_factory=list,
        description="List of failed rows that were routed to the Dead Letter Queue."
    )