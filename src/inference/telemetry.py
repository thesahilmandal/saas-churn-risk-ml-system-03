"""
Telemetry and Observability Logger for the Inference Service.

Responsibilities:
- Establish a Singleton connection to MongoDB to avoid connection pooling exhaustion.
- Log successful predictions alongside their raw input features (for data drift monitoring).
- Log failed validation rows to the Dead Letter Queue (DLQ) with exact error reasons.
- Execute safely without crashing the main application if the database connection drops.
"""

import os
import sys
from datetime import datetime, timezone
from typing import List
from dotenv import load_dotenv

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.exception import CustomerChurnException
from src.inference.schemas import ErrorDetail, PredictionResult
from src.logging import logging

load_dotenv()


class TelemetryLogger:
    """
    Production-grade MongoDB telemetry client.
    Logs inference payloads and dead-letter queue items.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one MongoDB client is created."""
        if cls._instance is None:
            cls._instance = super(TelemetryLogger, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self) -> None:
        """Initialize the MongoDB client and select collections."""
        try:
            mongo_url = os.getenv("MONGODB_URL")
            db_name = os.getenv("MONGODB_DATABASE")
            pred_collection = os.getenv("MONGODB_PREDICTION_COLLECTION")
            
            # Note: Using the exact spelling from your .env file
            dlq_collection = os.getenv("MONGODB_DLQ_COLLETION") 

            if not all([mongo_url, db_name, pred_collection, dlq_collection]):
                logging.warning(
                    "[TELEMETRY] Missing MongoDB environment variables. "
                    "Telemetry will be disabled."
                )
                self.client = None
                return

            self.client = MongoClient(mongo_url)
            self.db = self.client[db_name]
            self.prediction_logs = self.db[pred_collection]
            self.dead_letter_queue = self.db[dlq_collection]

            logging.info(f"[TELEMETRY] Connected to MongoDB database: {db_name}")

        except PyMongoError as e:
            logging.exception("[TELEMETRY ERROR] Failed to connect to MongoDB.")
            self.client = None
            # We do not raise CustomerChurnException here because we don't want 
            # the entire API to crash just because the logging DB is temporarily down.

    def log_predictions(
        self,
        request_id: str,
        model_version: str,
        valid_df: pd.DataFrame,
        predictions: List[PredictionResult]
    ) -> None:
        """
        Merges the raw user inputs with the model's predictions and inserts them into MongoDB.
        """
        if self.client is None or valid_df.empty or not predictions:
            return

        try:
            # Convert raw inputs to a list of dictionaries
            raw_inputs = valid_df.to_dict(orient="records")
            
            # Convert Pydantic prediction models to dictionaries
            prediction_dicts = [p.model_dump() for p in predictions]

            documents_to_insert = []
            timestamp = datetime.now(timezone.utc).isoformat()

            # Zip them together: 1 row of input -> 1 prediction output
            for raw_data, pred_data in zip(raw_inputs, prediction_dicts):
                doc = {
                    "request_id": request_id,
                    "model_version": model_version,
                    "timestamp_utc": timestamp,
                    "raw_features": raw_data,
                    "prediction_output": pred_data
                }
                documents_to_insert.append(doc)

            if documents_to_insert:
                self.prediction_logs.insert_many(documents_to_insert)
                logging.info(
                    f"[TELEMETRY] Logged {len(documents_to_insert)} successful predictions to MongoDB."
                )

        except Exception as e:
            # We catch broadly to ensure background tasks never crash the server loop
            logging.error(f"[TELEMETRY ERROR] Failed to log predictions: {e}")

    def log_dlq_errors(
        self,
        request_id: str,
        invalid_df: pd.DataFrame,
        errors: List[ErrorDetail]
    ) -> None:
        """
        Merges the invalid rows with their specific validation errors and inserts them into the DLQ.
        """
        if self.client is None or invalid_df.empty or not errors:
            return

        try:
            raw_invalid_inputs = invalid_df.to_dict(orient="records")
            error_dicts = [e.model_dump() for e in errors]

            documents_to_insert = []
            timestamp = datetime.now(timezone.utc).isoformat()

            for raw_data, error_data in zip(raw_invalid_inputs, error_dicts):
                doc = {
                    "request_id": request_id,
                    "timestamp_utc": timestamp,
                    "raw_features": raw_data,
                    "error_details": error_data
                }
                documents_to_insert.append(doc)

            if documents_to_insert:
                self.dead_letter_queue.insert_many(documents_to_insert)
                logging.info(
                    f"[TELEMETRY] Logged {len(documents_to_insert)} failed rows to the Dead Letter Queue."
                )

        except Exception as e:
            logging.error(f"[TELEMETRY ERROR] Failed to log DLQ errors: {e}")