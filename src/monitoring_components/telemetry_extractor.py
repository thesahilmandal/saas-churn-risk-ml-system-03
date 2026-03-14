"""
Telemetry Extractor for the Monitoring Pipeline.

Responsibilities
----------------
1. Connect to MongoDB using environment credentials.
2. Query the prediction logs for the last N days.
3. Flatten the JSON payloads (raw_features + prediction_output) into a tabular format.
4. Enforce minimum row requirements before proceeding with drift analysis.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from src.entity.artifact_entity import TelemetryExtractionArtifact
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging

load_dotenv()


class TelemetryExtractor:
    """
    Extracts time-bounded production telemetry for drift analysis.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            os.makedirs(self.config.telemetry_data_dir, exist_ok=True)
            
            # MongoDB Configuration
            self.mongo_url = os.getenv("MONGODB_URL")
            self.db_name = os.getenv("MONGODB_DATABASE")
            self.collection_name = os.getenv("MONGODB_PREDICTION_LOGGING_COLLECTION")

            if not all([self.mongo_url, self.db_name, self.collection_name]):
                raise EnvironmentError("MongoDB credentials missing from environment variables.")

            logging.info("[TELEMETRY EXTRACTOR] Initialized successfully.")

        except Exception as e:
            logging.exception("[TELEMETRY EXTRACTOR] Initialization failed.")
            raise CustomerChurnException(e, sys) from e

    def _extract_data(self) -> pd.DataFrame:
        """
        Execute the time-bounded MongoDB query and flatten the results.
        """
        client = None
        try:
            # Calculate time window
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.config.data_window_days)

            logging.info(
                f"[TELEMETRY EXTRACTOR] Querying DB from {start_date.isoformat()} "
                f"to {end_date.isoformat()}"
            )

            client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
            collection = client[self.db_name][self.collection_name]

            # Query filtering by ISODate string comparison
            query = {
                "timestamp_utc": {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            }
            
            # Project only necessary fields to reduce RAM overhead
            projection = {"_id": 0, "raw_features": 1, "prediction_output": 1}
            
            cursor = collection.find(query, projection)
            records = list(cursor)

            if not records:
                logging.warning("[TELEMETRY EXTRACTOR] No data found in the specified window.")
                return pd.DataFrame(), start_date.isoformat(), end_date.isoformat()

            # Flatten the nested MongoDB documents
            flattened_data = []
            for doc in records:
                row = doc.get("raw_features", {})
                
                # Optionally append the model's output to monitor prediction drift
                pred = doc.get("prediction_output", {})
                if "churn_probability" in pred:
                    row["model_churn_probability"] = pred["churn_probability"]
                if "churn_decision" in pred:
                    row["model_churn_decision"] = pred["churn_decision"]
                    
                flattened_data.append(row)

            df = pd.DataFrame(flattened_data)
            return df, start_date.isoformat(), end_date.isoformat()

        except Exception as e:
            raise CustomerChurnException(e, sys) from e
        finally:
            if client:
                client.close()

    def initiate_extraction(self) -> TelemetryExtractionArtifact:
        """
        Execute extraction, validate data volume, and return artifact.
        """
        try:
            logging.info("[TELEMETRY EXTRACTOR] Execution started.")

            df, start_str, end_str = self._extract_data()
            row_count = len(df)

            logging.info(f"[TELEMETRY EXTRACTOR] Extracted {row_count} rows.")

            # Minimum volume safeguard
            if row_count < self.config.min_rows_required:
                raise ValueError(
                    f"Insufficient data for statistically significant drift analysis. "
                    f"Required: {self.config.min_rows_required}, Found: {row_count}."
                )

            # Save dataset locally for the Drift Analyzer
            df.to_csv(self.config.telemetry_csv_path, index=False)

            artifact = TelemetryExtractionArtifact(
                telemetry_data_path=self.config.telemetry_csv_path,
                extracted_rows=row_count,
                window_start_date=start_str,
                window_end_date=end_str
            )

            logging.info(f"[TELEMETRY EXTRACTOR] Completed. Data saved to {self.config.telemetry_csv_path}")
            return artifact

        except Exception as e:
            logging.exception("[TELEMETRY EXTRACTOR] Execution failed.")
            raise CustomerChurnException(e, sys) from e