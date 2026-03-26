"""
Live Data Loader for the Monitoring Subsystem.

Responsibilities:
- Establish a read-only connection to MongoDB.
- Fetch unlabelled live inference data over a configurable rolling time window.
- Extract the 'raw_features' payload from the telemetry logs.
- Persist the extracted data as a CSV for traceability and return a DataFrame.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging

load_dotenv()


class LiveDataLoader:
    """
    Production-grade data acquirer for live inference logs.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the LiveDataLoader with MongoDB credentials.
        """
        try:
            self.config = config
            
            self.mongo_url = os.getenv("MONGODB_URL")
            self.db_name = os.getenv("MONGODB_DATABASE")
            self.collection_name = os.getenv("MONGODB_PREDICTION_COLLECTION")

            if not all([self.mongo_url, self.db_name, self.collection_name]):
                raise EnvironmentError(
                    "Missing MongoDB environment variables required for Monitoring."
                )

            logging.info("[LIVE DATA LOADER] Initialized successfully.")

        except Exception as e:
            logging.exception("[LIVE DATA LOADER] Initialization failed.")
            raise CustomerChurnException(e, sys)

    def fetch_recent_inference_data(self) -> pd.DataFrame:
        """
        Query MongoDB for predictions made within the configured time window.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the raw features submitted by users.
        """
        client: Optional[MongoClient] = None
        
        try:
            logging.info(
                f"[LIVE DATA LOADER] Connecting to DB: {self.db_name} | "
                f"Collection: {self.collection_name}"
            )
            client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
            collection = client[self.db_name][self.collection_name]

            # Calculate the cutoff timestamp for our rolling window
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.days_window)
            cutoff_iso = cutoff_date.isoformat()

            logging.info(f"[LIVE DATA LOADER] Querying logs from {cutoff_iso} to now.")

            # Query: Fetch records where timestamp >= cutoff, return only raw_features
            query = {"timestamp_utc": {"$gte": cutoff_iso}}
            projection = {"_id": 0, "raw_features": 1}

            cursor = collection.find(query, projection)
            records = list(cursor)

            if not records:
                logging.warning("[LIVE DATA LOADER] No inference data found in the given window.")
                return pd.DataFrame()

            # Extract the nested 'raw_features' dictionary into a flat DataFrame
            features_list = [doc.get("raw_features", {}) for doc in records]
            live_df = pd.DataFrame(features_list)

            logging.info(
                f"[LIVE DATA LOADER] Successfully fetched {len(live_df)} records. "
                f"Saving to {self.config.live_data_file_path}"
            )

            # Persist locally for artifact traceability
            live_df.to_csv(self.config.live_data_file_path, index=False)

            return live_df

        except PyMongoError as e:
            logging.exception("[LIVE DATA LOADER] MongoDB query failed.")
            raise CustomerChurnException(e, sys)
        except Exception as e:
            logging.exception("[LIVE DATA LOADER] Unexpected error during data fetch.")
            raise CustomerChurnException(e, sys)
        finally:
            if client is not None:
                client.close()