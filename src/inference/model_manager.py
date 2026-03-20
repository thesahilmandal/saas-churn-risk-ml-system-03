"""
Artifact Handler (Model Manager) for the Inference Service.

Responsibilities:
- Implement a thread-safe Singleton to prevent memory leaks in FastAPI.
- Fetch immutable metadata from S3 to resolve the current champion model pointer.
- Download ONLY the active champion model to minimize I/O and startup latency.
- Cache the scikit-learn pipeline (Preprocessor + Model) in RAM.
- Provide a safe hot-reload mechanism.
"""

import os
import sys
import threading
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from src.cloud.s3_syncer import S3Sync
from src.constants.pipeline_constants import (
    MODEL_REGISTRY_DIR,
    S3_BUCKET_NAME,
    S3_MODEL_REGISTRY_DIR_NAME,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import load_object, read_json_file


class ModelManager:
    """
    Thread-safe Singleton class to manage the lifecycle of the production ML pipeline.
    Ensures the model is loaded into memory exactly once per server instance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Ensure only one instance of ModelManager is ever created.
        """
        if not cls._instance:
            with cls._lock:
                # Double-checked locking
                if not cls._instance:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize configurations. The 'initialized' flag prevents
        re-initialization if __init__ is called multiple times by the Singleton.
        """
        if not hasattr(self, "initialized"):
            try:
                self.s3_sync = S3Sync()
                
                # S3 URIs
                self.s3_metadata_uri = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_REGISTRY_DIR_NAME}/registry_metadata.json"
                
                # Local Cache Paths
                self.local_cache_dir = str(MODEL_REGISTRY_DIR / "champion_cache")
                self.local_metadata_path = os.path.join(self.local_cache_dir, "registry_metadata.json")
                
                os.makedirs(self.local_cache_dir, exist_ok=True)
                
                # In-memory cache
                self.champion_pipeline: Optional[Pipeline] = None
                self.current_version: Optional[str] = None
                
                self.initialized = True
                logging.info("[MODEL MANAGER] Initialized Singleton instance.")

            except Exception as e:
                logging.exception("[MODEL MANAGER] Failed to initialize.")
                raise CustomerChurnException(e, sys)

    # ==========================================================
    # ARTIFACT RETRIEVAL & CACHING
    # ==========================================================

    def _fetch_metadata_from_s3(self) -> None:
        """
        Download only the registry metadata file from S3 to resolve the champion pointer.
        """
        logging.info(f"[MODEL MANAGER] Fetching registry metadata from S3: {self.s3_metadata_uri}")
        self.s3_sync.download_file(
            s3_uri=self.s3_metadata_uri,
            local_path=self.local_metadata_path,
        )

    def _resolve_champion_pointer(self) -> Tuple[str, str]:
        """
        Parse the local metadata to find the active production version and its S3 path.
        
        Returns:
            Tuple containing (version_name, s3_model_uri)
        """
        metadata = read_json_file(self.local_metadata_path)
        
        version = metadata.get("current_production_version")
        if not version:
            raise ValueError("No 'current_production_version' found in registry metadata.")

        version_info = metadata.get("versions_metadata", {}).get(version)
        if not version_info:
            raise ValueError(f"Metadata missing for champion version: {version}")

        relative_model_path = version_info.get("model_path")
        if not relative_model_path:
            raise ValueError(f"No model path defined for version {version}")

        # Construct exact S3 URI for the specific model file
        relative_model_path = relative_model_path.lstrip("./\\")
        s3_model_uri = f"s3://{S3_BUCKET_NAME}/{relative_model_path}"

        return version, s3_model_uri

    def _fetch_model_from_s3(self, version: str, s3_model_uri: str) -> str:
        """
        Download the specific champion model from S3 to the local cache.
        """
        local_model_path = os.path.join(self.local_cache_dir, version, "model.pkl")
        
        logging.info(f"[MODEL MANAGER] Downloading champion model {version} from S3: {s3_model_uri}")
        self.s3_sync.download_file(
            s3_uri=s3_model_uri,
            local_path=local_model_path,
        )
        
        return local_model_path

    def load_champion_model(self) -> None:
        """
        Optimized sequence to fetch metadata, resolve the pointer, download the specific 
        champion model, and load the `.pkl` into RAM.
        """
        try:
            with self._lock:
                logging.info("[MODEL MANAGER] Starting optimized champion model load sequence...")
                
                # 1. Get the Single Source of Truth
                self._fetch_metadata_from_s3()
                
                # 2. Find out what the current model is
                version, s3_model_uri = self._resolve_champion_pointer()
                logging.info(f"[MODEL MANAGER] Resolved champion version: {version}")
                
                # 3. Download ONLY that model
                local_model_path = self._fetch_model_from_s3(version, s3_model_uri)
                
                # 4. Load the bundled preprocessor+model pipeline into RAM
                self.champion_pipeline = load_object(local_model_path)
                self.current_version = version
                
                logging.info(f"[MODEL MANAGER] Successfully loaded version {version} into memory.")

        except Exception as e:
            logging.exception("[MODEL MANAGER] Failed to load champion model.")
            raise CustomerChurnException(e, sys)

    # ==========================================================
    # INFERENCE INTERFACE
    # ==========================================================

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """
        Apply the cached pipeline (transform -> predict) on valid data.
        
        Args:
            X (pd.DataFrame): Validated input features.
            
        Returns:
            NumPy array of prediction probabilities.
        """
        if self.champion_pipeline is None:
            raise RuntimeError("Model pipeline is not loaded into memory. Call `load_champion_model()` first.")
        
        try:
            # The pipeline automatically applies the preprocessor and then calls the estimator
            probabilities = self.champion_pipeline.predict_proba(X)
            return probabilities
            
        except Exception as e:
            logging.exception("[MODEL MANAGER] Inference failed during pipeline execution.")
            raise CustomerChurnException(e, sys)

    def get_current_version(self) -> str:
        """Return the version string currently loaded in memory."""
        if not self.current_version:
            return "unknown"
        return self.current_version