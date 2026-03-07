"""
Artifact Handler (Model Manager) for the Inference Service.

Responsibilities:
- Implement a thread-safe Singleton to prevent memory leaks in FastAPI.
- Synchronize the Model Registry from S3 to the local environment.
- Read immutable metadata to resolve the current champion model pointer.
- Cache the scikit-learn pipeline (Preprocessor + Model) in RAM.
- Provide a safe hot-reload mechanism.
"""

import sys
import threading
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from src.cloud.s3_syncer import S3Sync
from src.constants.pipeline_constants import (
    MODEL_REGISTRY_DIR,
    MODEL_REGISTRY_METADATA_PATH,
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
                
                # Reconstruct the S3 URI based on your constants
                self.s3_registry_uri = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_REGISTRY_DIR_NAME}"
                self.local_registry_dir = str(MODEL_REGISTRY_DIR)
                self.metadata_path = str(MODEL_REGISTRY_METADATA_PATH)
                
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

    def _sync_registry_from_s3(self) -> None:
        """
        Pull the latest Model Registry folder from AWS S3.
        """
        logging.info(
            f"[MODEL MANAGER] Syncing registry from S3: {self.s3_registry_uri} "
            f"-> {self.local_registry_dir}"
        )
        self.s3_sync.sync_folder_from_s3(
            folder=self.local_registry_dir,
            aws_bucket_url=self.s3_registry_uri,
        )

    def _get_champion_metadata(self) -> Tuple[str, str]:
        """
        Read the registry metadata to resolve the current production model.
        
        Returns:
            Tuple containing (version_name, model_file_path)
        """
        metadata = read_json_file(self.metadata_path)
        
        version = metadata.get("current_production_version")
        if not version:
            raise ValueError("No 'current_production_version' found in registry metadata.")

        version_info = metadata.get("versions_metadata", {}).get(version)
        if not version_info:
            raise ValueError(f"Metadata missing for champion version: {version}")

        model_path = version_info.get("model_path")
        if not model_path:
            raise ValueError(f"No model path defined for version {version}")

        return version, model_path

    def load_champion_model(self) -> None:
        """
        Full sequence to sync S3, resolve the pointer, and load the `.pkl` into RAM.
        This should be called during the FastAPI startup event.
        """
        try:
            with self._lock:
                logging.info("[MODEL MANAGER] Loading Champion Model...")
                
                self._sync_registry_from_s3()
                version, model_path = self._get_champion_metadata()
                
                logging.info(f"[MODEL MANAGER] Resolved champion version: {version} at {model_path}")
                
                # Load the bundled preprocessor+model pipeline into RAM
                self.champion_pipeline = load_object(model_path)
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