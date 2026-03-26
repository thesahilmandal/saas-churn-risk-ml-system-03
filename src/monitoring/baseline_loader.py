"""
Baseline Loader for the Monitoring Subsystem.

Responsibilities:
- Connect to the S3 Model Registry.
- Download the SSOT registry_metadata.json.
- Resolve the pointer for the 'current_production_version'.
- Download and return that version's baseline.json artifact.
"""

import os
import sys
from typing import Any, Dict

from src.cloud.s3_syncer import S3Sync
from src.constants.pipeline_constants import (
    S3_BUCKET_NAME,
    S3_MODEL_REGISTRY_DIR_NAME,
)
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file


class BaselineLoader:
    """
    Safely retrieves the statistical baseline of the active production model.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            self.s3_sync = S3Sync()
            
            # Temporary local path to store the fetched registry metadata
            self.local_metadata_path = os.path.join(
                self.config.monitoring_dir, 
                "temp_registry_metadata.json"
            )
            
            # Temporary local path to store the fetched baseline
            self.local_baseline_path = os.path.join(
                self.config.monitoring_dir, 
                "temp_baseline.json"
            )

            logging.info("[BASELINE LOADER] Initialized successfully.")

        except Exception as e:
            logging.exception("[BASELINE LOADER] Initialization failed.")
            raise CustomerChurnException(e, sys)

    def _get_production_baseline_uri(self) -> str:
        """
        Fetch registry metadata and extract the S3 URI for the current baseline.
        """
        s3_metadata_uri = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_REGISTRY_DIR_NAME}/registry_metadata.json"
        
        logging.info(f"[BASELINE LOADER] Fetching registry metadata from: {s3_metadata_uri}")
        self.s3_sync.download_file(s3_metadata_uri, self.local_metadata_path)

        metadata = read_json_file(self.local_metadata_path)
        
        current_version = metadata.get("current_production_version")
        if not current_version:
            raise ValueError("No 'current_production_version' defined in S3 registry metadata.")

        version_info = metadata.get("versions_metadata", {}).get(current_version)
        if not version_info:
            raise ValueError(f"Metadata missing for champion version: {current_version}")

        relative_baseline_path = version_info.get("baseline_path")
        if not relative_baseline_path:
            raise ValueError(f"No baseline_path defined for version {current_version}")

        # Clean the path and construct the exact S3 URI
        relative_baseline_path = relative_baseline_path.lstrip("./\\")
        s3_baseline_uri = f"s3://{S3_BUCKET_NAME}/{relative_baseline_path}"
        
        logging.info(f"[BASELINE LOADER] Resolved baseline URI for {current_version}: {s3_baseline_uri}")
        return s3_baseline_uri

    def fetch_baseline(self) -> Dict[str, Any]:
        """
        Download the baseline.json for the active model and load it into memory.
        
        Returns
        -------
        Dict[str, Any]
            The loaded baseline distribution dictionary.
        """
        try:
            s3_baseline_uri = self._get_production_baseline_uri()
            
            logging.info("[BASELINE LOADER] Downloading baseline artifact...")
            self.s3_sync.download_file(s3_baseline_uri, self.local_baseline_path)
            
            baseline_data = read_json_file(self.local_baseline_path)
            
            logging.info("[BASELINE LOADER] Baseline successfully loaded into memory.")
            
            # Clean up temporary files to keep the monitoring artifact directory pristine
            if os.path.exists(self.local_metadata_path):
                os.remove(self.local_metadata_path)
            if os.path.exists(self.local_baseline_path):
                os.remove(self.local_baseline_path)
                
            return baseline_data

        except Exception as e:
            logging.exception("[BASELINE LOADER] Failed to fetch production baseline.")
            raise CustomerChurnException(e, sys)