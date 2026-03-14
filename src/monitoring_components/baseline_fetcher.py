"""
Baseline Fetcher for the Monitoring Pipeline.

Responsibilities
----------------
1. Synchronize the Model Registry from S3 (to ensure the latest metadata).
2. Parse the registry metadata to identify the active champion model.
3. Locate and copy the champion's baseline.json to the monitoring artifacts directory.
"""

import os
import shutil
import sys

from src.cloud.s3_syncer import S3Sync
from src.constants.pipeline_constants import (
    MODEL_REGISTRY_DIR,
    MODEL_REGISTRY_METADATA_PATH,
    S3_BUCKET_NAME,
    S3_MODEL_REGISTRY_DIR_NAME,
)
from src.entity.artifact_entity import BaselineFetchArtifact
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file


class BaselineFetcher:
    """
    Retrieves the statistical baseline of the current production model.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            self.s3_sync = S3Sync()
            
            self.s3_registry_uri = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_REGISTRY_DIR_NAME}"
            self.local_registry_dir = str(MODEL_REGISTRY_DIR)
            self.metadata_path = str(MODEL_REGISTRY_METADATA_PATH)

            os.makedirs(self.config.baseline_download_dir, exist_ok=True)

            logging.info("[BASELINE FETCHER] Initialized successfully.")

        except Exception as e:
            logging.exception("[BASELINE FETCHER] Initialization failed.")
            raise CustomerChurnException(e, sys) from e

    def _sync_registry(self) -> None:
        """Pull the latest Model Registry state from S3."""
        try:
            logging.info("[BASELINE FETCHER] Syncing registry from S3...")
            self.s3_sync.sync_folder_from_s3(
                folder=self.local_registry_dir,
                aws_bucket_url=self.s3_registry_uri,
            )
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def _get_champion_baseline_path(self) -> tuple[str, str]:
        """
        Parse registry metadata to find the champion's baseline path.
        
        Returns
        -------
        tuple[str, str]
            (champion_version_name, absolute_path_to_baseline)
        """
        try:
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(f"Registry metadata not found at {self.metadata_path}")

            metadata = read_json_file(self.metadata_path)
            
            version = metadata.get("current_production_version")
            if not version:
                raise ValueError("No 'current_production_version' found in registry metadata.")

            version_info = metadata.get("versions_metadata", {}).get(version)
            if not version_info:
                raise ValueError(f"Metadata missing for champion version: {version}")

            baseline_source_path = version_info.get("baseline_path")
            if not baseline_source_path or not os.path.exists(baseline_source_path):
                raise FileNotFoundError(
                    f"Baseline file missing for version {version} at {baseline_source_path}"
                )

            return version, baseline_source_path

        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def initiate_baseline_fetch(self) -> BaselineFetchArtifact:
        """
        Execute the fetch operation and return the artifact.
        """
        try:
            logging.info("[BASELINE FETCHER] Execution started.")

            self._sync_registry()
            champion_version, source_path = self._get_champion_baseline_path()

            # Copy baseline to the current monitoring run's artifact directory
            destination_path = self.config.baseline_path
            shutil.copy2(source_path, destination_path)

            artifact = BaselineFetchArtifact(
                baseline_file_path=destination_path,
                champion_model_version=champion_version
            )

            logging.info(
                f"[BASELINE FETCHER] Completed. Champion: {champion_version} | "
                f"Path: {destination_path}"
            )
            return artifact

        except Exception as e:
            logging.exception("[BASELINE FETCHER] Execution failed.")
            raise CustomerChurnException(e, sys) from e