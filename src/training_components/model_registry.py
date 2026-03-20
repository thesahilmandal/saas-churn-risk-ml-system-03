"""
Model Registry Pipeline (Production-Grade, S3-Backed Architecture)

Promotes approved models into an immutable model registry.
Synchronizes with an S3-backed single source of truth to prevent
version collisions across distributed training runs, and immediately
pushes updates to S3 to guarantee state consistency.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.cloud.s3_syncer import S3Sync
from src.constants.pipeline_constants import (
    S3_BUCKET_NAME,
    S3_MODEL_REGISTRY_DIR_NAME,
    TARGET_COLUMN,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelRegistryConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.training_components.baseline_generator import BaselineGenerator
from src.utils.main_utils import (
    load_object,
    read_csv_file,
    read_json_file,
    write_json_file,
)


class ModelRegistry:
    """
    Production-grade Model Registry utilizing S3 as the Single Source of Truth.

    Responsibilities
    ----------------
    - Fetch remote registry metadata to ensure version consistency
    - Register newly approved models
    - Maintain immutable version directories locally
    - Store monitoring artifacts (baselines, feature importance)
    - Maintain and update registry metadata safely
    - Immediately sync the updated local registry back to S3
    """

    PIPELINE_VERSION = "2.0.0"

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        config: ModelRegistryConfig,
        ingestion_artifact: DataIngestionArtifact,
        trainer_artifact: ModelTrainerArtifact,
        evaluation_artifact: Optional[ModelEvaluationArtifact] = None,
    ) -> None:
        """Initialize Model Registry."""

        try:
            logging.info("[MODEL REGISTRY INIT] Initializing")

            self.config = config
            self.ingestion_artifact = ingestion_artifact
            self.trainer_artifact = trainer_artifact
            self.evaluation_artifact = evaluation_artifact
            
            self.s3_sync = S3Sync()

            os.makedirs(self.config.registry_dir, exist_ok=True)

            logging.info(
                "[MODEL REGISTRY INIT] Initialized | registry_dir=%s",
                self.config.registry_dir,
            )

        except Exception as exc:
            logging.exception("[MODEL REGISTRY INIT] Failed")
            raise CustomerChurnException(exc, sys) from exc

    # ==========================================================
    # METADATA MANAGEMENT
    # ==========================================================

    def _initialize_metadata(self) -> Dict[str, Any]:
        """Create initial metadata structure for a cold start."""
        return {
            "registry_version": "2.0",
            "current_production_version": None,
            "registered_versions": [],
            "versions_metadata": {},
            "rollback_history": [],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_updated_at_utc": None,
        }

    def _load_registry_metadata(self) -> Dict[str, Any]:
        """
        Load registry metadata directly from S3 to ensure distributed consistency.
        Falls back to initialization if S3 metadata does not exist (Cold Start).
        """
        s3_metadata_uri = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_REGISTRY_DIR_NAME}/registry_metadata.json"
        
        try:
            logging.info("[MODEL REGISTRY] Fetching remote metadata from S3: %s", s3_metadata_uri)
            self.s3_sync.download_file(s3_metadata_uri, self.config.registry_metadata_path)
            
            if os.path.exists(self.config.registry_metadata_path):
                return read_json_file(self.config.registry_metadata_path)
            
            logging.info("[MODEL REGISTRY] Remote metadata not found. Initializing new registry.")
            return self._initialize_metadata()

        except CustomerChurnException:
            # S3 file doesn't exist yet (first run) or permission issue
            logging.warning("[MODEL REGISTRY] Failed to fetch remote metadata (Cold Start). Initializing new registry.")
            return self._initialize_metadata()
        except Exception as exc:
            logging.exception("[MODEL REGISTRY] Unexpected error fetching metadata.")
            raise CustomerChurnException(exc, sys) from exc

    def _atomic_write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Safely write metadata to local disk using an atomic operation."""
        try:
            temp_path = f"{self.config.registry_metadata_path}.tmp"

            os.makedirs(os.path.dirname(self.config.registry_metadata_path), exist_ok=True)

            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4)

            os.replace(temp_path, self.config.registry_metadata_path)
            logging.info("[MODEL REGISTRY] Local metadata updated atomically.")

        except Exception as exc:
            logging.exception("[MODEL REGISTRY] Failed writing local registry metadata")
            raise CustomerChurnException(exc, sys) from exc

    # ==========================================================
    # VERSION MANAGEMENT
    # ==========================================================

    @staticmethod
    def _compute_checksum(file_path: str) -> str:
        """Compute SHA256 checksum to ensure artifact integrity."""
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _get_next_version(self, metadata: Dict[str, Any]) -> str:
        """Determine next version identifier based on SSOT metadata."""
        versions = metadata.get("registered_versions", [])

        if not versions:
            return "v1"

        latest = max(int(v[1:]) for v in versions)
        return f"v{latest + 1}"

    # ==========================================================
    # ARTIFACT GENERATION
    # ==========================================================

    def _generate_and_store_baseline(
        self,
        version_dir: str,
        train_df: pd.DataFrame,
    ) -> str:
        """Generate dataset statistical baseline for data drift monitoring."""
        try:
            logging.info("[MODEL REGISTRY] Generating baseline report")

            generator = BaselineGenerator()
            baseline = generator.generate_baseline_report(train_df)

            path = os.path.join(version_dir, "baseline.json")
            write_json_file(path, baseline)

            return path

        except Exception as exc:
            logging.exception("[MODEL REGISTRY] Baseline generation failed")
            raise CustomerChurnException(exc, sys) from exc

    def _generate_and_store_metrics(self, version_dir: str) -> str:
        """Store evaluation metrics for the approved candidate."""
        try:
            if not self.evaluation_artifact:
                raise ValueError("Evaluation artifact required.")

            report = read_json_file(self.evaluation_artifact.evaluation_report_path)
            best_model = report["best_model"]
            metrics = report["candidate_results"][best_model]["metrics"]

            metrics_report = {
                "model_name": best_model,
                "metrics": metrics,
            }

            path = os.path.join(version_dir, "metrics.json")
            write_json_file(path, metrics_report)

            return path

        except Exception as exc:
            logging.exception("[MODEL REGISTRY] Metrics generation failed")
            raise CustomerChurnException(exc, sys) from exc

    def _generate_and_store_requirements(self, version_dir: str) -> str:
        """Capture Python environment requirements for reproducibility."""
        try:
            logging.info("[MODEL REGISTRY] Capturing environment requirements")

            path = os.path.join(version_dir, "requirements.txt")

            with open(path, "w", encoding="utf-8") as file:
                subprocess.run(
                    [sys.executable, "-m", "pip", "freeze"],
                    stdout=file,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )

            return path

        except subprocess.CalledProcessError as exc:
            logging.exception("[MODEL REGISTRY] pip freeze failed")
            raise CustomerChurnException(exc, sys) from exc

    def _generate_and_store_feature_importance(self, version_dir: str, train_df: pd.DataFrame) -> str:
        """Extract, map, and store human-readable feature importances."""
        try:
            logging.info("[MODEL REGISTRY] Generating human-readable feature importance report")
            pipeline = load_object(self.evaluation_artifact.best_model_path)
            
            # 1. Separate the preprocessor from the model
            if hasattr(pipeline, "steps"):
                model = pipeline.steps[-1][1]
                preprocessor = pipeline[:-1]
            else:
                model = pipeline
                preprocessor = None
                
            # 2. Extract raw importances
            raw_importances = []
            if hasattr(model, "feature_importances_"):
                raw_importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                raw_importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            else:
                path = os.path.join(version_dir, "feature_importance.json")
                write_json_file(path, {"message": "Model type does not support native feature importance extraction."})
                return path

            # 3. Attempt to get actual feature names from the preprocessor
            feature_names = []
            if preprocessor is not None:
                try:
                    feature_names = list(preprocessor.get_feature_names_out())
                except Exception as e:
                    logging.warning(f"[MODEL REGISTRY] Could not extract names from preprocessor: {e}")
                    feature_names = []
            
            # Fallback if no preprocessor or extraction failed
            if len(feature_names) == 0:
                base_cols = [col for col in train_df.columns if col != TARGET_COLUMN]
                if len(base_cols) == len(raw_importances):
                    feature_names = base_cols
                else:
                    feature_names = [f"feature_{i}" for i in range(len(raw_importances))]

            # 4. Map names to values and clean up Scikit-Learn prefixes
            importances = {}
            for i, val in enumerate(raw_importances):
                clean_name = feature_names[i].split('__')[-1] if i < len(feature_names) else f"feature_{i}"
                importances[clean_name] = round(float(val), 5)

            # 5. Sort the dictionary by importance descending for readability
            sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

            path = os.path.join(version_dir, "feature_importance.json")
            write_json_file(path, sorted_importances)
            
            logging.info("[MODEL REGISTRY] Human-readable feature importance stored | path=%s", path)
            return path

        except Exception as exc:
            logging.exception("[MODEL REGISTRY] Feature importance generation failed")
            raise CustomerChurnException(exc, sys) from exc
        
    # ==========================================================
    # MODEL VERSION REGISTRATION
    # ==========================================================

    def _register_model_version(self, version: str) -> Dict[str, str]:
        """Create version directory and store all associated artifacts locally."""
        version_dir = os.path.join(self.config.registry_dir, version)

        if os.path.exists(version_dir):
            raise ValueError(f"Version directory already exists locally: {version}")

        os.makedirs(version_dir)

        try:
            model_path = os.path.join(version_dir, "model.pkl")

            shutil.copy2(
                self.evaluation_artifact.best_model_path,
                model_path,
            )

            train_df = read_csv_file(self.ingestion_artifact.train_file_path)

            baseline_path = self._generate_and_store_baseline(version_dir, train_df)
            metrics_path = self._generate_and_store_metrics(version_dir)
            feature_path = self._generate_and_store_feature_importance(version_dir, train_df)
            requirements_path = self._generate_and_store_requirements(version_dir)

            return {
                "model_path": model_path,
                "baseline_path": baseline_path,
                "metrics_path": metrics_path,
                "feature_importance_path": feature_path,
                "requirements_path": requirements_path,
            }

        except Exception as exc:
            logging.exception("[MODEL REGISTRY] Local artifact registration failed")

            if os.path.exists(version_dir):
                shutil.rmtree(version_dir, ignore_errors=True)

            raise CustomerChurnException(exc, sys) from exc

    # ==========================================================
    # PROMOTION
    # ==========================================================

    def initiate_model_registry(self) -> Optional[str]:
        """Promote approved model into the registry and sync to S3."""
        try:
            logging.info("[MODEL REGISTRY PIPELINE] Started")

            if not self.evaluation_artifact:
                raise ValueError("Evaluation artifact required.")

            if not self.evaluation_artifact.approval_status:
                logging.info("[MODEL REGISTRY] Candidate model was not approved. Skipping promotion.")
                return None

            # 1. Fetch remote SSOT metadata
            metadata = self._load_registry_metadata()

            # 2. Determine next consistent version
            version = self._get_next_version(metadata)

            # 3. Build local artifacts
            artifact_paths = self._register_model_version(version)

            checksum = self._compute_checksum(artifact_paths["model_path"])

            model_size_mb = round(
                os.path.getsize(artifact_paths["model_path"]) / (1024 * 1024),
                4,
            )

            # 4. Update Metadata
            # Ensure relative paths are stored so S3 URI construction in downstream apps works
            relative_base = f"{S3_MODEL_REGISTRY_DIR_NAME}/{version}"
            
            metadata.setdefault("registered_versions", []).append(version)
            metadata.setdefault("versions_metadata", {})[version] = {
                "model_name": self.evaluation_artifact.best_model_name,
                "model_path": f"{relative_base}/model.pkl",
                "baseline_path": f"{relative_base}/baseline.json",
                "metrics_path": f"{relative_base}/metrics.json",
                "feature_importance_path": f"{relative_base}/feature_importance.json",
                "requirements_path": f"{relative_base}/requirements.txt",
                "model_size_mb": model_size_mb,
                "checksum_sha256": checksum,
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            metadata["current_production_version"] = version
            metadata["last_updated_at_utc"] = datetime.now(timezone.utc).isoformat()

            # 5. Save updated metadata locally
            self._atomic_write_metadata(metadata)

            # 6. Immediately sync the local registry to S3 to maintain SSOT
            s3_registry_uri = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_REGISTRY_DIR_NAME}"
            logging.info("[MODEL REGISTRY] Syncing updated registry to S3: %s", s3_registry_uri)
            
            # Using aws s3 sync handles both the new version folder and the updated metadata file
            self.s3_sync.sync_folder_to_s3(
                folder=self.config.registry_dir,
                aws_bucket_url=s3_registry_uri,
            )
            
            logging.info("[MODEL REGISTRY] S3 sync completed successfully.")

            logging.info(
                "[MODEL REGISTRY PIPELINE] Completed | Promoted version=%s",
                version,
            )

            return version

        except Exception as exc:
            logging.exception("[MODEL REGISTRY PIPELINE] Failed")
            raise CustomerChurnException(exc, sys) from exc