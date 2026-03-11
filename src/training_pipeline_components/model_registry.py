"""
Model Registry Pipeline (Production-Grade, Registry-Only Architecture)

Promotes approved models into an immutable model registry.
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

from src.constants.pipeline_constants import TARGET_COLUMN
from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    DataIngestionArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelRegistryConfig
from src.training_pipeline_components.baseline_generator import BaselineGenerator
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import (
    read_json_file,
    write_json_file,
    read_csv_file,
    load_object,
)


class ModelRegistry:
    """
    Production-grade Model Registry.

    Responsibilities
    ----------------
    - Register approved models
    - Maintain immutable version directories
    - Store monitoring artifacts
    - Maintain registry metadata
    """

    PIPELINE_VERSION = "1.0.0"

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
            logging.info("Initializing Model Registry")

            self.config = config
            self.ingestion_artifact = ingestion_artifact
            self.trainer_artifact = trainer_artifact
            self.evaluation_artifact = evaluation_artifact

            os.makedirs(self.config.registry_dir, exist_ok=True)

            logging.info(
                "Model Registry initialized | registry_dir=%s",
                self.config.registry_dir,
            )

        except Exception as exc:
            logging.exception("Model Registry initialization failed")
            raise CustomerChurnException(exc, sys) from exc

    # ==========================================================
    # METADATA MANAGEMENT
    # ==========================================================

    def _initialize_metadata(self) -> Dict[str, Any]:
        """Create initial metadata structure."""

        return {
            "registry_version": "1.0",
            "current_production_version": None,
            "registered_versions": [],
            "versions_metadata": {},
            "rollback_history": [],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_updated_at_utc": None,
        }

    def _load_registry_metadata(self) -> Dict[str, Any]:
        """Load registry metadata or initialize new registry."""

        if not os.path.exists(self.config.registry_metadata_path):
            logging.info("Registry metadata not found. Initializing.")
            return self._initialize_metadata()

        return read_json_file(self.config.registry_metadata_path)

    def _atomic_write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Safely write metadata to disk using atomic operation."""

        try:
            temp_path = f"{self.config.registry_metadata_path}.tmp"

            os.makedirs(os.path.dirname(self.config.registry_metadata_path), exist_ok=True)

            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4)

            os.replace(temp_path, self.config.registry_metadata_path)

        except Exception as exc:
            logging.exception("Failed writing registry metadata")
            raise CustomerChurnException(exc, sys) from exc

    # ==========================================================
    # VERSION MANAGEMENT
    # ==========================================================

    @staticmethod
    def _compute_checksum(file_path: str) -> str:
        """Compute SHA256 checksum."""

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _get_next_version(self, metadata: Dict[str, Any]) -> str:
        """Determine next version identifier."""

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
        """Generate dataset baseline."""

        try:
            logging.info("Generating baseline report")

            generator = BaselineGenerator()
            baseline = generator.generate_baseline_report(train_df)

            path = os.path.join(version_dir, "baseline.json")
            write_json_file(path, baseline)

            return path

        except Exception as exc:
            logging.exception("Baseline generation failed")
            raise CustomerChurnException(exc, sys) from exc

    def _generate_and_store_metrics(self, version_dir: str) -> str:
        """Store evaluation metrics."""

        try:
            if not self.evaluation_artifact:
                raise ValueError("Evaluation artifact required.")

            report = read_json_file(
                self.evaluation_artifact.evaluation_report_path
            )

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
            logging.exception("Metrics generation failed")
            raise CustomerChurnException(exc, sys) from exc

    def _generate_and_store_requirements(self, version_dir: str) -> str:
        """Capture Python environment."""

        try:
            logging.info("Capturing environment requirements")

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
            logging.exception("pip freeze failed")
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
                    # FIX: Cast the numpy array to a standard Python list
                    feature_names = list(preprocessor.get_feature_names_out())
                except Exception as e:
                    logging.warning(f"[MODEL REGISTRY] Could not extract names from preprocessor: {e}")
                    feature_names = [] # Reset to empty list on failure
            
            # Fallback if no preprocessor or extraction failed
            # Now len() and boolean checks are perfectly safe
            if len(feature_names) == 0:
                base_cols = [col for col in train_df.columns if col != TARGET_COLUMN]
                # If lengths match, use base columns, otherwise fallback to index
                if len(base_cols) == len(raw_importances):
                    feature_names = base_cols
                else:
                    feature_names = [f"feature_{i}" for i in range(len(raw_importances))]

            # 4. Map names to values and clean up Scikit-Learn prefixes (e.g., 'num__TotalCharges')
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
        """Register model version and store artifacts."""

        version_dir = os.path.join(self.config.registry_dir, version)

        if os.path.exists(version_dir):
            raise ValueError(f"Version directory already exists: {version}")

        os.makedirs(version_dir)

        try:
            model_path = os.path.join(version_dir, "model.pkl")

            shutil.copy2(
                self.evaluation_artifact.best_model_path,
                model_path,
            )

            train_df = read_csv_file(self.ingestion_artifact.train_file_path)

            baseline_path = self._generate_and_store_baseline(
                version_dir, train_df
            )

            metrics_path = self._generate_and_store_metrics(version_dir)

            feature_path = self._generate_and_store_feature_importance(
                version_dir,
                train_df,
            )

            requirements_path = self._generate_and_store_requirements(
                version_dir
            )

            return {
                "model_path": model_path,
                "baseline_path": baseline_path,
                "metrics_path": metrics_path,
                "feature_importance_path": feature_path,
                "requirements_path": requirements_path,
            }

        except Exception as exc:
            logging.exception("Model registration failed")

            if os.path.exists(version_dir):
                shutil.rmtree(version_dir, ignore_errors=True)

            raise CustomerChurnException(exc, sys) from exc

    # ==========================================================
    # PROMOTION
    # ==========================================================

    def initiate_model_registry(self) -> Optional[str]:
        """Promote approved model into registry."""

        try:
            logging.info("Model promotion started")

            if not self.evaluation_artifact:
                raise ValueError("Evaluation artifact required.")

            if not self.evaluation_artifact.approval_status:
                logging.info("Model not approved. Skipping promotion.")
                return None

            metadata = self._load_registry_metadata()

            version = self._get_next_version(metadata)

            artifact_paths = self._register_model_version(version)

            checksum = self._compute_checksum(
                artifact_paths["model_path"]
            )

            model_size_mb = round(
                os.path.getsize(artifact_paths["model_path"])
                / (1024 * 1024),
                4,
            )

            metadata.setdefault("registered_versions", []).append(version)

            metadata.setdefault("versions_metadata", {})[version] = {
                "model_name": self.evaluation_artifact.best_model_name,
                "model_path": artifact_paths["model_path"],
                "baseline_path": artifact_paths["baseline_path"],
                "metrics_path": artifact_paths["metrics_path"],
                "feature_importance_path": artifact_paths[
                    "feature_importance_path"
                ],
                "requirements_path": artifact_paths["requirements_path"],
                "model_size_mb": model_size_mb,
                "checksum_sha256": checksum,
                "registered_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            metadata["current_production_version"] = version
            metadata["last_updated_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()

            self._atomic_write_metadata(metadata)

            logging.info(
                "Model promotion completed | version=%s",
                version,
            )

            return version

        except Exception as exc:
            logging.exception("Model promotion failed")
            raise CustomerChurnException(exc, sys) from exc