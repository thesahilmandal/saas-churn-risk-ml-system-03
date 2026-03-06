"""
Model Registry Pipeline (Production-Grade, Registry-Only Architecture)

Responsibilities:
- Promote approved models from evaluation pipeline
- Maintain immutable versioned storage
- Maintain registry metadata as single source of truth
- Manage production pointer via metadata
- Support safe rollback via pointer updates
- Preserve full lineage and audit trail

Design Guarantees:
- No production_model alias directory
- Immutable version directories
- Atomic metadata updates
- Deterministic promotion
- Structured logging and failure handling
"""

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import ModelRegistryConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file


class ModelRegistry:
    """
    Production-Grade Model Registry (Registry-Only Architecture).

    This registry:
    - Stores immutable model versions
    - Tracks current production version via metadata pointer
    - Supports rollback without file duplication
    - Ensures reproducibility and lineage integrity
    """

    PIPELINE_VERSION = "1.0.0"

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        config: ModelRegistryConfig,
        evaluation_artifact: Optional[ModelEvaluationArtifact] = None,
    ) -> None:
        """
        Initialize ModelRegistry.

        Args:
            config: Registry configuration.
            evaluation_artifact: Required for promotion,
                                 optional for rollback.
        """
        try:
            logging.info("[MODEL REGISTRY INIT] Initializing registry")

            self.config = config
            self.evaluation_artifact = evaluation_artifact

            os.makedirs(self.config.registry_dir, exist_ok=True)

            logging.info(
                "[MODEL REGISTRY INIT] Completed | "
                f"registry_dir={self.config.registry_dir}"
            )

        except Exception as e:
            logging.exception("[MODEL REGISTRY INIT] Failed")
            raise CustomerChurnException(e, sys)

    # ==========================================================
    # METADATA MANAGEMENT
    # ==========================================================

    def _initialize_metadata(self) -> Dict[str, Any]:
        """
        Create initial metadata structure.
        """
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
        """
        Load registry metadata or initialize if missing.
        """
        if not os.path.exists(self.config.registry_metadata_path):
            logging.info(
                "[MODEL REGISTRY] No metadata found. "
                "Initializing new registry."
            )
            return self._initialize_metadata()

        return read_json_file(self.config.registry_metadata_path)

    def _atomic_write_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Atomically write metadata to prevent partial corruption.
        """
        temp_path = f"{self.config.registry_metadata_path}.tmp"

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        os.replace(temp_path, self.config.registry_metadata_path)

    # ==========================================================
    # VERSION MANAGEMENT
    # ==========================================================

    @staticmethod
    def _compute_checksum(file_path: str) -> str:
        """
        Compute SHA256 checksum of a file.
        """
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _get_next_version(self, metadata: Dict[str, Any]) -> str:
        """
        Determine next semantic version (v1, v2, v3...).
        """
        versions = metadata.get("registered_versions", [])

        if not versions:
            return "v1"

        latest_version_number = max(int(v[1:]) for v in versions)
        return f"v{latest_version_number + 1}"

    def _register_model_version(self, version: str) -> str:
        """
        Register approved model under a new immutable version directory.
        """
        version_dir = os.path.join(self.config.registry_dir, version)

        if os.path.exists(version_dir):
            raise ValueError(
                f"Version directory already exists: {version}"
            )

        os.makedirs(version_dir, exist_ok=False)

        destination_model_path = os.path.join(version_dir, "model.pkl")

        shutil.copy2(
            self.evaluation_artifact.best_model_path,
            destination_model_path,
        )

        logging.info(
            "[MODEL REGISTRY] Model registered | version=%s",
            version,
        )

        return destination_model_path

    # ==========================================================
    # PROMOTION
    # ==========================================================

    def initiate_model_registry(self) -> Optional[str]:
        """
        Promote approved model to registry.

        Returns:
            New version string if promotion successful,
            otherwise None.
        """
        try:
            logging.info("[MODEL REGISTRY PIPELINE] Promotion started")

            if not self.evaluation_artifact:
                raise ValueError(
                    "Evaluation artifact required for promotion."
                )

            if not self.evaluation_artifact.approval_status:
                logging.info(
                    "[MODEL REGISTRY] Model not approved. "
                    "Skipping promotion."
                )
                return None

            metadata = self._load_registry_metadata()

            version = self._get_next_version(metadata)

            version_model_path = self._register_model_version(version)

            checksum = self._compute_checksum(version_model_path)

            model_size_mb = round(
                os.path.getsize(version_model_path) / (1024 * 1024),
                4,
            )

            # Update metadata
            metadata.setdefault("registered_versions", []).append(
                version
            )

            metadata.setdefault("versions_metadata", {})[version] = {
                "model_name":
                    self.evaluation_artifact.best_model_name,
                "model_path": version_model_path,
                "evaluation_report_path":
                    self.evaluation_artifact.evaluation_report_path,
                "evaluation_metadata_path":
                    self.evaluation_artifact.metadata_path,
                "model_size_mb": model_size_mb,
                "checksum_sha256": checksum,
                "registered_at_utc":
                    datetime.now(timezone.utc).isoformat(),
            }

            metadata["current_production_version"] = version

            metadata["last_updated_at_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )

            self._atomic_write_metadata(metadata)

            logging.info(
                "[MODEL REGISTRY PIPELINE] Promotion completed | "
                "production_version=%s",
                version,
            )

            return version

        except Exception as e:
            logging.exception(
                "[MODEL REGISTRY PIPELINE] Promotion failed"
            )
            raise CustomerChurnException(e, sys)

    # ==========================================================
    # ROLLBACK
    # ==========================================================

    def rollback_to_version(self, version: str) -> str:
        """
        Rollback production pointer to an existing version.

        Args:
            version: Target version (e.g., "v2")

        Returns:
            Version string after rollback.
        """
        try:
            logging.info(
                "[MODEL REGISTRY] Rollback requested | version=%s",
                version,
            )

            metadata = self._load_registry_metadata()

            if version not in metadata.get("registered_versions", []):
                raise ValueError(
                    f"Version '{version}' not found in registry."
                )

            metadata["current_production_version"] = version

            metadata["last_updated_at_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )

            metadata.setdefault("rollback_history", []).append(
                {
                    "rolled_back_to": version,
                    "timestamp_utc":
                        datetime.now(timezone.utc).isoformat(),
                }
            )

            self._atomic_write_metadata(metadata)

            logging.info(
                "[MODEL REGISTRY] Rollback successful | "
                "production_version=%s",
                version,
            )

            return version

        except Exception as e:
            logging.exception("[MODEL REGISTRY] Rollback failed")
            raise CustomerChurnException(e, sys)