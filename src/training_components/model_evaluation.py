"""
Model Evaluation Pipeline (S3-Backed Registry Architecture)

Responsibilities:
- Evaluate candidate models on unseen test data
- Select best candidate deterministically
- Fetch production champion metadata and model from S3 Registry
- Evaluate current champion model (gracefully handling schema evolution)
- Perform champion–challenger comparison
- Produce approval decision
- Persist evaluation report and metadata

Design Guarantees:
- Zero dependency on local production directories
- S3 is the single source of truth for champion models
- Robust handling of 'Cold Starts' (first run), 'Ghost Champions', and schema mismatches
- Deterministic decision logic
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import (
    load_object, 
    read_json_file,
    write_json_file,
    read_csv_file
)


class ModelEvaluation:
    """
    Production-grade model evaluation pipeline utilizing S3 for champion resolution.
    """

    PIPELINE_VERSION = "2.0.0"

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        config: ModelEvaluationConfig,
        ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        """
        Initialize the Model Evaluation pipeline.
        """
        try:
            logging.info("[MODEL EVALUATION INIT] Initializing")

            self.config = config
            self.ingestion_artifact = ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            
            self.s3_sync = S3Sync()

            # Ensure local evaluation directory exists
            os.makedirs(self.config.evaluation_dir, exist_ok=True)
            
            # Create a localized cache directory for downloaded S3 champion artifacts
            self.champion_cache_dir = os.path.join(
                self.config.evaluation_dir, "champion_cache"
            )
            os.makedirs(self.champion_cache_dir, exist_ok=True)

            logging.info(
                "[MODEL EVALUATION INIT] Completed | pipeline_version=%s",
                self.PIPELINE_VERSION,
            )

        except Exception as e:
            logging.exception("[MODEL EVALUATION INIT] Failed")
            raise CustomerChurnException(e, sys) from e

    # ==========================================================
    # INTERNAL UTILITIES
    # ==========================================================

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        """Read CSV file safely."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    def _load_candidate_models(self) -> Dict[str, str]:
        """
        Discover trained candidate models from the upstream training pipeline.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping model names to their file paths.
        """
        models: Dict[str, str] = {}
        models_dir = self.model_trainer_artifact.trained_models_dir

        for model_name in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, model_name)
            model_path = os.path.join(model_dir, "model.pkl")

            if os.path.exists(model_path):
                models[model_name] = model_path

        if not models:
            raise ValueError("No candidate models found for evaluation in trained_models_dir.")

        logging.info("[MODEL EVALUATION] Discovered %d candidate models", len(models))
        return models

    # ==========================================================
    # METRIC COMPUTATION
    # ==========================================================

    def _compute_metrics(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Compute standardized evaluation metrics for a given model.
        """
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.config.decision_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        return {
            "recall": round(recall_score(y_test, y_pred), 6),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 6),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 6),
            "pr_auc": round(average_precision_score(y_test, y_prob), 6),
            "log_loss": round(log_loss(y_test, y_prob), 6),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    # ==========================================================
    # CANDIDATE SELECTION
    # ==========================================================

    def _select_best_candidate(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best candidate model based on recall and precision thresholds.
        """
        tolerance = self.config.recall_tolerance

        # Sort primarily by recall (descending)
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["metrics"]["recall"],
            reverse=True,
        )

        best_name, best_result = sorted_models[0]

        # Tie-breaker logic using Precision within a given Recall tolerance
        for name, result in sorted_models[1:]:
            recall_diff = best_result["metrics"]["recall"] - result["metrics"]["recall"]

            if abs(recall_diff) <= tolerance:
                if result["metrics"]["precision"] > best_result["metrics"]["precision"]:
                    best_name, best_result = name, result

        logging.info("[MODEL EVALUATION] Best candidate selected: %s", best_name)
        return best_name, best_result

    # ==========================================================
    # S3 REGISTRY RESOLUTION
    # ==========================================================

    def _fetch_registry_metadata_from_s3(self) -> Optional[Dict[str, Any]]:
        """
        Fetch registry metadata from S3. Handles 'Cold Start' scenario gracefully.
        """
        s3_metadata_uri = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_REGISTRY_DIR_NAME}/registry_metadata.json"
        local_cache_path = os.path.join(self.champion_cache_dir, "registry_metadata.json")

        try:
            logging.info("[MODEL EVALUATION] Fetching registry metadata from S3: %s", s3_metadata_uri)
            self.s3_sync.download_file(s3_metadata_uri, local_cache_path)
            
            if os.path.exists(local_cache_path):
                return read_json_file(local_cache_path)
            return None

        except CustomerChurnException as e:
            # Typical for a first run when the registry directory doesn't exist yet in S3
            logging.warning("[MODEL EVALUATION] Registry metadata not found in S3 (Cold start detected).")
            return None
        except Exception as e:
            logging.error("[MODEL EVALUATION] Unexpected error fetching metadata: %s", e)
            return None

    def _resolve_champion_model_path_from_s3(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Determine current production version from metadata and download the model artifact.
        Handles 'Ghost Champion' scenario gracefully.
        """
        current_version = metadata.get("current_production_version")
        if not current_version:
            logging.info("[MODEL EVALUATION] No production version set in registry metadata.")
            return None

        version_info = metadata.get("versions_metadata", {}).get(current_version)
        if not version_info:
            logging.warning("[MODEL EVALUATION] Production version metadata missing for %s.", current_version)
            return None

        # Reconstruct the S3 URI from the relative path stored in the registry metadata
        s3_model_relative_path = version_info.get("model_path")
        if not s3_model_relative_path:
            logging.warning("[MODEL EVALUATION] Model path missing in version metadata.")
            return None

        # Clean the relative path to ensure valid S3 URI construction
        s3_model_relative_path = s3_model_relative_path.lstrip("./\\")
        s3_model_uri = f"s3://{S3_BUCKET_NAME}/{s3_model_relative_path}"
        
        local_model_path = os.path.join(self.champion_cache_dir, current_version, "champion_model.pkl")

        try:
            logging.info("[MODEL EVALUATION] Downloading champion model from S3: %s", s3_model_uri)
            self.s3_sync.download_file(s3_model_uri, local_model_path)
            
            if os.path.exists(local_model_path):
                return local_model_path
            return None

        except CustomerChurnException as e:
            logging.error("[MODEL EVALUATION] Failed to download champion model from S3 (Ghost Champion).")
            return None
        except Exception as e:
            logging.error("[MODEL EVALUATION] Unexpected error downloading champion model: %s", e)
            return None

    def _evaluate_champion_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch and evaluate the current champion model.
        Protects against Schema Evolution failures.
        """
        metadata = self._fetch_registry_metadata_from_s3()
        if not metadata:
            return None

        model_path = self._resolve_champion_model_path_from_s3(metadata)
        if not model_path:
            return None

        logging.info("[MODEL EVALUATION] Evaluating S3 champion model")

        try:
            champion_model = load_object(model_path)
            return self._compute_metrics(champion_model, X_test, y_test)
            
        except ValueError as ve:
            # Handles Scikit-learn schema mismatch errors
            logging.warning("[MODEL EVALUATION] Champion model failed evaluation due to schema mismatch: %s", ve)
            return None
        except Exception as e:
            logging.warning("[MODEL EVALUATION] Champion model failed evaluation unexpectedly: %s", e)
            return None

    # ==========================================================
    # CHAMPION–CHALLENGER COMPARISON
    # ==========================================================

    def _compare_with_champion(
        self,
        candidate_metrics: Dict[str, Any],
        champion_metrics: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        Compare challenger against the current production champion.
        """
        if champion_metrics is None:
            return True, "Approved automatically: No valid production model available (Cold Start / Registry Error)"

        candidate_recall = candidate_metrics["recall"]
        champion_recall = champion_metrics["recall"]

        improvement = candidate_recall - champion_recall

        if improvement >= self.config.min_recall_improvement:
            return True, f"Approved: Recall improved by {round(improvement, 6)} over Champion"

        return False, f"Rejected: Recall improvement ({round(improvement, 6)}) below threshold"

    # ==========================================================
    # ENTRY POINT
    # ==========================================================

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Execute the Model Evaluation Pipeline.
        """
        try:
            logging.info("[MODEL EVALUATION PIPELINE] Started")

            started_at_utc = datetime.now(timezone.utc).isoformat()

            # Load Test Data
            test_df = self._read_csv(self.ingestion_artifact.test_file_path)
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # Evaluate all local candidate models
            candidate_models = self._load_candidate_models()
            evaluation_results: Dict[str, Dict[str, Any]] = {}

            for model_name, model_path in candidate_models.items():
                logging.info("[MODEL EVALUATION] Evaluating local candidate model=%s", model_name)
                
                model = load_object(model_path)
                metrics = self._compute_metrics(model, X_test, y_test)

                evaluation_results[model_name] = {
                    "model_path": model_path,
                    "metrics": metrics,
                }

            # Select the Best Challenger
            best_model_name, best_result = self._select_best_candidate(evaluation_results)

            # Evaluate Champion from S3
            champion_metrics = self._evaluate_champion_model(X_test, y_test)

            # Compare Challenger vs Champion
            approved, reason = self._compare_with_champion(
                candidate_metrics=best_result["metrics"],
                champion_metrics=champion_metrics,
            )

            completed_at_utc = datetime.now(timezone.utc).isoformat()

            # Compile Evaluation Report
            report = {
                "pipeline": {
                    "name": "model_evaluation",
                    "version": self.PIPELINE_VERSION,
                    "architecture": "s3_backed_registry",
                },
                "timing": {
                    "started_at_utc": started_at_utc,
                    "completed_at_utc": completed_at_utc,
                },
                "best_model": best_model_name,
                "approved": approved,
                "reason": reason,
                "candidate_results": evaluation_results,
                "champion_metrics": champion_metrics,
            }

            write_json_file(self.config.evaluation_report_file_path, report)

            # Compile Pipeline Metadata
            metadata = {
                "pipeline_version": self.PIPELINE_VERSION,
                "decision_threshold": self.config.decision_threshold,
                "recall_tolerance": self.config.recall_tolerance,
                "min_recall_improvement": self.config.min_recall_improvement,
                "best_model": best_model_name,
                "approved": approved,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(self.config.metadata_file_path, metadata)

            # Create Artifact
            artifact = ModelEvaluationArtifact(
                best_model_name=best_model_name,
                best_model_path=best_result["model_path"],
                evaluation_report_path=self.config.evaluation_report_file_path,
                metadata_path=self.config.metadata_file_path,
                approval_status=approved,
            )

            logging.info("[MODEL EVALUATION PIPELINE] Completed | approved=%s", approved)
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[MODEL EVALUATION PIPELINE] Failed")
            raise CustomerChurnException(e, sys) from e