"""
Model Evaluation Pipeline (Registry-Only Architecture)

Responsibilities:
- Evaluate candidate models on unseen test data
- Select best candidate deterministically
- Evaluate current production (champion) model from registry metadata
- Perform champion–challenger comparison
- Produce approval decision
- Persist evaluation report and metadata

Design Guarantees:
- No dependency on production_model directory
- Registry metadata is single source of truth
- Deterministic decision logic
- Fair evaluation using identical test dataset
- Defensive handling of missing or partial registry state
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

from src.constants.pipeline_constants import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import load_object, read_json_file, write_json_file


class ModelEvaluation:
    """
    Production-grade model evaluation pipeline.
    """

    PIPELINE_VERSION = "1.0.0"

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        config: ModelEvaluationConfig,
        ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        try:
            logging.info("[MODEL EVALUATION INIT] Initializing")

            self.config = config
            self.ingestion_artifact = ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

            os.makedirs(self.config.evaluation_dir, exist_ok=True)

            logging.info(
                "[MODEL EVALUATION INIT] Completed | "
                f"pipeline_version={self.PIPELINE_VERSION}"
            )

        except Exception as e:
            logging.exception("[MODEL EVALUATION INIT] Failed")
            raise CustomerChurnException(e, sys)

    # ==========================================================
    # INTERNAL UTILITIES
    # ==========================================================

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return pd.read_csv(file_path)

    def _load_candidate_models(self) -> Dict[str, str]:
        """
        Discover trained candidate models.
        """

        models: Dict[str, str] = {}

        for model_name in os.listdir(
            self.model_trainer_artifact.trained_models_dir
        ):
            model_dir = os.path.join(
                self.model_trainer_artifact.trained_models_dir,
                model_name,
            )

            model_path = os.path.join(model_dir, "model.pkl")

            if os.path.exists(model_path):
                models[model_name] = model_path

        if not models:
            raise ValueError("No candidate models found for evaluation")

        logging.info(
            "[MODEL EVALUATION] Discovered %d candidate models",
            len(models),
        )

        return models

    # ==========================================================
    # METRIC COMPUTATION
    # ==========================================================

    def _compute_metrics(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.config.decision_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        return {
            "recall": round(recall_score(y_test, y_pred), 6),
            "precision": round(
                precision_score(y_test, y_pred, zero_division=0),
                6,
            ),
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

        tolerance = self.config.recall_tolerance

        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["metrics"]["recall"],
            reverse=True,
        )

        best_name, best_result = sorted_models[0]

        for name, result in sorted_models[1:]:

            recall_diff = (
                best_result["metrics"]["recall"]
                - result["metrics"]["recall"]
            )

            if abs(recall_diff) <= tolerance:
                if (
                    result["metrics"]["precision"]
                    > best_result["metrics"]["precision"]
                ):
                    best_name, best_result = name, result

        logging.info(
            "[MODEL EVALUATION] Best candidate selected: %s",
            best_name,
        )

        return best_name, best_result

    # ==========================================================
    # CHAMPION RESOLUTION
    # ==========================================================

    def _resolve_champion_model_path(self) -> Optional[str]:
        """
        Resolve current production model path from registry metadata.
        """

        metadata_path = self.config.model_registry_metadata_file_path

        if not os.path.exists(metadata_path):
            logging.info(
                "[MODEL EVALUATION] Registry metadata not found."
            )
            return None

        registry_info = read_json_file(metadata_path)

        current_version = registry_info.get("current_production_version")

        if not current_version:
            logging.info(
                "[MODEL EVALUATION] No production version set."
            )
            return None

        version_info = (
            registry_info.get("versions_metadata", {})
            .get(current_version)
        )

        if not version_info:
            logging.warning(
                "[MODEL EVALUATION] Production version metadata missing."
            )
            return None

        model_path = version_info.get("model_path")

        if not model_path or not os.path.exists(model_path):
            logging.warning(
                "[MODEL EVALUATION] Production model file missing."
            )
            return None

        return model_path

    def _evaluate_champion_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Optional[Dict[str, Any]]:

        model_path = self._resolve_champion_model_path()

        if not model_path:
            return None

        logging.info("[MODEL EVALUATION] Evaluating champion model")

        champion_model = load_object(model_path)

        return self._compute_metrics(
            champion_model,
            X_test,
            y_test,
        )

    # ==========================================================
    # CHAMPION–CHALLENGER COMPARISON
    # ==========================================================

    def _compare_with_champion(
        self,
        candidate_metrics: Dict[str, Any],
        champion_metrics: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:

        if champion_metrics is None:
            return True, "No production model available"

        candidate_recall = candidate_metrics["recall"]
        champion_recall = champion_metrics["recall"]

        improvement = candidate_recall - champion_recall

        if improvement >= self.config.min_recall_improvement:
            return True, f"Recall improved by {round(improvement, 6)}"

        return False, "Recall improvement below threshold"

    # ==========================================================
    # ENTRY POINT
    # ==========================================================

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:
            logging.info("[MODEL EVALUATION PIPELINE] Started")

            started_at_utc = datetime.now(timezone.utc).isoformat()

            test_df = self._read_csv(
                self.ingestion_artifact.test_file_path
            )

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            candidate_models = self._load_candidate_models()

            evaluation_results: Dict[str, Dict[str, Any]] = {}

            for model_name, model_path in candidate_models.items():

                logging.info(
                    "[MODEL EVALUATION] Evaluating model=%s",
                    model_name,
                )

                model = load_object(model_path)

                metrics = self._compute_metrics(
                    model,
                    X_test,
                    y_test,
                )

                evaluation_results[model_name] = {
                    "model_path": model_path,
                    "metrics": metrics,
                }

            best_model_name, best_result = self._select_best_candidate(
                evaluation_results
            )

            champion_metrics = self._evaluate_champion_model(
                X_test,
                y_test,
            )

            approved, reason = self._compare_with_champion(
                best_result["metrics"],
                champion_metrics,
            )

            completed_at_utc = datetime.now(timezone.utc).isoformat()

            report = {
                "pipeline": {
                    "name": "model_evaluation",
                    "version": self.PIPELINE_VERSION,
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

            write_json_file(
                self.config.evaluation_report_file_path,
                report,
            )

            metadata = {
                "pipeline_version": self.PIPELINE_VERSION,
                "decision_threshold": self.config.decision_threshold,
                "recall_tolerance": self.config.recall_tolerance,
                "min_recall_improvement":
                    self.config.min_recall_improvement,
                "best_model": best_model_name,
                "approved": approved,
                "created_at_utc":
                    datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(
                self.config.metadata_file_path,
                metadata,
            )

            artifact = ModelEvaluationArtifact(
                best_model_name=best_model_name,
                best_model_path=best_result["model_path"],
                evaluation_report_path=
                    self.config.evaluation_report_file_path,
                metadata_path=self.config.metadata_file_path,
                approval_status=approved,
            )

            logging.info(
                "[MODEL EVALUATION PIPELINE] Completed | approved=%s",
                approved,
            )

            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[MODEL EVALUATION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)