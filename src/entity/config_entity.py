import os
import sys
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

from src.constants import pipeline_constants
from src.exception import CustomerChurnException

load_dotenv()


class TrainingPipelineConfig:
    """
    Root configuration for training pipeline.
    Responsible for creating base artifact directory.
    """

    def __init__(self) -> None:
        try:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.artifact_name: str = (
                pipeline_constants.ARTIFACT_DIR / "training_pipeline"
            )

            self.artifact_dir: str = os.path.join(
                self.artifact_name,
                self.run_id,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# ETL Config
# ============================================================


class ETLconfig:
    """Configuration for ETL stage."""

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig,
    ) -> None:
        try:
            self.etl_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.ETL_DIR_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.etl_dir,
                pipeline_constants.ETL_METADATA_FILE_NAME,
            )

            self.raw_data_dir: str = os.path.join(
                self.etl_dir,
                pipeline_constants.ETL_RAW_DATA_DIR_NAME,
            )

            self.database_url: str = os.getenv("MONGODB_URL")
            self.database_name: str = os.getenv("MONGODB_DATABASE")
            self.collection_name: str = os.getenv("MONGODB_RAW_COLLECTION")
            self.data_source: str = os.getenv("DATA_SOURCE")

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Data Ingestion Config
# ============================================================


class DataIngestionConfig:
    """Configuration for data ingestion stage."""

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig,
    ) -> None:
        try:
            self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.DATA_INGESTION_DIR_NAME,
            )

            self.train_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_TRAIN_FILE_NAME,
            )

            self.test_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_TEST_FILE_NAME,
            )

            self.val_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_VAL_FILE_NAME,
            )

            self.schema_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_SCHEMA_FILE_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_METADATA_FILE_NAME,
            )

            self.train_temp_split_ratio: float = (
                pipeline_constants.DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO
            )

            self.test_val_split_ratio: float = (
                pipeline_constants.DATA_INGESTION_TEST_VAL_SPLIT_RATIO
            )

            self.random_state: int = pipeline_constants.RANDOM_STATE

            self.database_name: str = os.getenv("MONGODB_DATABASE")
            self.collection_name: str = os.getenv("MONGODB_RAW_COLLECTION")
            self.database_url: str = os.getenv("MONGODB_URL")

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Data Validation Config
# ============================================================


class DataValidationConfig:
    """Configuration for data validation stage."""

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig,
    ) -> None:
        try:
            self.data_validation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.DATA_VALIDATION_DIR_NAME,
            )

            self.validation_report_file_path: str = os.path.join(
                self.data_validation_dir,
                pipeline_constants.DATA_VALIDATION_REPORT_FILE_NAME,
            )

            self.reference_schema_file_path: str = pipeline_constants.REFERENCE_SCHEMA_PATH

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Data Transformation Config
# ============================================================


class DataTransformationConfig:
    """Configuration for data transformation stage."""

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig,
    ) -> None:
        try:
            self.data_transformation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.DATA_TRANSFORMATION_DIR_NAME,
            )

            self.lr_preprocessor_file_path: str = os.path.join(
                self.data_transformation_dir,
                pipeline_constants
                .DATA_TRANSFORMATION_LINEAR_PREPROCESSOR_FILE_NAME,
            )

            self.tree_preprocessor_file_path: str = os.path.join(
                self.data_transformation_dir,
                pipeline_constants
                .DATA_TRANSFORMATION_TREE_PREPROCESSOR_FILE_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.data_transformation_dir,
                pipeline_constants.DATA_TRANSFORMATION_METADATA_FILE_NAME,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Model Training Config
# ============================================================


class ModelTrainingConfig:
    """Configuration for model training stage."""

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig,
    ) -> None:
        try:
            self.model_trainer_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.MODEL_TRAINING_DIR_NAME,
            )

            self.trained_models_dir: str = os.path.join(
                self.model_trainer_dir,
                pipeline_constants
                .MODEL_TRAINING_TRAINED_MODELS_DIR_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.model_trainer_dir,
                pipeline_constants.MODEL_TRAINING_METADATA_FILE_NAME,
            )

            self.models = (
                pipeline_constants.MODEL_TRAINING_MODELS_REGISTERY
            )

            self.models_hyperparameters = (
                pipeline_constants
                .MODEL_TRAINING_MODELS_HYPERPARAMETERS
            )

            self.primary_metric: str = (
                pipeline_constants.MODEL_TRAINING_PRIMARY_METRIC
            )

            self.decision_threshold: float = (
                pipeline_constants.MODEL_TRAINING_DECISION_THRESHOLD
            )

            self.n_iter: int = pipeline_constants.MODEL_TRAINING_N_ITER

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Model Evaluation Config
# ============================================================


class ModelEvaluationConfig:
    """Configuration for model evaluation stage."""

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig,
    ) -> None:
        try:
            self.evaluation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.MODEL_EVALUATION_DIR_NAME,
            )

            self.evaluation_report_file_path: str = os.path.join(
                self.evaluation_dir,
                pipeline_constants.MODEL_EVALUATION_REPORT_FILE_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.evaluation_dir,
                pipeline_constants.MODEL_EVALUATION_METADATA_FILE_NAME,
            )

            self.model_registry_metadata_file_path: str = (
                pipeline_constants.MODEL_REGISTRY_METADATA_PATH
            )

            self.decision_threshold: float = (
                pipeline_constants.MODEL_EVALUATION_DECISION_THRESHOLD
            )

            self.recall_tolerance: float = (
                pipeline_constants.MODEL_EVALUATION_RECALL_TOLERANCE
            )

            self.min_recall_improvement: float = (
                pipeline_constants.MODEL_EVALUATION_MIN_IMPROVEMENT
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Model Registry Config
# ============================================================


class ModelRegistryConfig:
    """Configuration for model registry."""

    def __init__(self) -> None:
        try:
            self.registry_dir: str = (
                pipeline_constants.MODEL_REGISTRY_DIR
            )
            
            self.registry_metadata_path: str = (
                pipeline_constants.MODEL_REGISTRY_METADATA_PATH
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Model Monitoring Config
# ============================================================


class MonitoringConfig:
    """Configuration for monitoring pipeline."""

    def __init__(self) -> None:
        try:
            self.run_id = datetime.now(timezone.utc).strftime(
                "%Y%m%d_%H%M%S"
            )

            self.monitoring_dir: str = os.path.join(
                pipeline_constants.ARTIFACT_DIR,
                pipeline_constants.MONITORING_DIR_NAME,
                self.run_id
            )
            
            self.baseline_download_dir: str = os.path.join(self.monitoring_dir, "baseline")
            self.telemetry_data_dir: str = os.path.join(self.monitoring_dir, "telemetry_data")
            self.drift_report_dir: str = os.path.join(self.monitoring_dir, "drift_report")

            self.telemetry_csv_path: str = os.path.join(self.telemetry_data_dir, "telemetry_snapshot.csv")
            self.baseline_path: str = os.path.join(self.baseline_download_dir, "baseline.json")
            self.drift_report_path: str = os.path.join(self.drift_report_dir, "drift_report.json")
            self.trigger_metadata_path: str = os.path.join(self.monitoring_dir, "retraining_trigger.json")

            self.data_window_days: int = pipeline_constants.MONITORING_DATA_WINDOW_DAYS
            self.min_row_required: int = pipeline_constants.MONITORING_MIN_ROWS_REQUIRED
            self.num_drift_threshold: float = pipeline_constants.MONITORING_NUMERICAL_DRIFT_THRESHOLD
            self.cat_drift_threshold: float = pipeline_constants.MONITORING_CATEGORICAL_DRIFT_THRESHOLD
            self.cooldown_days: int = pipeline_constants.MONITORING_RETRAINING_COOLDOWN_DAYS
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
        



class MonitoringConfig:
    """
    Configuration for the Monitoring and Continual Learning Pipeline.
    Constructs deterministic paths for drift analysis artifacts.
    """
    def __init__(self):
        try:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Base monitoring directory
            self.monitoring_dir: str = os.path.join(
                pipeline_constants.ARTIFACT_DIR,
                pipeline_constants.MONITORING_DIR_NAME,
                self.run_id
            )
            
            # Sub-directories for organized artifact storage
            self.baseline_download_dir: str = os.path.join(self.monitoring_dir, "baseline")
            self.telemetry_data_dir: str = os.path.join(self.monitoring_dir, "telemetry_data")
            self.drift_report_dir: str = os.path.join(self.monitoring_dir, "drift_report")
            
            # File paths
            self.telemetry_csv_path: str = os.path.join(self.telemetry_data_dir, "telemetry_snapshot.csv")
            self.baseline_path: str = os.path.join(self.baseline_download_dir, "baseline.json")
            self.drift_report_path: str = os.path.join(self.drift_report_dir, "drift_report.json")
            self.trigger_metadata_path: str = os.path.join(self.monitoring_dir, "retraining_trigger.json")
            
            # System Rules mapped from constants
            self.data_window_days: int = pipeline_constants.MONITORING_DATA_WINDOW_DAYS
            self.min_rows_required: int = pipeline_constants.MONITORING_MIN_ROWS_REQUIRED
            self.num_drift_threshold: float = pipeline_constants.MONITORING_NUMERICAL_DRIFT_THRESHOLD
            self.cat_drift_threshold: float = pipeline_constants.MONITORING_CATEGORICAL_DRIFT_THRESHOLD
            self.cooldown_days: int = pipeline_constants.MONITORING_RETRAINING_COOLDOWN_DAYS

        except Exception as e:
            raise CustomerChurnException(e, sys)