import sys
from typing import Any

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.etl import CustomerChurnETL
from src.components.model_evaluation import ModelEvaluation
from src.components.model_registry import ModelRegistry
from src.components.model_training import ModelTrainer

from src.constants.pipeline_constants import (
    ARTIFACT_DIR,
    MODEL_REGISTRY_DIR,
    S3_ARTIFACT_DIR_NAME,
    S3_MODEL_REGISTRY_DIR_NAME,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ETLArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)

from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ETLconfig,
    ModelEvaluationConfig,
    ModelRegistryConfig,
    ModelTrainingConfig,
    TrainingPipelineConfig,
)

from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import sync_to_s3


class TrainingPipeline:
    """
    Production-grade ML Training Pipeline Orchestrator.

    Responsibilities
    ----------------
    - Controls full ML lifecycle execution
    - Manages artifact propagation
    - Handles failures deterministically
    - Provides structured observability
    """

    # ==========================================================
    # Initialization
    # ==========================================================

    def __init__(self) -> None:
        try:
            self.pipeline_config = TrainingPipelineConfig()

            logging.info("TRAINING PIPELINE INITIALIZED\n")
            
        except Exception as exc:
            raise CustomerChurnException(exc, sys)

    # ==========================================================
    # Generic Stage Executor
    # ==========================================================

    def _execute_stage(self, stage_name: str, fn) -> Any:
        """
        Standardized execution wrapper for all pipeline stages.
        """
        try:
            logging.info(">>>>>> %s STARTED <<<<<<", stage_name)

            artifact = fn()

            logging.info(">>>>>> %s COMPLETED <<<<<<\n", stage_name)

            return artifact

        except Exception as exc:
            logging.exception("%s FAILED", stage_name)
            raise CustomerChurnException(exc, sys)

    # ==========================================================
    # Pipeline Stages
    # ==========================================================

    def start_etl(self) -> ETLArtifact:
        def run():
            config = ETLconfig(self.pipeline_config)
            etl = CustomerChurnETL(config)
            return etl.initiate_etl()

        return self._execute_stage("Stage 1: ETL", run)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        def run():
            config = DataIngestionConfig(self.pipeline_config)
            ingestion = DataIngestion(config)
            return ingestion.initiate_data_ingestion()

        return self._execute_stage("Stage 2: Data Ingestion", run)

    def start_data_validation(
        self,
        ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:

        def run():
            config = DataValidationConfig(self.pipeline_config)
            validation = DataValidation(config, ingestion_artifact)
            return validation.initiate_data_validation()

        return self._execute_stage("Stage 3: Data Validation", run)

    def start_data_transformation(
        self,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:

        def run():
            config = DataTransformationConfig(self.pipeline_config)

            transformation = DataTransformation(
                config,
                ingestion_artifact,
                validation_artifact,
            )

            return transformation.initiate_data_transformation()

        return self._execute_stage("Stage 4: Data Transformation", run)

    def start_model_training(
        self,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:

        def run():
            config = ModelTrainingConfig(self.pipeline_config)

            trainer = ModelTrainer(
                config,
                ingestion_artifact,
                transformation_artifact,
            )

            return trainer.initiate_model_training()

        return self._execute_stage("Stage 5: Model Training", run)

    def start_model_evaluation(
        self,
        ingestion_artifact: DataIngestionArtifact,
        trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:

        def run():
            config = ModelEvaluationConfig(self.pipeline_config)

            evaluation = ModelEvaluation(
                config,
                ingestion_artifact,
                trainer_artifact,
            )

            return evaluation.initiate_model_evaluation()

        return self._execute_stage("Stage 6: Model Evaluation", run)

    def start_model_registry(
        self,
        evaluation_artifact: ModelEvaluationArtifact,
    ):
        def run():
            config = ModelRegistryConfig()

            registry = ModelRegistry(config, evaluation_artifact)

            return registry.initiate_model_registry()

        return self._execute_stage("Stage 7: Model Registry", run)

    # ==========================================================
    # Pipeline Execution
    # ==========================================================

    def run_pipeline(self):

        try:
            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE EXECUTION STARTED")
            logging.info("=" * 60)

            etl_artifact = self.start_etl()

            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(
                ingestion_artifact
            )

            transformation_artifact = self.start_data_transformation(
                ingestion_artifact,
                validation_artifact,
            )

            trainer_artifact = self.start_model_training(
                ingestion_artifact,
                transformation_artifact,
            )

            evaluation_artifact = self.start_model_evaluation(
                ingestion_artifact,
                trainer_artifact,
            )

            registry_artifact = self.start_model_registry(
                evaluation_artifact
            )

            # --------------------------------------------------
            # Artifact Synchronization (Cloud Backup)
            # --------------------------------------------------

            logging.info("Syncing artifacts to S3...")

            sync_to_s3(ARTIFACT_DIR, S3_ARTIFACT_DIR_NAME)
            sync_to_s3(MODEL_REGISTRY_DIR, S3_MODEL_REGISTRY_DIR_NAME)

            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE EXECUTION COMPLETED")
            logging.info("=" * 60)

            return registry_artifact

        except Exception as exc:
            logging.exception("Training pipeline execution failed")
            raise CustomerChurnException(exc, sys)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    try:
        pipeline = TrainingPipeline()

        pipeline.run_pipeline()

    except Exception:
        logging.exception("Unhandled exception in main execution")