# from dataclasses import dataclass


# @dataclass(frozen=True)
# class ETLArtifact:
#     raw_data_dir_path: str
#     metadata_file_path: str

#     def __str__(self) -> str:
#         return (
#             "\nETLArtifact(\n"
#             f"  raw_data_file_path = {self.raw_data_dir_path}\n"
#             f"  metadata_file_path = {self.metadata_file_path}\n"
#             ")"
#         )


# @dataclass(frozen=True)
# class DataIngestionArtifact:
#     """
#     Artifact generated after the Data Ingestion stage.

#     Attributes:
#         train_file_path (str): Path to the training dataset
#         test_file_path (str): Path to the test dataset
#         val_file_path (str): Path to the validation dataset
#         schema_file_path (str): Path to the schema file
#         metadata_file_path (str): Path to metadata generated during ingestion
#     """

#     train_file_path: str
#     test_file_path: str
#     val_file_path: str
#     schema_file_path: str
#     metadata_file_path: str

#     def __str__(self) -> str:
#         return (
#             "\nDataIngestionArtifact(\n"
#             f"  train_file_path    = {self.train_file_path}\n"
#             f"  test_file_path     = {self.test_file_path}\n"
#             f"  val_file_path      = {self.val_file_path}\n"
#             f"  schema_file_path   = {self.schema_file_path}\n"
#             f"  metadata_file_path = {self.metadata_file_path}\n"
#             ")"
#         )


# @dataclass(frozen=True)
# class DataValidationArtifact:
#     """
#     Artifact generated after the Data Validation stage.

#     Attributes:
#         validation_status (bool): Whether validation passed or failed
#         validation_report (str): Path to validation report
#     """

#     validation_status: bool
#     validation_report: str

#     def __str__(self) -> str:
#         return (
#             "\nDataValidationArtifact(\n"
#             f"  validation_status = {self.validation_status}\n"
#             f"  validation_report = {self.validation_report}\n"
#             ")"
#         )


# @dataclass(frozen=True)
# class DataTransformationArtifact:
#     """
#     Artifact generated after the Data Transformation stage.

#     Responsibilities
#     ----------------
#     Encapsulates all outputs produced by the transformation pipeline,
#     including preprocessing artifacts, transformation metadata,
#     and monitoring baseline reference.

#     Attributes
#     ----------
#     tree_preprocessor_file_path (str)
#         Path to fitted tree-based preprocessor.

#     linear_preprocessor_file_path (str)
#         Path to fitted linear-model preprocessor.

#     metadata_file_path (str)
#         Path to transformation metadata artifact.
#     """

#     tree_preprocessor_file_path: str
#     linear_preprocessor_file_path: str
#     metadata_file_path: str

#     def __str__(self) -> str:
#         """
#         Human-readable artifact representation for structured logging.
#         """
#         return (
#             "\nDataTransformationArtifact(\n"
#             f"  tree_preprocessor_file_path   = "
#             f"{self.tree_preprocessor_file_path}\n"
#             f"  linear_preprocessor_file_path = "
#             f"{self.linear_preprocessor_file_path}\n"
#             f"  metadata_file_path            = "
#             f"{self.metadata_file_path}\n"
#             ")"
#         )


# @dataclass(frozen=True)
# class ModelTrainerArtifact:
#     trained_models_dir: str
#     metadata_file_path: str

#     def __str__(self) -> str:
#         return (
#             "\nModelTrainingArtifact(\n"
#             f"  trained_models_dir = {self.trained_models_dir}\n"
#             f"  metadata_file_path = {self.metadata_file_path}\n"
#             ")"
#         )


# @dataclass(frozen=True)
# class ModelEvaluationArtifact:
#     """
#     Artifact produced by Model Evaluation Pipeline.

#     Contains:
#         - Selected best candidate model
#         - Evaluation outputs
#         - Approval decision for model registration
#     """

#     best_model_name: str
#     best_model_path: str
#     evaluation_report_path: str
#     metadata_path: str
#     approval_status: bool

#     def __str__(self) -> str:
#         return (
#             "\nModelEvaluationArtifact(\n"
#             f"  best_model_name        = {self.best_model_name}\n"
#             f"  best_model_path        = {self.best_model_path}\n"
#             f"  evaluation_report_path = {self.evaluation_report_path}\n"
#             f"  metadata_path          = {self.metadata_path}\n"
#             f"  approval_status        = {self.approval_status}\n"
#             ")"
#         )


# # ==========================================================
# # MONITORING & CONTINUAL LEARNING ARTIFACTS
# # ==========================================================

# @dataclass
# class LabelIngestionArtifact:
#     """Output of the label_ingestor.py step."""
#     collection_name: str
#     new_labels_ingested_count: int
#     total_unprocessed_labels: int


# @dataclass
# class FastLaneDriftArtifact:
#     """Output of the fast_lane_drift.py step."""
#     window_start_time: str
#     window_end_time: str
#     features_analyzed: int
#     drift_detected: bool
#     drift_report_path: str


# @dataclass
# class SlowLanePerformanceArtifact:
#     """Output of the slow_lane_performance.py step."""
#     joined_data_path: str
#     performance_report_path: str
#     f1_score: float
#     precision: float
#     recall: float
#     is_retraining_required: bool
#     labels_evaluated_count: int


# @dataclass
# class DataCurationArtifact:
#     """Output of the data_curator.py step."""
#     curated_dataset_local_path: str
#     s3_uploaded_version_uri: str
#     total_rows_in_curated_set: int





"""
Artifact definitions used across the ML pipeline stages.

Each artifact represents the output produced by a specific pipeline step.
Artifacts are implemented as immutable dataclasses where appropriate to
ensure pipeline reproducibility and prevent accidental mutation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ETLArtifact:
    """Artifact generated after the ETL stage."""

    raw_data_dir_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nETLArtifact(\n"
            f"  raw_data_dir_path  = {self.raw_data_dir_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Artifact generated after the Data Ingestion stage.
    """

    train_file_path: str
    test_file_path: str
    val_file_path: str
    schema_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataIngestionArtifact(\n"
            f"  train_file_path    = {self.train_file_path}\n"
            f"  test_file_path     = {self.test_file_path}\n"
            f"  val_file_path      = {self.val_file_path}\n"
            f"  schema_file_path   = {self.schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataValidationArtifact:
    """
    Artifact generated after the Data Validation stage.
    """

    validation_status: bool
    validation_report: str

    def __str__(self) -> str:
        return (
            "\nDataValidationArtifact(\n"
            f"  validation_status = {self.validation_status}\n"
            f"  validation_report = {self.validation_report}\n"
            ")"
        )


@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Artifact generated after the Data Transformation stage.
    """

    tree_preprocessor_file_path: str
    linear_preprocessor_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataTransformationArtifact(\n"
            f"  tree_preprocessor_file_path   = {self.tree_preprocessor_file_path}\n"
            f"  linear_preprocessor_file_path = {self.linear_preprocessor_file_path}\n"
            f"  metadata_file_path            = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelTrainerArtifact:
    """Artifact generated after the Model Training stage."""

    trained_models_dir: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nModelTrainerArtifact(\n"
            f"  trained_models_dir = {self.trained_models_dir}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    """
    Artifact produced by the Model Evaluation pipeline.
    """

    best_model_name: str
    best_model_path: str
    evaluation_report_path: str
    metadata_path: str
    approval_status: bool

    def __str__(self) -> str:
        return (
            "\nModelEvaluationArtifact(\n"
            f"  best_model_name        = {self.best_model_name}\n"
            f"  best_model_path        = {self.best_model_path}\n"
            f"  evaluation_report_path = {self.evaluation_report_path}\n"
            f"  metadata_path          = {self.metadata_path}\n"
            f"  approval_status        = {self.approval_status}\n"
            ")"
        )


# ==========================================================
# MONITORING & CONTINUAL LEARNING ARTIFACTS
# ==========================================================


@dataclass
class LabelIngestionArtifact:
    """Output of the label_ingestor step."""

    collection_name: str
    new_labels_ingested_count: int
    total_unprocessed_labels: int

    def __str__(self) -> str:
        return (
            "\nLabelIngestionArtifact(\n"
            f"  collection_name            = {self.collection_name}\n"
            f"  new_labels_ingested_count  = {self.new_labels_ingested_count}\n"
            f"  total_unprocessed_labels   = {self.total_unprocessed_labels}\n"
            ")"
        )


@dataclass
class FastLaneDriftArtifact:
    """Output of the fast_lane_drift step."""

    window_start_time: str
    window_end_time: str
    features_analyzed: int
    drift_detected: bool
    drift_report_path: str

    def __str__(self) -> str:
        return (
            "\nFastLaneDriftArtifact(\n"
            f"  window_start_time  = {self.window_start_time}\n"
            f"  window_end_time    = {self.window_end_time}\n"
            f"  features_analyzed  = {self.features_analyzed}\n"
            f"  drift_detected     = {self.drift_detected}\n"
            f"  drift_report_path  = {self.drift_report_path}\n"
            ")"
        )


@dataclass
class SlowLanePerformanceArtifact:
    """Output of the slow_lane_performance step."""

    joined_data_path: str
    performance_report_path: str
    f1_score: float
    precision: float
    recall: float
    is_retraining_required: bool
    labels_evaluated_count: int

    def __str__(self) -> str:
        return (
            "\nSlowLanePerformanceArtifact(\n"
            f"  joined_data_path          = {self.joined_data_path}\n"
            f"  performance_report_path   = {self.performance_report_path}\n"
            f"  f1_score                  = {self.f1_score}\n"
            f"  precision                 = {self.precision}\n"
            f"  recall                    = {self.recall}\n"
            f"  is_retraining_required    = {self.is_retraining_required}\n"
            f"  labels_evaluated_count    = {self.labels_evaluated_count}\n"
            ")"
        )


@dataclass
class DataCurationArtifact:
    """Output of the data_curator step."""

    curated_dataset_local_path: str
    s3_uploaded_version_uri: str
    total_rows_in_curated_set: int

    def __str__(self) -> str:
        return (
            "\nDataCurationArtifact(\n"
            f"  curated_dataset_local_path = {self.curated_dataset_local_path}\n"
            f"  s3_uploaded_version_uri    = {self.s3_uploaded_version_uri}\n"
            f"  total_rows_in_curated_set  = {self.total_rows_in_curated_set}\n"
            ")"
        )