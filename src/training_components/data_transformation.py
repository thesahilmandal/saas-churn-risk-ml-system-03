"""
Data Transformation Pipeline.

Responsibilities:
- Build model-aware preprocessing pipelines
- Fit preprocessors on training data only (no leakage)
- Persist fitted preprocessors
- Generate transformation metadata

Design Guarantees:
- Validation-gated execution
- Deterministic preprocessing
- No data leakage
- Reproducible experiment metadata
"""

import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants.pipeline_constants import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file, save_object, write_json_file


class DataTransformation:
    """
    Handles preprocessing pipeline creation and persistence.

    Builds preprocessing pipelines suitable for both linear models
    and tree-based models, fits them on training data, and persists
    the artifacts along with transformation metadata.
    """

    PIPELINE_VERSION = "1.0.0"

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> None:
        """Initialize Data Transformation pipeline."""

        try:
            logging.info("[DATA TRANSFORMATION INIT] Initializing")

            if not validation_artifact.validation_status:
                raise ValueError(
                    "Data validation failed. Transformation aborted."
                )

            self.config = transformation_config
            self.ingestion_artifact = ingestion_artifact
            self.validation_artifact = validation_artifact

            os.makedirs(
                self.config.data_transformation_dir,
                exist_ok=True,
            )

            logging.info(
                "[DATA TRANSFORMATION INIT] Initialized | "
                f"pipeline_version={self.PIPELINE_VERSION}"
            )

        except Exception as exc:
            logging.exception("[DATA TRANSFORMATION INIT] Failed")
            raise CustomerChurnException(exc, sys)

    # ============================================================
    # Utility Methods
    # ============================================================

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        """Read CSV file safely."""

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return pd.read_csv(file_path)

    @staticmethod
    def _get_feature_groups(
        X: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical features."""

        numeric_features = X.select_dtypes(
            include=["int", "float"]
        ).columns.tolist()

        categorical_features = [
            col for col in X.columns if col not in numeric_features
        ]

        if not numeric_features:
            raise ValueError("No numeric features detected.")

        if not categorical_features:
            raise ValueError("No categorical features detected.")

        return numeric_features, categorical_features

    # ============================================================
    # Preprocessor Builders
    # ============================================================

    def _build_linear_preprocessor(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> ColumnTransformer:
        """Build preprocessing pipeline for linear models."""

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

    def _build_tree_preprocessor(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> ColumnTransformer:
        """Build preprocessing pipeline for tree-based models."""

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

    # ============================================================
    # Encoder Metadata Extraction
    # ============================================================

    def _extract_encoder_metadata(
        self,
        preprocessor: ColumnTransformer,
        categorical_features: List[str],
    ) -> Dict[str, Any]:
        """Extract metadata from OneHotEncoder."""

        metadata: Dict[str, Any] = {
            "categorical_cardinality": {},
            "output_feature_names": [],
        }

        for name, transformer, features in preprocessor.transformers_:

            if name == "cat":

                encoder: OneHotEncoder = transformer.named_steps["encoder"]

                for feature, categories in zip(
                    features,
                    encoder.categories_,
                ):
                    metadata["categorical_cardinality"][feature] = len(
                        categories
                    )

                metadata["output_feature_names"].extend(
                    encoder.get_feature_names_out(features).tolist()
                )

            elif name == "num":
                metadata["output_feature_names"].extend(features)

        metadata["output_feature_count"] = len(
            metadata["output_feature_names"]
        )

        return metadata

    # ============================================================
    # Metadata Generation
    # ============================================================

    def _generate_metadata(
        self,
        X_train: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: List[str],
        linear_preprocessor: ColumnTransformer,
        tree_preprocessor: ColumnTransformer,
    ) -> Dict[str, Any]:
        """Generate transformation metadata."""

        ingestion_metadata = read_json_file(
            self.ingestion_artifact.metadata_file_path
        )

        train_checksum = ingestion_metadata["split"]["checksums"]["train"]

        linear_meta = self._extract_encoder_metadata(
            linear_preprocessor,
            categorical_features,
        )

        tree_meta = self._extract_encoder_metadata(
            tree_preprocessor,
            categorical_features,
        )

        metadata = {
            "pipeline_version": self.PIPELINE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "dataset": {
                "train_data_checksum": train_checksum,
                "train_rows": int(X_train.shape[0]),
                "input_features": int(X_train.shape[1]),
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
            },
            "linear_preprocessor": linear_meta,
            "tree_preprocessor": tree_meta,
        }

        return metadata

    # ============================================================
    # Pipeline Entry Point
    # ============================================================

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        """Execute the data transformation pipeline."""

        try:
            logging.info("[DATA TRANSFORMATION PIPELINE] Started")

            train_df = self._read_csv(
                self.ingestion_artifact.train_file_path
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Training data loaded | "
                f"rows={train_df.shape[0]} | cols={train_df.shape[1]}"
            )

            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' "
                    "not found in training data"
                )

            X_train = train_df.drop(columns=[TARGET_COLUMN])

            numeric_features, categorical_features = self._get_feature_groups(
                X_train
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Feature groups identified | "
                f"numeric={len(numeric_features)} | "
                f"categorical={len(categorical_features)}"
            )

            linear_preprocessor = self._build_linear_preprocessor(
                numeric_features,
                categorical_features,
            )

            tree_preprocessor = self._build_tree_preprocessor(
                numeric_features,
                categorical_features,
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Preprocessors created"
            )

            linear_preprocessor.fit(X_train)
            tree_preprocessor.fit(X_train)

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Preprocessors fitted"
            )

            save_object(
                self.config.lr_preprocessor_file_path,
                linear_preprocessor,
            )

            save_object(
                self.config.tree_preprocessor_file_path,
                tree_preprocessor,
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Preprocessors saved | "
                f"linear={self.config.lr_preprocessor_file_path} | "
                f"tree={self.config.tree_preprocessor_file_path}"
            )

            metadata = self._generate_metadata(
                X_train,
                numeric_features,
                categorical_features,
                linear_preprocessor,
                tree_preprocessor,
            )

            write_json_file(
                self.config.metadata_file_path,
                metadata,
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Metadata generated | "
                f"path={self.config.metadata_file_path}"
            )

            artifact = DataTransformationArtifact(
                tree_preprocessor_file_path=self.config.tree_preprocessor_file_path,
                linear_preprocessor_file_path=self.config.lr_preprocessor_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Completed successfully"
            )

            logging.info(artifact)

            return artifact

        except Exception as exc:

            logging.exception(
                "[DATA TRANSFORMATION PIPELINE] Failed"
            )

            raise CustomerChurnException(exc, sys)