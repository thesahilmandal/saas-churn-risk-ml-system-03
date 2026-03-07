"""
Data Transformation Pipeline.

Responsibilities:
- Build model-aware preprocessing pipelines
- Fit preprocessors on training data only (no leakage)
- Persist fitted preprocessors
- Generate transformation metadata
- Generate monitoring baseline artifact for drift detection

Design Guarantees:
- Validation-gated execution
- Deterministic preprocessing
- No data leakage
- Reproducible experiment metadata
- Monitoring baseline aligned with training distribution
"""

import hashlib
import json
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

from src.constants.pipeline_constants import (
    MONITORING_BASELINE_PATH,
    TARGET_COLUMN,
)
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

        except Exception as e:
            logging.exception("[DATA TRANSFORMATION INIT] Failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Utility Methods
    # ============================================================

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return pd.read_csv(file_path)

    @staticmethod
    def _compute_hash(obj: Any) -> str:

        payload = json.dumps(
            obj,
            sort_keys=True,
            default=str,
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _get_feature_groups(
        X: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:

        numeric = X.select_dtypes(
            include=["int", "float"]
        ).columns.tolist()

        categorical = [c for c in X.columns if c not in numeric]

        if not numeric:
            raise ValueError("No numeric features detected.")

        if not categorical:
            raise ValueError("No categorical features detected.")

        return numeric, categorical

    # ============================================================
    # Preprocessor Builders
    # ============================================================

    def _build_linear_preprocessor(
        self,
        num_features: List[str],
        cat_features: List[str],
    ) -> ColumnTransformer:

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
                ("num", numeric_pipeline, num_features),
                ("cat", categorical_pipeline, cat_features),
            ],
            remainder="drop",
        )

    def _build_tree_preprocessor(
        self,
        num_features: List[str],
        cat_features: List[str],
    ) -> ColumnTransformer:

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
                ("num", numeric_pipeline, num_features),
                ("cat", categorical_pipeline, cat_features),
            ],
            remainder="drop",
        )

    # ============================================================
    # Encoder Metadata Extraction
    # ============================================================

    def _extract_encoder_metadata(
        self,
        preprocessor: ColumnTransformer,
        cat_features: List[str],
    ) -> Dict[str, Any]:

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

            if name == "num":
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
        num_features: List[str],
        cat_features: List[str],
        linear_preprocessor: ColumnTransformer,
        tree_preprocessor: ColumnTransformer,
    ) -> Dict[str, Any]:

        linear_encoder_meta = self._extract_encoder_metadata(
            linear_preprocessor,
            cat_features,
        )

        tree_encoder_meta = self._extract_encoder_metadata(
            tree_preprocessor,
            cat_features,
        )

        metadata = {
            "pipeline_version": self.PIPELINE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "dataset": {
                "train_rows": int(X_train.shape[0]),
                "input_features": int(X_train.shape[1]),
                "numeric_features": num_features,
                "categorical_features": cat_features,
            },
            "linear_preprocessor": linear_encoder_meta,
            "tree_preprocessor": tree_encoder_meta,
        }

        metadata["metadata_hash"] = self._compute_hash(metadata)

        return metadata

    # ============================================================
    # Monitoring Baseline Generation
    # ============================================================

    def _compute_numerical_baseline(
        self,
        X_train: pd.DataFrame,
        num_features: List[str],
    ) -> Dict[str, Any]:

        baseline: Dict[str, Any] = {}

        for col in num_features:

            series = X_train[col]

            baseline[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "quantiles": {
                    "p10": float(series.quantile(0.10)),
                    "p25": float(series.quantile(0.25)),
                    "p50": float(series.quantile(0.50)),
                    "p75": float(series.quantile(0.75)),
                    "p90": float(series.quantile(0.90)),
                },
                "missing_ratio": float(series.isna().mean()),
            }

        return baseline

    def _compute_categorical_baseline(
        self,
        X_train: pd.DataFrame,
        cat_features: List[str],
    ) -> Dict[str, Any]:

        baseline: Dict[str, Any] = {}

        for col in cat_features:

            series = X_train[col]

            value_counts = (
                series.value_counts(
                    normalize=True,
                    dropna=False,
                ).to_dict()
            )

            baseline[col] = {
                "cardinality": int(series.nunique(dropna=True)),
                "distribution": {
                    str(k): float(v)
                    for k, v in value_counts.items()
                },
                "missing_ratio": float(series.isna().mean()),
            }

        return baseline

    def _generate_monitoring_baseline(
        self,
        X_train: pd.DataFrame,
        num_features: List[str],
        cat_features: List[str],
        preprocessor: ColumnTransformer,
    ) -> Dict[str, Any]:

        numerical_baseline = self._compute_numerical_baseline(
            X_train,
            num_features,
        )

        categorical_baseline = self._compute_categorical_baseline(
            X_train,
            cat_features,
        )

        encoder_metadata = self._extract_encoder_metadata(
            preprocessor,
            cat_features,
        )

        baseline = {
            "pipeline_version": self.PIPELINE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "numerical_features": numerical_baseline,
            "categorical_features": categorical_baseline,
            "encoded_feature_metadata": encoder_metadata,
        }

        baseline["baseline_hash"] = self._compute_hash(baseline)

        return baseline

    # ============================================================
    # Pipeline Entry Point
    # ============================================================

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:

        try:

            logging.info("[DATA TRANSFORMATION PIPELINE] Started")

            train_df = self._read_csv(
                self.ingestion_artifact.train_file_path
            )

            if TARGET_COLUMN not in train_df.columns:

                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' "
                    "not found in training data"
                )

            X_train = train_df.drop(columns=[TARGET_COLUMN])

            num_features, cat_features = self._get_feature_groups(
                X_train
            )

            linear_preprocessor = self._build_linear_preprocessor(
                num_features,
                cat_features,
            )

            linear_preprocessor.fit(X_train)

            tree_preprocessor = self._build_tree_preprocessor(
                num_features,
                cat_features,
            )

            tree_preprocessor.fit(X_train)

            save_object(
                self.config.lr_preprocessor_file_path,
                linear_preprocessor,
            )

            save_object(
                self.config.tree_preprocessor_file_path,
                tree_preprocessor,
            )

            metadata = self._generate_metadata(
                X_train,
                num_features,
                cat_features,
                linear_preprocessor,
                tree_preprocessor,
            )

            write_json_file(
                self.config.metadata_file_path,
                metadata,
            )

            monitoring_baseline = self._generate_monitoring_baseline(
                X_train,
                num_features,
                cat_features,
                linear_preprocessor,
            )

            monitoring_baseline_path = (
                self.config.monitoring_baseline_file_path
            )

            write_json_file(
                monitoring_baseline_path,
                monitoring_baseline,
            )

            write_json_file(
                MONITORING_BASELINE_PATH,
                monitoring_baseline,
            )

            artifact = DataTransformationArtifact(
                tree_preprocessor_file_path=self.config.tree_preprocessor_file_path,
                linear_preprocessor_file_path=self.config.lr_preprocessor_file_path,
                metadata_file_path=self.config.metadata_file_path,
                monitoring_baseline_file_path=self.config.monitoring_baseline_file_path,
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Completed successfully"
            )

            logging.info(artifact)

            return artifact

        except Exception as e:

            logging.exception(
                "[DATA TRANSFORMATION PIPELINE] Failed"
            )

            raise CustomerChurnException(e, sys)