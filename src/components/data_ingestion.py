"""
Data Ingestion Pipeline for Customer Churn Project.

Responsibilities
----------------
1. Load cleaned customer data from MongoDB
2. Perform reproducible stratified train/validation/test split
3. Persist datasets, schema, and metadata for experiment lineage
"""

import hashlib
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.constants.pipeline_constants import TARGET_COLUMN
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file


load_dotenv()


class DataIngestion:
    """
    Production-grade Data Ingestion component.

    Responsibilities
    ----------------
    - Load dataset from MongoDB
    - Perform deterministic cleaning
    - Execute stratified train/validation/test split
    - Generate schema and metadata artifacts

    Design Principles
    -----------------
    - Reproducible experiments
    - Deterministic dataset lineage
    - Observability through metadata and logging
    """

    PIPELINE_VERSION = "1.0.0"

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initialize the data ingestion pipeline.

        Parameters
        ----------
        config : DataIngestionConfig
            Configuration object for the data ingestion stage.
        """
        try:
            self.config = config
            self.target_column = TARGET_COLUMN

            self._validate_configuration()

            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)

            logging.info(
                "[DATA INGESTION INIT] Initialized | "
                "dataset=customer_churn | "
                "pipeline_version=%s",
                self.PIPELINE_VERSION,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    # =========================================================
    # CONFIGURATION VALIDATION
    # =========================================================

    def _validate_configuration(self) -> None:
        """
        Ensure required configuration parameters exist.
        """
        try:
            required = {
                "database_url": self.config.database_url,
                "database_name": self.config.database_name,
                "collection_name": self.config.collection_name,
            }

            missing = [k for k, v in required.items() if not v]

            if missing:
                raise EnvironmentError(
                    f"Missing required configuration values: {missing}"
                )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    # =========================================================
    # UTILITY METHODS
    # =========================================================

    @staticmethod
    def _compute_checksum(df: pd.DataFrame) -> str:
        """
        Compute SHA-256 checksum for a DataFrame.

        Ensures dataset identity and reproducibility.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        str
            SHA256 checksum
        """
        hash_bytes = pd.util.hash_pandas_object(
            df,
            index=True,
        ).values.tobytes()

        return hashlib.sha256(hash_bytes).hexdigest()

    # =========================================================
    # DATA LOADING
    # =========================================================

    def _load_from_mongodb(self) -> pd.DataFrame:
        """
        Load records from MongoDB.

        Returns
        -------
        pd.DataFrame
            Dataset retrieved from MongoDB.
        """
        try:
            logging.info(
                "[DATA INGESTION] Connecting to MongoDB | "
                "db=%s collection=%s",
                self.config.database_name,
                self.config.collection_name,
            )

            with pymongo.MongoClient(self.config.database_url) as client:
                collection = client[self.config.database_name][
                    self.config.collection_name
                ]

                records = list(collection.find({}, {"_id": 0}))

            if not records:
                raise ValueError(
                    "MongoDB collection returned zero records."
                )

            df = pd.DataFrame(records)

            logging.info(
                "[DATA INGESTION] MongoDB read successful | rows=%d cols=%d",
                df.shape[0],
                df.shape[1],
            )

            return df

        except Exception as e:
            logging.exception("[DATA INGESTION] MongoDB read failed.")
            raise CustomerChurnException(e, sys) from e

    # =========================================================
    # DATA CLEANING
    # =========================================================

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply deterministic data cleaning logic.

        Returns
        -------
        pd.DataFrame
            Cleaned dataset.
        """
        try:
            initial_rows = len(df)

            drop_columns = [
                "data_source",
                "ingested_at_utc",
                "customerID",
                "_id",
            ]

            df = df.drop(
                columns=[c for c in drop_columns if c in df.columns]
            )

            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce",
            )

            df["SeniorCitizen"] = df["SeniorCitizen"].map(
                {1: "Yes", 0: "No"}
            )

            df[self.target_column] = df[self.target_column].map(
                {"Yes": 1, "No": 0}
            )

            df = df.replace({"na": np.nan})
            df = df.drop_duplicates()

            if df[self.target_column].isna().any():
                raise ValueError(
                    "Target column contains null values after cleaning."
                )

            logging.info(
                "[DATA INGESTION] Cleaning completed | "
                "rows_before=%d rows_after=%d",
                initial_rows,
                len(df),
            )

            return df

        except Exception as e:
            logging.exception("[DATA INGESTION] Data cleaning failed.")
            raise CustomerChurnException(e, sys) from e

    # =========================================================
    # DATA SPLITTING
    # =========================================================

    def _split_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform reproducible stratified train/validation/test split.
        """
        try:
            if df[self.target_column].nunique() < 2:
                raise ValueError(
                    "Stratified split requires at least two target classes."
                )

            train_df, temp_df = train_test_split(
                df,
                test_size=self.config.train_temp_split_ratio,
                random_state=self.config.random_state,
                stratify=df[self.target_column],
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=self.config.test_val_split_ratio,
                random_state=self.config.random_state,
                stratify=temp_df[self.target_column],
            )

            logging.info(
                "[DATA INGESTION] Split completed | "
                "train=%d val=%d test=%d",
                len(train_df),
                len(val_df),
                len(test_df),
            )

            return train_df, val_df, test_df

        except Exception as e:
            logging.exception("[DATA INGESTION] Data split failed.")
            raise CustomerChurnException(e, sys) from e

    # =========================================================
    # SCHEMA GENERATION
    # =========================================================

    def _generate_schema(
        self,
        train_df: pd.DataFrame,
    ) -> Dict[str, Dict]:
        """
        Generate schema using training dataset.
        """
        schema: Dict[str, Dict] = {}

        for column in train_df.columns:
            col = train_df[column]

            schema[column] = {
                "dtype": str(col.dtype),
                "nullable": bool(col.isna().any()),
                "unique_values": int(col.nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(col):
                schema[column].update(
                    {
                        "min": float(col.min()),
                        "max": float(col.max()),
                        "mean": float(col.mean()),
                        "median": float(col.median()),
                    }
                )

        return schema

    # =========================================================
    # METADATA
    # =========================================================

    def _target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        counts = df[self.target_column].value_counts(normalize=True)

        return {str(k): round(v, 4) for k, v in counts.items()}

    def _generate_metadata(
        self,
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict:
        """
        Generate experiment lineage metadata.
        """
        total = len(train_df) + len(val_df) + len(test_df)

        return {
            "dataset": {
                "id": "customer_churn",
                "pipeline_version": self.PIPELINE_VERSION,
                "target_column": self.target_column,
                "feature_count": cleaned_df.shape[1] - 1,
            },
            "source": {
                "type": "mongodb",
                "database": self.config.database_name,
                "collection": self.config.collection_name,
                "raw_rows": len(raw_df),
                "cleaned_rows": len(cleaned_df),
                "raw_checksum": self._compute_checksum(raw_df),
                "cleaned_checksum": self._compute_checksum(cleaned_df),
            },
            "split": {
                "strategy": "stratified",
                "random_state": self.config.random_state,
                "counts": {
                    "train": len(train_df),
                    "validation": len(val_df),
                    "test": len(test_df),
                },
                "ratios": {
                    "train": round(len(train_df) / total, 4),
                    "validation": round(len(val_df) / total, 4),
                    "test": round(len(test_df) / total, 4),
                },
                "target_distribution": {
                    "train": self._target_distribution(train_df),
                    "validation": self._target_distribution(val_df),
                    "test": self._target_distribution(test_df),
                },
                "checksums": {
                    "train": self._compute_checksum(train_df),
                    "validation": self._compute_checksum(val_df),
                    "test": self._compute_checksum(test_df),
                },
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================
    # PIPELINE ENTRYPOINT
    # =========================================================

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the full data ingestion pipeline.
        """
        try:
            logging.info("[DATA INGESTION PIPELINE] Started")

            raw_df = self._load_from_mongodb()
            cleaned_df = self._clean_dataframe(raw_df)

            train_df, val_df, test_df = self._split_data(cleaned_df)

            for path in [
                self.config.train_file_path,
                self.config.val_file_path,
                self.config.test_file_path,
                self.config.schema_file_path,
                self.config.metadata_file_path,
            ]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            train_df.to_csv(self.config.train_file_path, index=False)
            val_df.to_csv(self.config.val_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            schema = self._generate_schema(train_df)
            write_json_file(self.config.schema_file_path, schema)

            metadata = self._generate_metadata(
                raw_df,
                cleaned_df,
                train_df,
                val_df,
                test_df,
            )

            write_json_file(self.config.metadata_file_path, metadata)

            artifact = DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                val_file_path=self.config.val_file_path,
                test_file_path=self.config.test_file_path,
                schema_file_path=self.config.schema_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info(
                "[DATA INGESTION PIPELINE] Completed successfully"
            )
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[DATA INGESTION PIPELINE] Failed")
            raise CustomerChurnException(e, sys) from e