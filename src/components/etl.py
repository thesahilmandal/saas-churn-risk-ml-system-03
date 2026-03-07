"""
ETL pipeline for Customer Churn data ingestion.

Responsibilities
---------------
1. Extract raw data from Kaggle
2. Apply lightweight, non-destructive transformations
3. Load cleaned data into MongoDB
4. Generate metadata for observability and auditability

Important
---------
This ETL layer intentionally avoids:
- Feature engineering
- Label processing
- Model-related logic
"""

import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from src.entity.artifact_entity import ETLArtifact
from src.entity.config_entity import ETLconfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file


load_dotenv()


class CustomerChurnETL:
    """
    Production-grade ETL orchestrator.

    Pipeline Flow
    -------------
        Extract → Transform → Load → Metadata

    Design Principles
    -----------------
    - Deterministic execution
    - Fail-fast behavior
    - Observability via logging and metadata
    - Modular, maintainable architecture
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        etl_config: ETLconfig,
        delete_old_data: bool = True,
    ) -> None:
        """
        Initialize ETL pipeline.

        Parameters
        ----------
        etl_config : ETLconfig
            Configuration object containing pipeline settings.
        delete_old_data : bool
            If True, existing records in the MongoDB collection
            will be deleted before inserting new records.
        """
        try:
            self.config = etl_config
            self.delete_old_data = delete_old_data

            # MongoDB configuration
            self.mongodb_url = self.config.database_url
            self.database_name = self.config.database_name
            self.collection_name = self.config.collection_name

            # Data source
            self.data_source = self.config.data_source

            self._validate_configuration()

            # TLS certificate
            self.ca_file = certifi.where()

            # Raw data directory
            self.raw_data_dir = Path(self.config.raw_data_dir)
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)

            # MongoDB client (connection pooling enabled)
            self.client = pymongo.MongoClient(
                self.mongodb_url,
                tlsCAFile=self.ca_file,
                serverSelectionTimeoutMS=5000,
            )

            logging.info(
                "[ETL INIT] CustomerChurnETL initialized successfully."
            )

        except Exception as e:
            logging.exception("[ETL INIT] Initialization failed.")
            raise CustomerChurnException(e, sys) from e

    # ==========================================================
    # CONFIG VALIDATION
    # ==========================================================

    def _validate_configuration(self) -> None:
        """
        Validate required configuration values.
        """
        try:
            required_config = {
                "database_url": self.mongodb_url,
                "database_name": self.database_name,
                "collection_name": self.collection_name,
                "data_source": self.data_source,
            }

            missing = [k for k, v in required_config.items() if not v]

            if missing:
                raise EnvironmentError(
                    f"Missing required configuration values: {missing}"
                )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    # ==========================================================
    # EXTRACT
    # ==========================================================

    def extract_data(self) -> pd.DataFrame:
        """
        Download dataset from Kaggle and load the first CSV file.

        Returns
        -------
        pd.DataFrame
            Extracted dataset.
        """
        try:
            logging.info(
                "[ETL EXTRACT] Starting Kaggle dataset download | source=%s",
                self.data_source,
            )

            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    self.data_source,
                    "-p",
                    str(self.raw_data_dir),
                    "--unzip",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )

            logging.debug("[ETL EXTRACT] Kaggle CLI execution completed.")

            csv_files = sorted(self.raw_data_dir.glob("*.csv"))

            if not csv_files:
                raise FileNotFoundError(
                    "No CSV files found after Kaggle download."
                )

            df = pd.read_csv(csv_files[0])

            if df.empty:
                raise ValueError("Extracted dataset is empty.")

            logging.info(
                "[ETL EXTRACT] Completed | rows=%d cols=%d",
                df.shape[0],
                df.shape[1],
            )

            return df

        except Exception as e:
            logging.exception("[ETL EXTRACT] Extraction failed.")
            raise CustomerChurnException(e, sys) from e

    # ==========================================================
    # TRANSFORM
    # ==========================================================

    def transform_data(
        self,
        df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Perform lightweight, non-destructive transformations.

        Steps
        -----
        - Strip whitespace from string columns
        - Remove duplicate rows
        - Add ingestion metadata

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset.

        Returns
        -------
        List[Dict[str, Any]]
            Cleaned records ready for database insertion.
        """
        try:
            logging.info("[ETL TRANSFORM] Starting transformation.")

            df_clean = df.copy()
            initial_rows = len(df_clean)

            # Trim whitespace in string columns
            for column in df_clean.select_dtypes(include="object"):
                df_clean[column] = df_clean[column].str.strip()

            # Remove duplicates
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)

            # Metadata fields
            df_clean["data_source"] = self.data_source
            df_clean["ingested_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()

            records = df_clean.to_dict(orient="records")

            logging.info(
                "[ETL TRANSFORM] Completed | before=%d after=%d",
                initial_rows,
                len(records),
            )

            return records

        except Exception as e:
            logging.exception("[ETL TRANSFORM] Transformation failed.")
            raise CustomerChurnException(e, sys) from e

    # ==========================================================
    # LOAD
    # ==========================================================

    def load_data(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """
        Insert records into MongoDB.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            Transformed dataset records.

        Returns
        -------
        int
            Number of inserted documents.
        """
        if not records:
            logging.warning(
                "[ETL LOAD] No records available for insertion."
            )
            return 0

        try:
            logging.info(
                "[ETL LOAD] Loading records into MongoDB | db=%s collection=%s",
                self.database_name,
                self.collection_name,
            )

            collection = self.client[self.database_name][
                self.collection_name
            ]

            if self.delete_old_data:
                logging.info(
                    "[ETL LOAD] Removing existing records from collection."
                )
                collection.delete_many({})

            result = collection.insert_many(records, ordered=True)

            inserted_count = len(result.inserted_ids)

            logging.info(
                "[ETL LOAD] Completed | inserted=%d",
                inserted_count,
            )

            return inserted_count

        except Exception as e:
            logging.exception("[ETL LOAD] MongoDB insertion failed.")
            raise CustomerChurnException(e, sys) from e

    # ==========================================================
    # METADATA GENERATION
    # ==========================================================

    def generate_metadata(
        self,
        raw_df: pd.DataFrame,
        records_inserted: int,
    ) -> None:
        """
        Generate ETL metadata for observability and auditing.
        """
        try:
            logging.info("[ETL METADATA] Generating metadata.")

            metadata = {
                "data_source": self.data_source,
                "extracted_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "dataset": {
                    "rows_raw": raw_df.shape[0],
                    "columns": raw_df.shape[1],
                    "column_names": list(raw_df.columns),
                    "dtypes": raw_df.dtypes.astype(str).to_dict(),
                },
                "data_quality": {
                    "duplicate_rows_removed": int(
                        raw_df.duplicated().sum()
                    ),
                    "missing_values_per_column": raw_df.isnull()
                    .sum()
                    .to_dict(),
                },
                "load_target": {
                    "database": self.database_name,
                    "collection": self.collection_name,
                    "records_inserted": records_inserted,
                },
            }

            write_json_file(
                self.config.metadata_file_path,
                metadata,
            )

            logging.info(
                "[ETL METADATA] Metadata written successfully | path=%s",
                self.config.metadata_file_path,
            )

        except Exception as e:
            logging.exception(
                "[ETL METADATA] Metadata generation failed."
            )
            raise CustomerChurnException(e, sys) from e

    # ==========================================================
    # PIPELINE ORCHESTRATION
    # ==========================================================

    def initiate_etl(self) -> ETLArtifact:
        """
        Execute the complete ETL pipeline.

        Returns
        -------
        ETLArtifact
            Artifact containing ETL output paths.
        """
        try:
            logging.info("[ETL PIPELINE] Execution started.")

            raw_df = self.extract_data()
            records = self.transform_data(raw_df)
            inserted_records = self.load_data(records)

            self.generate_metadata(raw_df, inserted_records)

            artifact = ETLArtifact(
                raw_data_dir_path=self.config.raw_data_dir,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info(
                "[ETL PIPELINE] Execution completed successfully."
            )
            logging.info("[ETL PIPELINE] Artifact created.")
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[ETL PIPELINE] Pipeline execution failed.")
            raise CustomerChurnException(e, sys) from e


# ==========================================================
# LOCAL TEST EXECUTION
# ==========================================================

if __name__ == "__main__":
    try:
        from src.entity.config_entity import TrainingPipelineConfig

        training_pipeline_config = TrainingPipelineConfig()
        etl_config = ETLconfig(training_pipeline_config)

        etl = CustomerChurnETL(etl_config)
        artifact = etl.initiate_etl()

        print(artifact)

    except Exception as e:
        raise CustomerChurnException(e, sys) from e