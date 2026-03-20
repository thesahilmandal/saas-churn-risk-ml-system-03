"""
Centralized configuration constants for the ML training pipeline.

Defines:
- Directory names
- File names
- Environment variables
- Model configuration
- Pipeline-wide constants

NOTE:
-----
⚠ No runtime logic should depend on side effects here.
⚠ Only constants are defined.
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ==========================================================
# ENVIRONMENT INITIALIZATION
# ==========================================================

# Load environment variables once at module import
load_dotenv()


# ==========================================================
# GLOBAL PIPELINE CONSTANTS
# ==========================================================

TARGET_COLUMN: str = "Churn"

LOGS_DIR: Path = Path("logs")
ARTIFACT_DIR: Path = Path("artifacts")

S3_BUCKET_NAME: str = "saas-customer-churn-ml-03"
S3_ARTIFACT_DIR_NAME: str = "artifacts"
S3_MODEL_REGISTRY_DIR_NAME: str = "model_registry"

RANDOM_STATE: int = 42
REFERENCE_SCHEMA_PATH: str = "data_schema/v1/schema.yaml"

MODEL_REGISTRY_DIR: Path = Path("model_registry")
MODEL_REGISTRY_METADATA_PATH: Path = MODEL_REGISTRY_DIR / "registry_metadata.json"

MONITORING_BASELINE_PATH: Path = Path("online") / "monitoring_baseline.json"

LOCK_FILE_PATH: str = "/tmp/churn_orchestrator.lock"


# ==========================================================
# ETL CONSTANTS
# ==========================================================

ETL_DIR_NAME: str = "01_etl"
ETL_METADATA_FILE_NAME: str = "metadata.json"
ETL_RAW_DATA_DIR_NAME: str = "raw_data"


# ==========================================================
# DATA INGESTION CONSTANTS
# ==========================================================

DATA_INGESTION_DIR_NAME: str = "02_data_ingestion"

DATA_INGESTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_INGESTION_TEST_FILE_NAME: str = "test.csv"
DATA_INGESTION_VAL_FILE_NAME: str = "val.csv"

DATA_INGESTION_SCHEMA_FILE_NAME: str = "ingestion_schema.json"
DATA_INGESTION_METADATA_FILE_NAME: str = "metadata.json"

DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO: float = 0.30
DATA_INGESTION_TEST_VAL_SPLIT_RATIO: float = 0.50


# ==========================================================
# DATA VALIDATION CONSTANTS
# ==========================================================

DATA_VALIDATION_DIR_NAME: str = "03_data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.json"


# ==========================================================
# DATA TRANSFORMATION CONSTANTS
# ==========================================================

DATA_TRANSFORMATION_DIR_NAME: str = "04_data_transformation"

DATA_TRANSFORMATION_LINEAR_PREPROCESSOR_FILE_NAME: str = "lr_preprocessor.pkl"

DATA_TRANSFORMATION_TREE_PREPROCESSOR_FILE_NAME: str = "tree_preprocessor.pkl"

DATA_TRANSFORMATION_METADATA_FILE_NAME: str = "metadata.json"

DATA_TRANSFORMATION_MONITORING_BASELINE_FILE_NAME: str = (
    "monitoring_baseline.json"
)


# ==========================================================
# MODEL TRAINING CONSTANTS
# ==========================================================

MODEL_TRAINING_DIR_NAME: str = "05_model_training"
MODEL_TRAINING_TRAINED_MODELS_DIR_NAME: str = "trained_models"
MODEL_TRAINING_METADATA_FILE_NAME: str = "metadata.json"


# Registry of candidate models used in the training pipeline
MODEL_TRAINING_MODELS_REGISTERY: dict = {
    "logistic_regression": LogisticRegression(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "random_forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        random_state=42
    ),
}


# Hyperparameter search space for each candidate model
MODEL_TRAINING_MODELS_HYPERPARAMETERS: dict = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "max_iter": [100, 300, 500],
        "solver": ["liblinear", "lbfgs", "saga"],
    },
    "random_forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    "gradient_boosting": {
        "n_estimators": [100, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    },
}

MODEL_TRAINING_PRIMARY_METRIC: str = "recall"
MODEL_TRAINING_DECISION_THRESHOLD: float = 0.5
MODEL_TRAINING_N_ITER: int = 1


# ==========================================================
# MODEL EVALUATION CONSTANTS
# ==========================================================

MODEL_EVALUATION_DIR_NAME: str = "06_model_evaluation"

MODEL_EVALUATION_REPORT_FILE_NAME: str = "evaluation_report.json"

MODEL_EVALUATION_METADATA_FILE_NAME: str = "metadata.json"

MODEL_EVALUATION_DECISION_THRESHOLD: float = 0.50
MODEL_EVALUATION_RECALL_TOLERANCE: float = 0.005
MODEL_EVALUATION_MIN_IMPROVEMENT: float = 0.01