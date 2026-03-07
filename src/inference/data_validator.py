"""
Data Validation Gatekeeper for the Inference Service.

Responsibilities:
- Load the exact reference schema used during model training.
- Apply structural and row-level checks on incoming inference batches.
- Perform basic deterministic cleaning (matching Data Ingestion step).
- Separate valid rows (for inference) from invalid rows (for the Dead Letter Queue).
"""

import sys
from typing import List, Tuple

import pandas as pd

from src.constants.pipeline_constants import TARGET_COLUMN
from src.exception import CustomerChurnException
from src.inference.schemas import ErrorDetail
from src.logging import logging
from src.utils.main_utils import read_yaml_file


class DataValidator:
    """
    Production-grade vectorized data validator.
    Ensures input data strictly adheres to the training schema contract.
    """

    def __init__(self, schema_file_path: str) -> None:
        try:
            self.schema_path = schema_file_path
            self.schema = read_yaml_file(self.schema_path)
            
            # Extract column rules, ignoring the target variable for inference
            self.columns_rules = self.schema.get("columns", {})
            if TARGET_COLUMN in self.columns_rules:
                del self.columns_rules[TARGET_COLUMN]

            logging.info("[DATA VALIDATOR] Initialized. Schema loaded successfully.")

        except Exception as e:
            logging.exception("[DATA VALIDATOR] Failed to initialize.")
            raise CustomerChurnException(e, sys)

    def _apply_basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the exact same deterministic cleaning logic used in the 
        Data Ingestion pipeline to prevent preprocessing skew.
        """
        try:
            df_clean = df.copy()

            # Convert TotalCharges to numeric, coercing errors to NaN 
            # (which will be caught by the null validator or imputed later)
            if "TotalCharges" in df_clean.columns:
                df_clean["TotalCharges"] = pd.to_numeric(
                    df_clean["TotalCharges"], errors="coerce"
                )

            # Map SeniorCitizen to Yes/No to match training distribution
            if "SeniorCitizen" in df_clean.columns:
                df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map(
                    {1: "Yes", 0: "No", "1": "Yes", "0": "No"}
                ).fillna(df_clean["SeniorCitizen"])

            # Standardize missing string representations
            df_clean = df_clean.replace({ "na": None, "": None, " ": None })

            return df_clean

        except Exception as e:
            logging.exception("[DATA VALIDATOR] Basic cleaning failed.")
            raise CustomerChurnException(e, sys)

    def validate_batch(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[ErrorDetail]]:
        """
        Validates the incoming DataFrame against the reference schema.

        Args:
            df: The raw uploaded pandas DataFrame.

        Returns:
            Tuple containing:
            - valid_df: Pristine DataFrame safe for prediction.
            - invalid_df: Raw DataFrame containing only the failed rows.
            - errors: List of Pydantic ErrorDetail objects for the DLQ.
        """
        try:
            logging.info(f"[DATA VALIDATOR] Starting batch validation for {len(df)} rows.")

            # 1. Structural Check: Missing Required Columns
            missing_cols = []
            for col, rules in self.columns_rules.items():
                if rules.get("required", False) and col not in df.columns:
                    missing_cols.append(col)

            if missing_cols:
                # If required columns are entirely missing, the whole batch is unprocessable
                raise ValueError(
                    f"Structural validation failed. Missing required columns: {missing_cols}"
                )

            # 2. Apply preliminary cleaning
            df = self._apply_basic_cleaning(df)

            errors: List[ErrorDetail] = []
            invalid_indices = set()

            # 3. Vectorized Row-Level Checks
            for col, rules in self.columns_rules.items():
                if col not in df.columns:
                    continue

                # Rule A: Nullability
                if rules.get("nullable") is False:
                    null_mask = df[col].isna()
                    if null_mask.any():
                        violating_idx = df[null_mask].index
                        invalid_indices.update(violating_idx)
                        for idx in violating_idx:
                            customer_id = str(df.loc[idx, "customerID"]) if "customerID" in df.columns else None
                            errors.append(
                                ErrorDetail(
                                    row_index=int(idx),
                                    customerID=customer_id,
                                    error_reason=f"Column '{col}' contains null/missing values."
                                )
                            )

                # Rule B: Numeric Minimum
                if "min" in rules:
                    num_series = pd.to_numeric(df[col], errors="coerce")
                    min_mask = num_series.notna() & (num_series < rules["min"])
                    if min_mask.any():
                        violating_idx = df[min_mask].index
                        invalid_indices.update(violating_idx)
                        for idx in violating_idx:
                            customer_id = str(df.loc[idx, "customerID"]) if "customerID" in df.columns else None
                            errors.append(
                                ErrorDetail(
                                    row_index=int(idx),
                                    customerID=customer_id,
                                    error_reason=f"Column '{col}' value falls below minimum ({rules['min']})."
                                )
                            )

                # Rule C: Numeric Maximum
                if "max" in rules:
                    num_series = pd.to_numeric(df[col], errors="coerce")
                    max_mask = num_series.notna() & (num_series > rules["max"])
                    if max_mask.any():
                        violating_idx = df[max_mask].index
                        invalid_indices.update(violating_idx)
                        for idx in violating_idx:
                            customer_id = str(df.loc[idx, "customerID"]) if "customerID" in df.columns else None
                            errors.append(
                                ErrorDetail(
                                    row_index=int(idx),
                                    customerID=customer_id,
                                    error_reason=f"Column '{col}' value exceeds maximum ({rules['max']})."
                                )
                            )

            # 4. Split the DataFrame
            invalid_idx_list = list(invalid_indices)
            
            invalid_df = df.loc[invalid_idx_list].copy()
            valid_df = df.drop(index=invalid_idx_list).copy()

            logging.info(
                f"[DATA VALIDATOR] Validation complete | "
                f"Valid rows: {len(valid_df)} | Invalid rows: {len(invalid_df)}"
            )

            return valid_df, invalid_df, errors

        except Exception as e:
            logging.exception("[DATA VALIDATOR] Validation pipeline failed.")
            raise CustomerChurnException(e, sys)