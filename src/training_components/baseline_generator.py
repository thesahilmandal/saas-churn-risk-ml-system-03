"""
Baseline Distribution Generator.

Purpose
-------
Generate a statistical baseline report from the training dataset.

This baseline serves as the reference distribution used for monitoring:
- Input feature drift
- Label distribution shift

Design Principles
-----------------
- Deterministic computation
- No file I/O (pure computation component)
- Monitoring-ready structure
- Clear observability via structured logging
"""

import sys
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

from src.constants.pipeline_constants import TARGET_COLUMN
from src.exception import CustomerChurnException
from src.logging import logging


class BaselineGenerator:
    """
    Production-grade baseline distribution generator.
    """

    PIPELINE_VERSION = "1.0.0"

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self) -> None:
        try:
            logging.info(
                "[BASELINE INIT] BaselineGenerator initialized | "
                "pipeline_version=%s",
                self.PIPELINE_VERSION,
            )
        except Exception as exc:
            logging.exception("[BASELINE INIT] Initialization failed.")
            raise CustomerChurnException(exc, sys) from exc

    # ==========================================================
    # NUMERIC FEATURE STATISTICS
    # ==========================================================

    @staticmethod
    def _numeric_feature_summary(series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for numerical feature."""
        return {
            "feature_type": "numerical",
            "pandas_dtype": str(series.dtype),
            "missing_ratio": float(series.isna().mean()),
            "statistics": {
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
            },
        }

    # ==========================================================
    # CATEGORICAL FEATURE STATISTICS
    # ==========================================================

    @staticmethod
    def _categorical_feature_summary(series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for categorical feature."""

        distribution = (
            series.value_counts(normalize=True, dropna=False)
            .to_dict()
        )

        allowed_values = (
            series.dropna().unique().tolist()
        )

        return {
            "feature_type": "categorical",
            "pandas_dtype": str(series.dtype),
            "missing_ratio": float(series.isna().mean()),
            "allowed_values": [str(v) for v in allowed_values],
            "distribution": {
                str(k): float(v) for k, v in distribution.items()
            },
        }

    # ==========================================================
    # TARGET BASELINE
    # ==========================================================

    @staticmethod
    def _compute_target_baseline(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute target column distribution."""

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found in dataset."
            )

        distribution = (
            df[TARGET_COLUMN]
            .value_counts(normalize=True, dropna=False)
            .to_dict()
        )

        return {
            "target_column": TARGET_COLUMN,
            "feature_type": "categorical",
            "distribution": {
                str(k): float(v) for k, v in distribution.items()
            },
        }

    # ==========================================================
    # FEATURE BASELINE GENERATION
    # ==========================================================

    def _compute_feature_baselines(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compute baseline statistics for all features.
        """

        try:

            features: Dict[str, Any] = {}

            for column in df.columns:

                if column == TARGET_COLUMN:
                    continue

                series = df[column]

                if pd.api.types.is_numeric_dtype(series):
                    features[column] = self._numeric_feature_summary(series)
                else:
                    features[column] = self._categorical_feature_summary(series)

            return features

        except Exception as exc:
            logging.exception("[BASELINE] Feature baseline computation failed.")
            raise CustomerChurnException(exc, sys) from exc

    # ==========================================================
    # BASELINE REPORT GENERATION
    # ==========================================================

    def generate_baseline_report(
        self,
        train_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generate baseline monitoring report.
        """

        try:

            if train_df.empty:
                raise ValueError("Input training DataFrame is empty.")

            logging.info(
                "[BASELINE PIPELINE] Baseline generation started | rows=%d cols=%d",
                train_df.shape[0],
                train_df.shape[1],
            )

            feature_baselines = self._compute_feature_baselines(train_df)

            target_baseline = self._compute_target_baseline(train_df)

            report: Dict[str, Any] = {
                "metadata": {
                    "pipeline_version": self.PIPELINE_VERSION,
                    "generated_at_utc": datetime.now(
                        timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "dataset_rows": int(train_df.shape[0]),
                    "feature_count": int(train_df.shape[1] - 1),
                },
                "features": feature_baselines,
                "target_baseline": target_baseline,
            }

            logging.info(
                "[BASELINE PIPELINE] Baseline report generated successfully"
            )

            return report

        except Exception as exc:
            logging.exception("[BASELINE PIPELINE] Baseline generation failed")
            raise CustomerChurnException(exc, sys) from exc
              

if __name__ == "__main__":
    from src.utils.main_utils import write_json_file

    df = pd.read_csv(r"/workspaces/saas-churn-risk-ml-system-03/artifacts/training_pipeline_runs/20260310_073106/02_data_ingestion/train.csv")
    baseline_generator = BaselineGenerator()
    report = baseline_generator.generate_baseline_report(df)

    write_json_file("rough/rough_2.py", report)
    print(report)