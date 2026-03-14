"""
Statistical Engine for the Monitoring Pipeline.

Responsibilities
----------------
1. Load the baseline JSON and telemetry CSV.
2. Compute the Population Stability Index (PSI) for categorical features.
3. Compute PSI for numerical features using quantile binning.
4. Output a raw dictionary of drift scores.
"""

import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.entity.artifact_entity import BaselineFetchArtifact, TelemetryExtractionArtifact
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file


class DriftAnalyzer:
    """
    Computes statistical distribution drift between production telemetry and training baseline.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            logging.info("[DRIFT ANALYZER] Initialized successfully.")
        except Exception as e:
            logging.exception("[DRIFT ANALYZER] Initialization failed.")
            raise CustomerChurnException(e, sys) from e

    @staticmethod
    def _calculate_psi(expected: List[float], actual: List[float]) -> float:
        """
        Calculate the Population Stability Index (PSI) between two distributions.
        Adds a tiny epsilon to prevent division by zero or log(0).
        """
        eps = 1e-6
        expected_arr = np.array(expected) + eps
        actual_arr = np.array(actual) + eps

        # Normalize to ensure they sum to exactly 1
        expected_arr = expected_arr / expected_arr.sum()
        actual_arr = actual_arr / actual_arr.sum()

        psi_values = (actual_arr - expected_arr) * np.log(actual_arr / expected_arr)
        return float(np.sum(psi_values))

    def _analyze_categorical(
        self, current_series: pd.Series, baseline_dist: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compute PSI for categorical features.
        """
        # Current distribution
        current_counts = current_series.value_counts(normalize=True, dropna=False).to_dict()

        # Align keys
        all_keys = set(baseline_dist.keys()).union(set(current_counts.keys()))
        
        expected_props = []
        actual_props = []

        for key in all_keys:
            expected_props.append(baseline_dist.get(str(key), 0.0))
            actual_props.append(current_counts.get(key, 0.0))

        psi_score = self._calculate_psi(expected_props, actual_props)
        is_drifted = psi_score > self.config.cat_drift_threshold

        return {
            "drift_score_psi": round(psi_score, 4),
            "is_drifted": is_drifted,
            "threshold": self.config.cat_drift_threshold
        }

    def _analyze_numerical(
        self, current_series: pd.Series, baseline_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute PSI for numerical features using baseline quantiles as bin edges.
        """
        quantiles = baseline_stats.get("quantiles", {})
        min_val = baseline_stats.get("min", current_series.min())
        max_val = baseline_stats.get("max", current_series.max())

        # Define bin edges using training quantiles. We use -inf and inf for bounds 
        # to capture new extreme values in production.
        bins = [
            -np.inf,
            quantiles.get("p10", min_val),
            quantiles.get("p25", min_val),
            quantiles.get("p50", min_val),
            quantiles.get("p75", max_val),
            quantiles.get("p90", max_val),
            np.inf
        ]

        # The expected proportions matching the bins defined above
        expected_props = [0.10, 0.15, 0.25, 0.25, 0.15, 0.10]

        # Cut current data into these bins and count proportions
        current_binned = pd.cut(current_series.dropna(), bins=bins)
        actual_counts = current_binned.value_counts(normalize=True, sort=False).tolist()

        psi_score = self._calculate_psi(expected_props, actual_counts)
        is_drifted = psi_score > self.config.num_drift_threshold

        return {
            "drift_score_psi": round(psi_score, 4),
            "is_drifted": is_drifted,
            "threshold": self.config.num_drift_threshold,
            "mean_shift": round(float(current_series.mean() - baseline_stats.get("mean", 0)), 4)
        }

    def initiate_analysis(
        self,
        baseline_artifact: BaselineFetchArtifact,
        telemetry_artifact: TelemetryExtractionArtifact,
    ) -> Dict[str, Any]:
        """
        Execute drift analysis across all features found in the baseline.
        """
        try:
            logging.info("[DRIFT ANALYZER] Execution started.")

            baseline_data = read_json_file(baseline_artifact.baseline_file_path)
            telemetry_df = pd.read_csv(telemetry_artifact.telemetry_data_path)

            baseline_features = baseline_data.get("features", {})
            analysis_results = {}
            total_drifted_features = 0

            for feature_name, feature_meta in baseline_features.items():
                if feature_name not in telemetry_df.columns:
                    logging.warning(f"[DRIFT ANALYZER] Feature '{feature_name}' missing in telemetry.")
                    continue

                feature_type = feature_meta.get("feature_type")
                current_series = telemetry_df[feature_name]

                if feature_type == "numerical":
                    result = self._analyze_numerical(current_series, feature_meta.get("statistics", {}))
                elif feature_type == "categorical":
                    result = self._analyze_categorical(current_series, feature_meta.get("distribution", {}))
                else:
                    continue

                if result["is_drifted"]:
                    total_drifted_features += 1

                analysis_results[feature_name] = result

            logging.info(f"[DRIFT ANALYZER] Analysis complete. Detected drift in {total_drifted_features} features.")

            return {
                "feature_metrics": analysis_results,
                "total_drifted_features": total_drifted_features,
                "total_features_analyzed": len(analysis_results)
            }

        except Exception as e:
            logging.exception("[DRIFT ANALYZER] Execution failed.")
            raise CustomerChurnException(e, sys) from e