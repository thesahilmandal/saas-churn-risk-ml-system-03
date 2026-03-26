"""
Data Drift Detector for the Monitoring Subsystem.

Responsibilities:
- Compare the S3 Baseline against the Current Stats.
- Compute Population Stability Index (PSI) for categorical features.
- Compute Relative Mean Shift for numerical features.
- Generate a comprehensive drift report and a boolean trigger flag.
"""

import math
import sys
from typing import Any, Dict, Tuple

from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file


class DriftDetector:
    """
    Evaluates statistical divergence between expected and actual data distributions.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            self.psi_threshold = config.psi_threshold
            
            # Threshold for numerical mean shift (e.g., 10% change)
            self.numerical_shift_threshold = 0.10 
            
            logging.info(
                f"[DRIFT DETECTOR] Initialized | "
                f"PSI Threshold: {self.psi_threshold} | "
                f"Mean Shift Threshold: {self.numerical_shift_threshold}"
            )
        except Exception as e:
            logging.exception("[DRIFT DETECTOR] Initialization failed.")
            raise CustomerChurnException(e, sys)

    @staticmethod
    def _calculate_psi(expected_dist: Dict[str, float], actual_dist: Dict[str, float]) -> float:
        """
        Calculate the Population Stability Index (PSI) for two distributions.
        """
        psi = 0.0
        epsilon = 1e-4  # Prevent division by zero or log(0)

        all_keys = set(expected_dist.keys()).union(set(actual_dist.keys()))

        for key in all_keys:
            exp_val = max(expected_dist.get(str(key), 0.0), epsilon)
            act_val = max(actual_dist.get(str(key), 0.0), epsilon)

            psi += (act_val - exp_val) * math.log(act_val / exp_val)

        return round(psi, 4)

    def detect_drift(
        self, baseline: Dict[str, Any], current_stats: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare baseline distributions against current live distributions.
        
        Parameters
        ----------
        baseline : Dict[str, Any]
            The historical ground truth from S3.
        current_stats : Dict[str, Any]
            The live distribution generated from recent inference data.
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            A boolean indicating if drift was detected, and the detailed drift report.
        """
        try:
            if not baseline or not current_stats:
                raise ValueError("Missing data: Both baseline and current_stats must be populated.")

            logging.info("[DRIFT DETECTOR] Starting drift evaluation...")

            baseline_features = baseline.get("features", {})
            current_features = current_stats.get("features", {})

            drift_report = {
                "summary": {
                    "total_features_evaluated": 0,
                    "drifted_features_count": 0,
                    "drift_detected": False
                },
                "feature_details": {}
            }

            any_drift_detected = False

            # Loop through all features tracked in the baseline
            for feature_name, base_metrics in baseline_features.items():
                if feature_name not in current_features:
                    logging.warning(f"[DRIFT DETECTOR] Feature '{feature_name}' missing in live data. Skipping.")
                    continue

                curr_metrics = current_features[feature_name]
                feature_type = base_metrics.get("feature_type")
                
                drift_report["summary"]["total_features_evaluated"] += 1
                
                is_drifted = False
                drift_metric_value = 0.0
                metric_name = "unknown"

                # Categorical Check (PSI)
                if feature_type == "categorical":
                    metric_name = "psi"
                    expected_dist = base_metrics.get("distribution", {})
                    actual_dist = curr_metrics.get("distribution", {})
                    
                    drift_metric_value = self._calculate_psi(expected_dist, actual_dist)
                    is_drifted = drift_metric_value > self.psi_threshold

                # Numerical Check (Mean Shift)
                elif feature_type == "numerical":
                    metric_name = "relative_mean_shift"
                    expected_mean = base_metrics.get("statistics", {}).get("mean", 0.0)
                    actual_mean = curr_metrics.get("statistics", {}).get("mean", 0.0)
                    
                    if expected_mean != 0:
                        drift_metric_value = abs(actual_mean - expected_mean) / abs(expected_mean)
                    else:
                        drift_metric_value = abs(actual_mean) # Fallback if expected mean is exactly 0
                        
                    drift_metric_value = round(drift_metric_value, 4)
                    is_drifted = drift_metric_value > self.numerical_shift_threshold

                # Record results
                drift_report["feature_details"][feature_name] = {
                    "type": feature_type,
                    "metric": metric_name,
                    "score": drift_metric_value,
                    "is_drifted": is_drifted
                }

                if is_drifted:
                    any_drift_detected = True
                    drift_report["summary"]["drifted_features_count"] += 1
                    logging.warning(f"[DRIFT] Feature '{feature_name}' drifted | {metric_name}: {drift_metric_value}")

            drift_report["summary"]["drift_detected"] = any_drift_detected

            # Persist report
            write_json_file(self.config.drift_report_file_path, drift_report)
            
            logging.info(
                f"[DRIFT DETECTOR] Evaluation complete | Drift Detected: {any_drift_detected} | "
                f"Features Drifted: {drift_report['summary']['drifted_features_count']}"
            )

            return any_drift_detected, drift_report

        except Exception as e:
            logging.exception("[DRIFT DETECTOR] Drift detection failed.")
            raise CustomerChurnException(e, sys)