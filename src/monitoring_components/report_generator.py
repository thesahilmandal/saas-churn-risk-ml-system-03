"""
Report Generator for the Monitoring Pipeline.

Responsibilities
----------------
1. Package the raw drift scores into a structured JSON report.
2. Attach execution metadata (timestamps, versions).
3. Save the report to the artifacts directory.
4. Output the official DriftAnalysisArtifact.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from src.entity.artifact_entity import (
    BaselineFetchArtifact,
    DriftAnalysisArtifact,
    TelemetryExtractionArtifact,
)
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file


class ReportGenerator:
    """
    Compiles drift metrics into a standardized, auditable artifact.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            os.makedirs(self.config.drift_report_dir, exist_ok=True)
            logging.info("[REPORT GENERATOR] Initialized successfully.")
        except Exception as e:
            logging.exception("[REPORT GENERATOR] Initialization failed.")
            raise CustomerChurnException(e, sys) from e

    def initiate_report_generation(
        self,
        raw_analysis_results: Dict[str, Any],
        baseline_artifact: BaselineFetchArtifact,
        telemetry_artifact: TelemetryExtractionArtifact,
    ) -> DriftAnalysisArtifact:
        """
        Generate and save the drift report.
        """
        try:
            logging.info("[REPORT GENERATOR] Execution started.")

            # Overall system drift threshold: if > 15% of features drift, we flag the system.
            # You can tune this logic based on your business rules.
            total_features = raw_analysis_results.get("total_features_analyzed", 1)
            drifted_features = raw_analysis_results.get("total_drifted_features", 0)
            
            system_drift_detected = (drifted_features / total_features) >= 0.15

            report = {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "champion_model_version": baseline_artifact.champion_model_version,
                    "telemetry_window_start": telemetry_artifact.window_start_date,
                    "telemetry_window_end": telemetry_artifact.window_end_date,
                    "rows_analyzed": telemetry_artifact.extracted_rows,
                },
                "summary": {
                    "system_drift_detected": system_drift_detected,
                    "total_features": total_features,
                    "drifted_features": drifted_features,
                },
                "feature_details": raw_analysis_results.get("feature_metrics", {})
            }

            write_json_file(self.config.drift_report_path, report)

            artifact = DriftAnalysisArtifact(
                drift_report_file_path=self.config.drift_report_path,
                drift_detected=system_drift_detected
            )

            logging.info(
                f"[REPORT GENERATOR] Report saved | Drift Detected: {system_drift_detected} | "
                f"Path: {self.config.drift_report_path}"
            )
            return artifact

        except Exception as e:
            logging.exception("[REPORT GENERATOR] Execution failed.")
            raise CustomerChurnException(e, sys) from e