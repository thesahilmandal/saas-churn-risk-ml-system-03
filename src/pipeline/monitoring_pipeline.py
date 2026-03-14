"""
Master DAG for the Monitoring and Continual Learning Pipeline.

Responsibilities
----------------
1. Instantiate configurations.
2. Execute the monitoring steps sequentially.
3. Manage artifact hand-offs between steps.
4. If the Retraining Trigger fires, initialize and execute Phase 1 (Training).
"""

import sys

from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging

# Monitoring Components
from src.monitoring_components.baseline_fetcher import BaselineFetcher
from src.monitoring_components.drift_analyzer import DriftAnalyzer
from src.monitoring_components.report_generator import ReportGenerator
from src.monitoring_components.retraining_trigger import RetrainingTrigger
from src.monitoring_components.telemetry_extractor import TelemetryExtractor

# Training Pipeline (Phase 1)
from src.pipeline.training_pipeline import TrainingPipeline


class MonitoringPipeline:
    """
    Production-grade MLOps Continual Learning Orchestrator.
    """

    def __init__(self) -> None:
        try:
            self.monitoring_config = MonitoringConfig()
            logging.info("MONITORING PIPELINE INITIALIZED\n")
        except Exception as exc:
            raise CustomerChurnException(exc, sys) from exc

    def run_pipeline(self) -> None:
        """
        Executes the end-to-end monitoring and continual learning workflow.
        """
        try:
            logging.info("=" * 60)
            logging.info("MONITORING PIPELINE EXECUTION STARTED")
            logging.info("=" * 60)

            # Step 1: Fetch the baseline of the current champion model
            baseline_fetcher = BaselineFetcher(self.monitoring_config)
            baseline_artifact = baseline_fetcher.initiate_baseline_fetch()

            # Step 2: Extract recent production telemetry from MongoDB
            telemetry_extractor = TelemetryExtractor(self.monitoring_config)
            telemetry_artifact = telemetry_extractor.initiate_extraction()

            # Step 3: Compute mathematical drift metrics
            drift_analyzer = DriftAnalyzer(self.monitoring_config)
            raw_analysis_results = drift_analyzer.initiate_analysis(
                baseline_artifact, telemetry_artifact
            )

            # Step 4: Package metrics into a structured report artifact
            report_generator = ReportGenerator(self.monitoring_config)
            drift_artifact = report_generator.initiate_report_generation(
                raw_analysis_results, baseline_artifact, telemetry_artifact
            )

            # Step 5: Evaluate business rules to determine if we should retrain
            trigger = RetrainingTrigger(self.monitoring_config)
            trigger_artifact = trigger.initiate_trigger(drift_artifact)

            logging.info("=" * 60)
            logging.info("MONITORING PIPELINE EXECUTION COMPLETED")
            logging.info("=" * 60)

            # Step 6: The Continual Learning Feedback Loop
            if trigger_artifact.should_retrain:
                logging.warning(">>> CONTINUAL LEARNING TRIGGERED: INITIATING PHASE 1 <<<")
                
                training_pipeline = TrainingPipeline()
                training_pipeline.run_pipeline()
                
                logging.info(">>> AUTOMATED RETRAINING CYCLE COMPLETED <<<")

        except Exception as exc:
            logging.exception("Monitoring pipeline execution failed.")
            raise CustomerChurnException(exc, sys) from exc


# ==========================================================
# Entry Point
# ==========================================================
if __name__ == "__main__":
    try:
        pipeline = MonitoringPipeline()
        pipeline.run_pipeline()
    except Exception:
        logging.exception("Unhandled exception in main execution")