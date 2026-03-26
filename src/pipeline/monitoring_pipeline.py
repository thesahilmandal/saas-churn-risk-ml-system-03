"""
Orchestrator for the Monitoring & Continual Learning Subsystem.

Responsibilities:
- Control the execution flow of the monitoring lifecycle.
- Handle component failures gracefully.
- Generate and return the final MonitoringArtifact.
"""

import sys

from src.entity.artifact_entity import MonitoringArtifact
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging

from src.monitoring.baseline_loader import BaselineLoader
from src.monitoring.data_loader import LiveDataLoader
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.stats_generator import CurrentStatsGenerator
from src.monitoring.trigger_engine import TriggerEngine

from src.constants.pipeline_constants import ARTIFACT_DIR, S3_ARTIFACT_DIR_NAME
from src.utils.main_utils import sync_to_s3

class MonitoringPipeline:
    """
    Production-grade ML Monitoring Pipeline Orchestrator.
    """

    def __init__(self) -> None:
        try:
            # We instantiate the base pipeline config to inherit the artifact directory
            self.monitoring_config = MonitoringConfig()
            
            logging.info("MONITORING PIPELINE INITIALIZED\n")
            
        except Exception as exc:
            raise CustomerChurnException(exc, sys)

    def run_pipeline(self) -> MonitoringArtifact:
        """
        Execute the monitoring components sequentially.
        """
        try:
            logging.info("=" * 60)
            logging.info("MONITORING & CONTINUAL LEARNING PIPELINE STARTED")
            logging.info("=" * 60)

            # ---------------------------------------------------------
            # Step 1: Data Acquisition
            # ---------------------------------------------------------
            logging.info(">>>>>> Stage 1: Live Data Loading <<<<<<")
            data_loader = LiveDataLoader(self.monitoring_config)
            live_df = data_loader.fetch_recent_inference_data()
            
            # Graceful exit if there are no new predictions to monitor
            if live_df.empty:
                logging.info("[PIPELINE] No recent inference data found. Exiting pipeline.")
                return MonitoringArtifact(
                    live_data_file_path="",
                    current_stats_file_path="",
                    drift_report_file_path="",
                    drift_detected=False
                )

            logging.info(">>>>>> Stage 2: Baseline Loading <<<<<<")
            baseline_loader = BaselineLoader(self.monitoring_config)
            baseline_dict = baseline_loader.fetch_baseline()

            # ---------------------------------------------------------
            # Step 2: Core Monitoring Logic
            # ---------------------------------------------------------
            logging.info(">>>>>> Stage 3: Current Stats Generation <<<<<<")
            stats_generator = CurrentStatsGenerator(self.monitoring_config)
            current_stats_dict = stats_generator.generate_current_stats(live_df)

            logging.info(">>>>>> Stage 4: Drift Detection <<<<<<")
            drift_detector = DriftDetector(self.monitoring_config)
            drift_detected, _ = drift_detector.detect_drift(baseline_dict, current_stats_dict)

            # ---------------------------------------------------------
            # Step 3: Action & Orchestration
            # ---------------------------------------------------------
            logging.info(">>>>>> Stage 5: Trigger Engine <<<<<<")
            trigger_engine = TriggerEngine(self.monitoring_config)
            trigger_engine.execute(drift_detected=drift_detected)

            # ---------------------------------------------------------
            # Step 4: Artifact Synchronization (Cloud Backup)
            # ---------------------------------------------------------
            logging.info("Syncing monitoring artifacts to S3...")
            sync_to_s3(ARTIFACT_DIR, S3_ARTIFACT_DIR_NAME)

            artifact = MonitoringArtifact(
                live_data_file_path=self.monitoring_config.live_data_file_path,
                current_stats_file_path=self.monitoring_config.current_stats_file_path,
                drift_report_file_path=self.monitoring_config.drift_report_file_path,
                drift_detected=drift_detected
            )
            logging.info(artifact)

            logging.info("=" * 60)
            logging.info("MONITORING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 60)

            # Return the final artifact footprint
            return artifact

        except Exception as exc:
            logging.exception("Monitoring pipeline execution failed.")
            raise CustomerChurnException(exc, sys)


if __name__ == "__main__":
    try:
        pipeline = MonitoringPipeline()
        pipeline.run_pipeline()
    except Exception:
        logging.exception("Unhandled exception in main monitoring execution.")