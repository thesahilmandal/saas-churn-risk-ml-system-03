"""
Decision Engine for the Monitoring Pipeline.

Responsibilities
----------------
1. Read the drift analysis artifact.
2. Maintain a persistent state file tracking previous retraining timestamps.
3. Enforce a cooldown period to prevent infinite retraining loops.
4. Output a strict boolean flag determining if Phase 1 should execute.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

from src.entity.artifact_entity import DriftAnalysisArtifact, RetrainingTriggerArtifact
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file, write_json_file


class RetrainingTrigger:
    """
    Evaluates drift reports and cooldown rules to safely trigger automated retraining.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            logging.info(
                f"[RETRAINING TRIGGER] Initialized. Cooldown period: {self.config.cooldown_days} days."
            )
        except Exception as e:
            logging.exception("[RETRAINING TRIGGER] Initialization failed.")
            raise CustomerChurnException(e, sys) from e

    def _check_cooldown_period(self) -> bool:
        """
        Verify if enough time has passed since the last automated retrain.
        Returns True if safe to retrain, False if currently in cooldown.
        """
        try:
            if not os.path.exists(self.config.trigger_metadata_path):
                # First time the system has ever run or triggered
                return True

            state_data = read_json_file(self.config.trigger_metadata_path)
            last_retrain_str = state_data.get("last_retrained_at_utc")

            if not last_retrain_str:
                return True

            last_retrain_date = datetime.fromisoformat(last_retrain_str)
            current_date = datetime.now(timezone.utc)
            days_since_retrain = (current_date - last_retrain_date).days

            if days_since_retrain < self.config.cooldown_days:
                logging.warning(
                    f"[RETRAINING TRIGGER] Cooldown active. Last retrain was {days_since_retrain} "
                    f"days ago. Required cooldown: {self.config.cooldown_days} days."
                )
                return False

            return True

        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def _update_trigger_state(self) -> None:
        """
        Update the persistent state file with the current timestamp.
        """
        try:
            state_data = {
                "last_retrained_at_utc": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(self.config.trigger_metadata_path, state_data)
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def initiate_trigger(self, drift_artifact: DriftAnalysisArtifact) -> RetrainingTriggerArtifact:
        """
        Determine if the training pipeline should be invoked.
        """
        try:
            logging.info("[RETRAINING TRIGGER] Execution started.")

            should_retrain = False

            if drift_artifact.drift_detected:
                logging.warning("[RETRAINING TRIGGER] System drift detected in the report.")
                
                # Only approve retraining if we are outside the cooldown window
                if self._check_cooldown_period():
                    logging.info("[RETRAINING TRIGGER] Cooldown cleared. Approving retrain.")
                    should_retrain = True
                    self._update_trigger_state()
                else:
                    logging.info("[RETRAINING TRIGGER] Retrain blocked by cooldown policy.")
            else:
                logging.info("[RETRAINING TRIGGER] No system drift detected. Retrain not required.")

            artifact = RetrainingTriggerArtifact(
                should_retrain=should_retrain,
                trigger_metadata_path=self.config.trigger_metadata_path
            )

            return artifact

        except Exception as e:
            logging.exception("[RETRAINING TRIGGER] Execution failed.")
            raise CustomerChurnException(e, sys) from e