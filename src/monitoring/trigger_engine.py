"""
Decision Trigger Engine for the Monitoring Subsystem.

Responsibilities:
- Evaluate the drift_detected boolean flag.
- Enforce a "cool-down" period using an S3-backed state file to prevent infinite retraining loops.
- Authenticate and fire an HTTP POST request to the GitHub Actions REST API.
- Update the S3 state file upon a successful trigger.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

from src.cloud.s3_syncer import S3Sync
from src.constants.pipeline_constants import (
    S3_BUCKET_NAME,
    S3_MODEL_REGISTRY_DIR_NAME,
)
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging

load_dotenv()


class TriggerEngine:
    """
    Safely orchestrates the automated retraining trigger based on data drift.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            self.s3_sync = S3Sync()
            
            # CI/CD Environment Variables
            self.github_token = os.getenv("GITHUB_PAT")
            self.github_repo = os.getenv("GITHUB_REPO")  # e.g., "username/repo-name"
            self.workflow_id = os.getenv("GITHUB_WORKFLOW_ID", "training.yml")

            # Local and Remote paths for the state file
            self.state_file_name = "last_trigger_state.json"
            self.s3_state_uri = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_REGISTRY_DIR_NAME}/{self.state_file_name}"
            self.local_state_path = os.path.join(self.config.monitoring_dir, self.state_file_name)

            logging.info("[TRIGGER ENGINE] Initialized successfully.")

        except Exception as e:
            logging.exception("[TRIGGER ENGINE] Initialization failed.")
            raise CustomerChurnException(e, sys)

    def _is_cooldown_active(self) -> bool:
        """
        Check S3 to see if a retraining pipeline was triggered recently.
        
        Returns
        -------
        bool
            True if we are still within the cooldown window, False otherwise.
        """
        try:
            logging.info("[TRIGGER ENGINE] Checking cooldown state from S3...")
            self.s3_sync.download_file(self.s3_state_uri, self.local_state_path)
            
            with open(self.local_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                
            last_trigger_iso = state.get("last_trigger_utc")
            if not last_trigger_iso:
                return False

            last_trigger = datetime.fromisoformat(last_trigger_iso)
            time_since_trigger = datetime.now(timezone.utc) - last_trigger
            
            is_active = time_since_trigger < timedelta(days=self.config.cooldown_days)
            
            if is_active:
                logging.warning(
                    f"[TRIGGER ENGINE] Cooldown active. Last trigger was {time_since_trigger.days} "
                    f"days and {time_since_trigger.seconds // 3600} hours ago."
                )
            return is_active

        except CustomerChurnException:
            # File doesn't exist in S3 (first time running the pipeline)
            logging.info("[TRIGGER ENGINE] No previous trigger state found in S3. Cooldown inactive.")
            return False
        except Exception as e:
            logging.error(f"[TRIGGER ENGINE] Error checking cooldown, defaulting to safe (inactive): {e}")
            return False

    def _update_trigger_state(self) -> None:
        """
        Update the local state file and push it to S3 to reset the cooldown timer.
        """
        try:
            state = {
                "last_trigger_utc": datetime.now(timezone.utc).isoformat(),
                "reason": "Data Drift Detected"
            }
            
            with open(self.local_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
                
            # We use an S3 CLI command or custom method to upload the single file
            upload_cmd = f"aws s3 cp {self.local_state_path} {self.s3_state_uri}"
            os.system(upload_cmd)
            
            logging.info("[TRIGGER ENGINE] S3 cooldown state updated successfully.")
            
        except Exception as e:
            logging.error(f"[TRIGGER ENGINE] Failed to update S3 trigger state: {e}")

    def _trigger_github_action(self) -> None:
        """
        Fire a webhook to the GitHub Actions REST API to start the Training Pipeline.
        """
        if not all([self.github_token, self.github_repo, self.workflow_id]):
            logging.warning("[TRIGGER ENGINE] Missing GitHub credentials. Cannot trigger retraining.")
            return

        logging.info(f"[TRIGGER ENGINE] Dispatching GitHub workflow: {self.workflow_id}")

        url = f"https://api.github.com/repos/{self.github_repo}/actions/workflows/{self.workflow_id}/dispatches"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {self.github_token}"
        }
        payload = {"ref": "main"}  # Trigger the workflow on the main branch

        response = requests.post(url, headers=headers, json=payload, timeout=10)

        if response.status_code == 204:
            logging.info("[TRIGGER ENGINE] Successfully triggered GitHub Actions Training Pipeline.")
            self._update_trigger_state()
        else:
            logging.error(
                f"[TRIGGER ENGINE] Failed to trigger GitHub Actions. "
                f"Status: {response.status_code}, Response: {response.text}"
            )

    def execute(self, drift_detected: bool) -> None:
        """
        Execute the decision logic based on the drift report and cooldown status.
        """
        logging.info("=" * 50)
        logging.info(f"[TRIGGER ENGINE] Execution started | Drift Detected: {drift_detected}")
        
        if not drift_detected:
            logging.info("[TRIGGER ENGINE] No drift detected. System is healthy. Exiting cleanly.")
            return

        if self._is_cooldown_active():
            logging.info("[TRIGGER ENGINE] Retraining skipped due to active cooldown.")
            return

        logging.info("[TRIGGER ENGINE] Conditions met. Initiating retraining protocol.")
        self._trigger_github_action()