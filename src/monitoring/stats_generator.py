"""
Current Statistics Generator for the Monitoring Subsystem.

Responsibilities:
- Ingest the live inference DataFrame.
- Reuse the training BaselineGenerator to ensure structural parity.
- Handle unlabelled inference data gracefully.
- Persist the current statistics to disk.
"""

import sys
from typing import Any, Dict

import pandas as pd

from src.constants.pipeline_constants import TARGET_COLUMN
from src.entity.config_entity import MonitoringConfig
from src.exception import CustomerChurnException
from src.logging import logging
from src.training_components.baseline_generator import BaselineGenerator
from src.utils.main_utils import write_json_file


class CurrentStatsGenerator:
    """
    Computes the statistical distribution of recent production data.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        try:
            self.config = config
            logging.info("[STATS GENERATOR] Initialized successfully.")
        except Exception as e:
            logging.exception("[STATS GENERATOR] Initialization failed.")
            raise CustomerChurnException(e, sys)

    def generate_current_stats(self, live_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute the baseline report for the live data.
        
        Parameters
        ----------
        live_df : pd.DataFrame
            The freshly fetched live inference data.
            
        Returns
        -------
        Dict[str, Any]
            The statistical report matching the schema of baseline.json.
        """
        if live_df.empty:
            logging.warning("[STATS GENERATOR] Provided DataFrame is empty. Cannot generate stats.")
            return {}

        try:
            logging.info("[STATS GENERATOR] Starting statistics computation.")
            
            # WORKAROUND: The training BaselineGenerator strictly requires the TARGET_COLUMN.
            # Since live inference data does not have labels, we inject a dummy column 
            # to satisfy the validation check and allow feature generation to proceed.
            df_for_stats = live_df.copy()
            df_for_stats = live_df.drop(["customerID"], axis=1)
            if TARGET_COLUMN not in df_for_stats.columns:
                logging.debug(f"[STATS GENERATOR] Injecting dummy '{TARGET_COLUMN}' to satisfy generator constraints.")
                df_for_stats[TARGET_COLUMN] = "Inference_Unlabelled"

            generator = BaselineGenerator()
            current_stats = generator.generate_baseline_report(df_for_stats)

            # Persist the output locally as an artifact
            write_json_file(self.config.current_stats_file_path, current_stats)
            
            logging.info(
                f"[STATS GENERATOR] Successfully generated current stats | "
                f"Saved to: {self.config.current_stats_file_path}"
            )
            
            return current_stats

        except Exception as e:
            logging.exception("[STATS GENERATOR] Failed to generate current statistics.")
            raise CustomerChurnException(e, sys)