"""
Core Prediction Engine for the Inference Service.

Responsibilities:
- Coordinate with the Model Manager to execute the inference pipeline.
- Apply the business decision threshold to raw probabilities.
- Map the resulting arrays back to customer IDs.
- Serialize the outputs into strict Pydantic Response models.
"""

import sys
import uuid
from typing import List

import pandas as pd

from src.constants.pipeline_constants import MODEL_TRAINING_DECISION_THRESHOLD
from src.exception import CustomerChurnException
from src.inference.model_manager import ModelManager
from src.inference.schemas import PredictionResult
from src.logging import logging


class PredictionEngine:
    """
    Production-grade inference engine. 
    Decouples the mathematical model execution from the API response formatting.
    """

    def __init__(self, threshold: float = MODEL_TRAINING_DECISION_THRESHOLD) -> None:
        try:
            self.model_manager = ModelManager()
            self.threshold = threshold
            
            logging.info(
                f"[PREDICTION ENGINE] Initialized | Decision Threshold: {self.threshold}"
            )
            
        except Exception as e:
            logging.exception("[PREDICTION ENGINE] Initialization failed.")
            raise CustomerChurnException(e, sys)

    def execute_batch_prediction(self, valid_df: pd.DataFrame) -> List[PredictionResult]:
        """
        Executes the prediction pipeline on a validated DataFrame.
        
        Args:
            valid_df: Pristine pandas DataFrame containing user features.
            
        Returns:
            List of Pydantic PredictionResult objects.
        """
        if valid_df.empty:
            logging.info("[PREDICTION ENGINE] Received empty DataFrame. Skipping inference.")
            return []

        try:
            logging.info(f"[PREDICTION ENGINE] Executing inference on {len(valid_df)} rows.")

            # 1. Extract Customer IDs for tracking
            # If customerID is missing for some reason, generate a UUID to trace the row
            if "customerID" in valid_df.columns:
                customer_ids = valid_df["customerID"].astype(str).tolist()
            else:
                logging.warning("[PREDICTION ENGINE] 'customerID' missing. Generating UUIDs.")
                customer_ids = [str(uuid.uuid4()) for _ in range(len(valid_df))]

            # 2. Generate Probabilities
            # The ModelManager's champion_pipeline handles the data transformation
            # via ColumnTransformer(remainder='drop'), so it safely ignores 'customerID'
            probabilities_array = self.model_manager.predict_proba(valid_df)

            # predict_proba returns an array of shape (n_samples, n_classes)
            # Index 1 is the probability for the positive class (Churn = Yes)
            churn_probabilities = probabilities_array[:, 1]

            # 3. Apply Decision Threshold and Map to Pydantic Models
            prediction_results: List[PredictionResult] = []

            for cust_id, prob in zip(customer_ids, churn_probabilities):
                
                # Apply business logic
                decision = "Yes" if prob >= self.threshold else "No"
                
                # Construct strict data contract
                result = PredictionResult(
                    customerID=cust_id,
                    churn_probability=round(float(prob), 4),
                    churn_decision=decision
                )
                prediction_results.append(result)

            logging.info("[PREDICTION ENGINE] Inference execution completed successfully.")
            
            return prediction_results

        except Exception as e:
            logging.exception("[PREDICTION ENGINE] Inference execution failed.")
            raise CustomerChurnException(e, sys)