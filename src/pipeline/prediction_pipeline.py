"""
Prediction pipeline.

Loads the existing saved artifacts (scaler + final model) and
performs end-to-end inference on new customer data.

This is the primary entry point for Streamlit integration.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any

from src.constants import paths as p
from src.configuration.configuration import ConfigurationManager
from src.entity.config_entity import PredictionPipelineConfig
from src.exception.custom_exception import CustomException
from src.logger.logger import get_logger
from src.utils.helpers import load_model

logger = get_logger(__name__)


class PredictionPipeline:
    """
    Load artifacts and run inference on new customer feature vectors.

    Usage:
        pipeline = PredictionPipeline()
        result = pipeline.predict(features_df)
        probability = pipeline.predict_proba(features_df)
    """

    def __init__(self, config: PredictionPipelineConfig = None):
        if config is None:
            config = ConfigurationManager.get_prediction_pipeline_config()
        self.config = config

        self._scaler = None
        self._model = None
        self._feature_columns = config.feature_columns

    def _load_artifacts(self) -> None:
        """Lazy-load scaler and model from disk."""
        if self._scaler is None:
            logger.info("Loading scaler and model artifacts ...")
            self._scaler = load_model(self.config.scaler_path)
            self._model = load_model(self.config.final_model_path)
            logger.info("Artifacts loaded successfully.")

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the input DataFrame has the expected columns.

        Args:
            df: Input features.

        Returns:
            DataFrame with columns in the correct order.
        """
        missing = set(self._feature_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required features: {missing}. "
                f"Expected: {self._feature_columns}"
            )
        return df[self._feature_columns]

    def preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing used during training.

        This includes:
        - Missing value imputation (median for review_score, delivery_days)
        - Feature engineering (recency, frequency, ratios, etc.)
        - Scaling

        Args:
            features: Raw feature DataFrame from the user.

        Returns:
            Scaled feature array (as DataFrame for compatibility).
        """
        self._load_artifacts()
        df = features.copy()

        # ---- Impute missing values ----
        if "avg_review_score" in df.columns:
            df["avg_review_score"] = df["avg_review_score"].fillna(
                df["avg_review_score"].median()
            )
        if "avg_delivery_days" in df.columns:
            df["avg_delivery_days"] = df["avg_delivery_days"].fillna(
                df["avg_delivery_days"].median()
            )

        # ---- Feature engineering (if datetime columns are present) ----
        if "first_purchase" in df.columns and "last_purchase" in df.columns:
            df["first_purchase"] = pd.to_datetime(df["first_purchase"])
            df["last_purchase"] = pd.to_datetime(df["last_purchase"])
            reference_date = df["last_purchase"].max()
            df["recency_days"] = (reference_date - df["last_purchase"]).dt.days
            df["customer_age_days"] = (
                df["last_purchase"] - df["first_purchase"]
            ).dt.days
            df["purchase_frequency"] = df["total_orders"] / (
                df["customer_age_days"] + 1
            )

        if "unique_products" in df.columns and "total_orders" in df.columns:
            df["products_per_order"] = (
                df["unique_products"] / df["total_orders"]
            ).round(2)

        if "total_freight" in df.columns and "total_spent" in df.columns:
            df["freight_ratio"] = df["total_freight"] / df["total_spent"]

        # ---- Keep only the expected feature columns ----
        df = self._validate_features(df)

        # ---- Scale ----
        scaled = self._scaler.transform(df)
        return pd.DataFrame(scaled, columns=self._feature_columns)

    def predict(self, features: pd.DataFrame) -> int:
        """
        Predict class label (0 = non-high-value, 1 = high-value).

        Args:
            features: Raw feature DataFrame.

        Returns:
            Predicted label.
        """
        self._load_artifacts()
        processed = self.preprocess(features)
        pred = self._model.predict(processed)
        return int(pred[0])

    def predict_proba(self, features: pd.DataFrame) -> float:
        """
        Predict probability of being a high-value customer.

        Args:
            features: Raw feature DataFrame.

        Returns:
            Probability of class 1 (high-value).
        """
        self._load_artifacts()
        processed = self.preprocess(features)
        prob = self._model.predict_proba(processed)[0, 1]
        return float(prob)

    def predict_batch(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict labels and probabilities for a batch of customers.

        Args:
            features: Raw feature DataFrame.

        Returns:
            DataFrame with columns: predicted_label, probability.
        """
        self._load_artifacts()
        processed = self.preprocess(features)
        labels = self._model.predict(processed)
        probs = self._model.predict_proba(processed)[:, 1]
        return pd.DataFrame(
            {"predicted_label": labels, "probability": probs}
        )
