"""
Feature engineering component.

Performs the exact same feature engineering steps as in the notebook (Version 2),
including imputation, recency calculation, customer lifetime value derivation,
and creation of high-value customer labels.
"""

from pathlib import Path

import pandas as pd

from src.constants import paths as p
from src.logger.logger import get_logger
from src.utils.helpers import save_csv

logger = get_logger(__name__)


class FeatureEngineering:
    """
    Feature engineering pipeline (Version 2) matching the notebook logic.

    This class should be used **only for training** runs.  For inference,
    the ``PredictionPipeline`` loads the pre-fitted scaler and model.
    """

    def __init__(self, threshold_percentile: float = 0.75):
        self.threshold_percentile = threshold_percentile

    def engineer_features(self, ml_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps (V2 logic).

        Args:
            ml_df: Raw feature DataFrame from the ``ml_features_v2`` view.

        Returns:
            DataFrame with all engineered features and the target column.
        """
        df = ml_df.copy()
        logger.info(f"Input shape for feature engineering: {df.shape}")

        # ---- 1. Impute missing values ----
        df["avg_review_score"] = df["avg_review_score"].fillna(
            df["avg_review_score"].median()
        )
        df["avg_delivery_days"] = df["avg_delivery_days"].fillna(
            df["avg_delivery_days"].median()
        )

        # ---- 2. Convert datetime columns ----
        df["first_purchase"] = pd.to_datetime(df["first_purchase"])
        df["last_purchase"] = pd.to_datetime(df["last_purchase"])

        # ---- 3. Recency ----
        reference_date = df["last_purchase"].max()
        df["recency_days"] = (reference_date - df["last_purchase"]).dt.days

        # ---- 4. Customer age (tenure) ----
        df["customer_age_days"] = (
            df["last_purchase"] - df["first_purchase"]
        ).dt.days

        # ---- 5. Purchase frequency ----
        df["purchase_frequency"] = df["total_orders"] / (df["customer_age_days"] + 1)

        # ---- 6. Products per order ----
        df["products_per_order"] = (
            df["unique_products"] / df["total_orders"]
        ).round(2)

        # ---- 7. Freight ratio ----
        df["freight_ratio"] = df["total_freight"] / df["total_spent"]

        # ---- 8. Customer lifetime value (proxy) ----
        df["customer_lifetime_value"] = df["total_spent"]

        # ---- 9. High-review customer flag ----
        df["high_review_customer"] = (df["avg_review_score"] >= 4).astype(int)

        # ---- 10. High-value customer target (75th percentile) ----
        threshold = df["customer_lifetime_value"].quantile(self.threshold_percentile)
        df["high_value_customer"] = (
            df["customer_lifetime_value"] >= threshold
        ).astype(int)
        logger.info(
            f"High-value customer threshold: {threshold:.2f}  |  "
            f"Distribution:\n{df['high_value_customer'].value_counts().to_dict()}"
        )

        logger.info(f"Output shape after feature engineering: {df.shape}")
        return df

    def prepare_features_and_target(
        self, df: pd.DataFrame
    ):
        """
        Separate features (X) and target (y) as per notebook V2 logic.

        Drops: customer_unique_id, first_purchase, last_purchase,
        high_value_customer, total_orders.

        Args:
            df: DataFrame with all engineered features.

        Returns:
            Tuple of (X, y).
        """
        X = df.drop(
            columns=[
                "customer_unique_id",
                "first_purchase",
                "last_purchase",
                "high_value_customer",
                "total_orders",
            ]
        )
        y = df["high_value_customer"]
        logger.info(f"Features shape: {X.shape} | Target shape: {y.shape}")
        return X, y

    def save_ml_dataset(self, df: pd.DataFrame, path: Path = p.ML_FEATURE_DATASET_PATH) -> None:
        """Save the full engineered DataFrame to disk."""
        save_csv(df, path)

