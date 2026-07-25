"""
Configuration manager.

Provides a single entry point to obtain configuration objects
for each pipeline stage.
"""

from src.constants import paths as p
from src.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    DatabaseConfig,
    FeatureEngineeringConfig,
    ModelTrainingConfig,
    PredictionPipelineConfig,
    RandomizedSearchConfig,
)


class ConfigurationManager:
    """
    Factory for configuration dataclasses.

    Centralises the mapping between path constants and config entities.
    """

    @staticmethod
    def get_data_ingestion_config() -> DataIngestionConfig:
        return DataIngestionConfig(
            raw_data_dir=p.RAW_DATA_DIR,
            dataset_zip_path=p.DATASET_ZIP,
            raw_files=p.RAW_FILES,
        )

    @staticmethod
    def get_data_preprocessing_config() -> DataPreprocessingConfig:
        return DataPreprocessingConfig(
            processed_data_dir=p.PROCESSED_DATA_DIR,
            dataset_summary_path=p.DATASET_SUMMARY_PATH,
        )

    @staticmethod
    def get_database_config() -> DatabaseConfig:
        return DatabaseConfig(
            database_path=p.DATABASE_PATH,
            processed_data_dir=p.PROCESSED_DATA_DIR,
        )

    @staticmethod
    def get_feature_engineering_config() -> FeatureEngineeringConfig:
        return FeatureEngineeringConfig(
            ml_feature_dataset_path=p.ML_FEATURE_DATASET_PATH,
        )

    @staticmethod
    def get_model_training_config() -> ModelTrainingConfig:
        return ModelTrainingConfig(
            models_dir=p.MODELS_DIR,
            reports_dir=p.REPORTS_DIR,
            images_dir=p.ML_IMAGES_DIR,
            random_state=p.RANDOM_STATE,
        )

    @staticmethod
    def get_randomized_search_config() -> RandomizedSearchConfig:
        return RandomizedSearchConfig()

    @staticmethod
    def get_prediction_pipeline_config() -> PredictionPipelineConfig:
        # Feature columns used by the best model (X_v2 columns without target/id)
        feature_columns = [
            "total_spent",
            "avg_order_value",
            "avg_review_score",
            "unique_products",
            "avg_freight",
            "total_freight",
            "avg_product_price",
            "avg_installments",
            "credit_card_orders",
            "avg_delivery_days",
            "recency_days",
            "customer_age_days",
            "purchase_frequency",
            "products_per_order",
            "freight_ratio",
        ]
        return PredictionPipelineConfig(
            scaler_path=p.SCALER_PATH,
            final_model_path=p.FINAL_MODEL_PATH,
            feature_columns=feature_columns,
        )

