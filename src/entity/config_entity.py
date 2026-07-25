"""
Configuration dataclasses for pipeline stages.

These dataclasses define the configuration contracts consumed by
each component in the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataIngestionConfig:
    """Configuration for the data ingestion step."""

    raw_data_dir: Path
    dataset_zip_path: Path
    raw_files: dict


@dataclass
class DataPreprocessingConfig:
    """Configuration for preprocessing / cleaning."""

    processed_data_dir: Path
    dataset_summary_path: Path


@dataclass
class DatabaseConfig:
    """Configuration for SQLite database setup."""

    database_path: Path
    processed_data_dir: Path


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""

    ml_feature_dataset_path: Path
    threshold_percentile: float = 0.75


@dataclass
class ModelTrainingConfig:
    """Configuration for model training and hyperparameter tuning."""

    models_dir: Path
    reports_dir: Path
    images_dir: Path
    random_state: int = 42
    test_size: float = 0.20
    rf_n_estimators: int = 500
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    xgb_n_estimators: int = 400
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    lr_max_iter: int = 1000


@dataclass
class RandomizedSearchConfig:
    """Configuration for RandomizedSearchCV on Random Forest."""

    n_estimators: List[int] = field(default_factory=lambda: [200, 300, 500])
    max_depth: List[Optional[int]] = field(
        default_factory=lambda: [5, 10, 15, None]
    )
    min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 2, 4])
    n_iter: int = 10
    cv: int = 3
    scoring: str = "roc_auc"


@dataclass
class PredictionPipelineConfig:
    """Configuration for the prediction pipeline."""

    scaler_path: Path
    final_model_path: Path
    feature_columns: List[str] = field(default_factory=list)

