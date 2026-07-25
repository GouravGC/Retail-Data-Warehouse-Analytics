"""
Centralized path configuration for the entire project.

All artifact and data paths are defined here using pathlib.
No absolute or hardcoded paths should be used anywhere else.
"""

from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Artifacts Root
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Data Directories
# ---------------------------------------------------------------------------
RAW_DATA_DIR = ARTIFACTS_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_DIR = ARTIFACTS_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "retail_warehouse.db"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS_DIR = ARTIFACTS_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"
BEST_RANDOM_FOREST_PATH = MODELS_DIR / "best_random_forest.pkl"
LOGISTIC_REGRESSION_PATH = MODELS_DIR / "logistic_regression.pkl"
XGBOOST_PATH = MODELS_DIR / "xgboost.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DATASET_SUMMARY_PATH = REPORTS_DIR / "dataset_summary.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"

# ---------------------------------------------------------------------------
# Images / Plots
# ---------------------------------------------------------------------------
IMAGES_DIR = ARTIFACTS_DIR / "images"
EDA_IMAGES_DIR = IMAGES_DIR / "eda"
SQL_IMAGES_DIR = IMAGES_DIR / "sql"
ML_IMAGES_DIR = IMAGES_DIR / "ml"

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
SHAP_DIR = ARTIFACTS_DIR / "shap"

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------
LOGS_DIR = ARTIFACTS_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ---------------------------------------------------------------------------
# Dataset Files
# ---------------------------------------------------------------------------
DATASET_ZIP = PROJECT_ROOT / "Dataset Used" / "olist.zip"

# Processed CSV files
CUSTOMERS_CLEAN_PATH = PROCESSED_DATA_DIR / "customers_clean.csv"
ORDERS_CLEAN_PATH = PROCESSED_DATA_DIR / "orders_clean.csv"
ORDER_ITEMS_CLEAN_PATH = PROCESSED_DATA_DIR / "order_items_clean.csv"
PAYMENTS_CLEAN_PATH = PROCESSED_DATA_DIR / "payments_clean.csv"
PRODUCTS_CLEAN_PATH = PROCESSED_DATA_DIR / "products_clean.csv"
REVIEWS_CLEAN_PATH = PROCESSED_DATA_DIR / "reviews_clean.csv"
SELLERS_CLEAN_PATH = PROCESSED_DATA_DIR / "sellers_clean.csv"
GEOLOCATION_CLEAN_PATH = PROCESSED_DATA_DIR / "geolocation_clean.csv"
ML_FEATURE_DATASET_PATH = PROCESSED_DATA_DIR / "ml_feature_dataset.csv"

# ---------------------------------------------------------------------------
# Random State
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Raw CSV file names inside the zip
# ---------------------------------------------------------------------------
RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

