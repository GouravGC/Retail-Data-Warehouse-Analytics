"""
Data preprocessing component.

Performs initial data cleaning, type conversions, missing-value analysis,
and saves cleaned datasets for downstream use.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from pathlib import Path

from src.exception.custom_exception import CustomException
from src.logger.logger import get_logger
from src.constants import paths as p
from src.utils.helpers import save_csv

logger = get_logger(__name__)


class DataPreprocessing:
    """
    Clean and preprocess Olist raw datasets.
    """

    DATE_COLUMNS_ORDERS = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    DATE_COLUMNS_REVIEWS = [
        "review_creation_date",
        "review_answer_timestamp",
    ]
    DATE_COLUMNS_ORDER_ITEMS = ["shipping_limit_date"]

    def __init__(self, processed_dir: Path = p.PROCESSED_DATA_DIR):
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_dataset_summary(
        datasets: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Create a summary report of all datasets (rows, cols, missing, duplicates, memory).

        Args:
            datasets: Dictionary of dataset name → DataFrame.

        Returns:
            Summary DataFrame.
        """
        logger.info("Generating dataset summary report ...")
        rows = []
        for name, df in datasets.items():
            rows.append(
                {
                    "Dataset": name,
                    "Rows": df.shape[0],
                    "Columns": df.shape[1],
                    "Missing Values": int(df.isnull().sum().sum()),
                    "Duplicate Rows": int(df.duplicated().sum()),
                    "Memory (MB)": round(
                        df.memory_usage(deep=True).sum() / (1024**2), 2
                    ),
                }
            )
        summary = pd.DataFrame(rows)
        return summary

    @staticmethod
    def missing_value_analysis(
        datasets: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate missing value counts and percentages per column for each dataset.

        Args:
            datasets: Dictionary of dataset name → DataFrame.

        Returns:
            Dictionary of dataset name → missing-value DataFrame.
        """
        result = {}
        for name, df in datasets.items():
            missing = df.isnull().sum().to_frame("Missing Values")
            missing["Percentage"] = (missing["Missing Values"] / len(df)) * 100
            missing = missing.sort_values("Missing Values", ascending=False)
            result[name] = missing
        return result

    def convert_date_columns(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Convert date string columns to datetime objects for Orders, Reviews, Order Items.

        Args:
            datasets: Raw datasets dictionary.

        Returns:
            Datasets with date columns converted.
        """
        logger.info("Converting date columns to datetime ...")

        orders = datasets["Orders"].copy()
        for col in self.DATE_COLUMNS_ORDERS:
            orders[col] = pd.to_datetime(orders[col])

        reviews = datasets["Reviews"].copy()
        for col in self.DATE_COLUMNS_REVIEWS:
            reviews[col] = pd.to_datetime(reviews[col])

        order_items = datasets["Order Items"].copy()
        for col in self.DATE_COLUMNS_ORDER_ITEMS:
            order_items[col] = pd.to_datetime(order_items[col])

        datasets["Orders"] = orders
        datasets["Reviews"] = reviews
        datasets["Order Items"] = order_items

        logger.info("Date columns converted successfully.")
        return datasets

    def merge_translation(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Merge product category translation into Products dataset.

        Args:
            datasets: Datasets dictionary.

        Returns:
            Updated datasets with merged translation.
        """
        products = datasets["Products"].copy()
        translation = datasets["Translation"]
        products = products.merge(translation, on="product_category_name", how="left")
        datasets["Products"] = products
        logger.info(f"Products shape after translation merge: {products.shape}")
        return datasets

    def save_cleaned_datasets(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Save all cleaned datasets to the processed directory.

        Args:
            datasets: Dictionary of cleaned DataFrames.
        """
        mapping = {
            "Customers": "customers_clean.csv",
            "Orders": "orders_clean.csv",
            "Order Items": "order_items_clean.csv",
            "Payments": "payments_clean.csv",
            "Products": "products_clean.csv",
            "Reviews": "reviews_clean.csv",
            "Sellers": "sellers_clean.csv",
            "Geolocation": "geolocation_clean.csv",
        }
        for name, filename in mapping.items():
            save_csv(datasets[name], self.processed_dir / filename)
        logger.info("All cleaned datasets saved.")

