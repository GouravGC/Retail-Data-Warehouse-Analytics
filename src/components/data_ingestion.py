"""
Data ingestion component.

Handles extraction of the raw dataset zip file and loading
of all Olist CSV files into pandas DataFrames.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from src.exception.custom_exception import CustomException
from src.logger.logger import get_logger
from src.constants import paths as p
from src.utils.helpers import extract_zip, load_csv

logger = get_logger(__name__)


class DataIngestion:
    """
    Ingests raw Olist data from the packaged zip file.
    """

    def __init__(self, raw_data_dir: Path = p.RAW_DATA_DIR):
        self.raw_data_dir = raw_data_dir

    def extract_raw_data(self) -> None:
        """
        Extract the Olist zip archive into the raw data directory.
        """
        logger.info("Extracting raw data from zip archive ...")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(p.DATASET_ZIP, self.raw_data_dir)

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all Olist CSV files from the raw data directory.

        Returns:
            Dictionary mapping dataset names to DataFrames.
        """
        logger.info("Loading all Olist datasets ...")
        datasets = {}
        file_mapping = {
            "Customers": "olist_customers_dataset.csv",
            "Orders": "olist_orders_dataset.csv",
            "Order Items": "olist_order_items_dataset.csv",
            "Payments": "olist_order_payments_dataset.csv",
            "Products": "olist_products_dataset.csv",
            "Reviews": "olist_order_reviews_dataset.csv",
            "Sellers": "olist_sellers_dataset.csv",
            "Geolocation": "olist_geolocation_dataset.csv",
            "Translation": "product_category_name_translation.csv",
        }
        for name, filename in file_mapping.items():
            file_path = self.raw_data_dir / filename
            datasets[name] = load_csv(file_path)
            logger.info(f"  → {name}: {datasets[name].shape}")
        return datasets

