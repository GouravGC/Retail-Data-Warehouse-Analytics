"""
Utility / helper functions used across the project.
"""

import zipfile
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any

from src.exception.custom_exception import CustomException
from src.logger.logger import get_logger

logger = get_logger(__name__)


def create_directories(dirs: list) -> None:
    """
    Create a list of directories if they don't already exist.

    Args:
        dirs: List of Path objects or directory paths.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created {len(dirs)} director(ies)")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a zip file to a target directory.

    Args:
        zip_path: Path to the .zip file.
        extract_to: Destination directory.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        logger.info(f"Extracted '{zip_path.name}' to '{extract_to}'")
    except FileNotFoundError:
        logger.error(f"Zip file not found: {zip_path}")
        raise
    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {zip_path}")
        raise


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file.
        **kwargs: Additional arguments passed to ``pd.read_csv``.

    Returns:
        Loaded DataFrame.
    """
    logger.debug(f"Loading CSV: {path}")
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """
    Save a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Destination path.
        **kwargs: Additional arguments passed to ``df.to_csv``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info(f"Saved CSV: {path}")


def save_model(obj: Any, path: Path) -> None:
    """
    Serialise an object with joblib.

    Args:
        obj: Python object (model, scaler, etc.).
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved model artifact: {path}")


def load_model(path: Path) -> Any:
    """
    Deserialise a joblib file.

    Args:
        path: Path to the .pkl file.

    Returns:
        Deserialised object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    logger.info(f"Loaded model artifact: {path}")
    return joblib.load(path)


def load_all_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all Olist CSV files from a directory.

    Args:
        data_dir: Directory containing the CSV files.

    Returns:
        Dictionary mapping dataset names to DataFrames.
    """
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
    datasets = {}
    for name, filename in file_mapping.items():
        path = data_dir / filename
        datasets[name] = load_csv(path)
        logger.info(f"Loaded {name}: {datasets[name].shape}")
    return datasets

