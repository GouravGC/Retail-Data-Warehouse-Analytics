"""
SQL Analytics component.

Creates an SQLite database from the cleaned datasets, executes
analytical SQL queries, and creates rich feature views for ML.
"""

import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd

from src.constants import paths as p
from src.logger.logger import get_logger
from src.utils.helpers import load_csv

logger = get_logger(__name__)


class SQLAnalytics:
    """
    Set up the SQLite database and run analytical queries.

    The cleaned CSVs are loaded into an SQLite database for
    advanced analysis and feature engineering.
    """

    def __init__(
        self,
        database_path: Path = p.DATABASE_PATH,
        processed_dir: Path = p.PROCESSED_DATA_DIR,
    ):
        self.database_path = database_path
        self.processed_dir = processed_dir
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection."""
        return sqlite3.connect(str(self.database_path))

    def load_data_to_db(self) -> None:
        """
        Load cleaned CSV files into SQLite tables.

        Tables created: customers, orders, order_items, payments,
        products, reviews, sellers, geolocation, translation.
        """
        logger.info("Loading cleaned data into SQLite database ...")
        conn = self._get_connection()

        table_files = [
            ("customers", "customers_clean.csv"),
            ("orders", "orders_clean.csv"),
            ("order_items", "order_items_clean.csv"),
            ("payments", "payments_clean.csv"),
            ("products", "products_clean.csv"),
            ("reviews", "reviews_clean.csv"),
            ("sellers", "sellers_clean.csv"),
            ("geolocation", "geolocation_clean.csv"),
        ]

        for table, filename in table_files:
            df = load_csv(self.processed_dir / filename)
            df.to_sql(table, conn, if_exists="replace", index=False)
            row_count = len(df)
            logger.info(f"  → Loaded {row_count} rows into '{table}'")

        # Load translation (from raw if not processed separately)
        trans_path = self.processed_dir.parent / "raw" / "product_category_name_translation.csv"
        if trans_path.exists():
            trans = load_csv(trans_path)
            trans.to_sql("translation", conn, if_exists="replace", index=False)
            logger.info(f"  → Loaded {len(trans)} rows into 'translation'")

        conn.commit()
        conn.close()
        logger.info("Database load complete.")

    def create_ml_features_view(self) -> None:
        """
        Create the ``ml_features`` view with engineered features.

        This view aggregates customer-level metrics from multiple tables
        and is the primary source for ML feature engineering.
        """
        query = """
        CREATE VIEW IF NOT EXISTS ml_features AS
        SELECT
            c.customer_unique_id,
            COUNT(DISTINCT o.order_id) AS total_orders,
            ROUND(SUM(p.payment_value), 2) AS total_spent,
            ROUND(AVG(p.payment_value), 2) AS avg_order_value,
            AVG(r.review_score) AS avg_review_score,
            MAX(o.order_purchase_timestamp) AS last_purchase,
            COUNT(DISTINCT oi.product_id) AS unique_products
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN payments p ON o.order_id = p.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        GROUP BY c.customer_unique_id
        """
        conn = self._get_connection()
        conn.execute("DROP VIEW IF EXISTS ml_features")
        conn.execute(query)
        conn.commit()
        conn.close()
        logger.info("Created 'ml_features' view.")

    def create_ml_features_v2_view(self) -> None:
        """
        Create the advanced ``ml_features_v2`` view with extended features.

        Includes freight, installment, delivery, and product price aggregations.
        """
        query = """
        CREATE VIEW IF NOT EXISTS ml_features_v2 AS
        SELECT
            c.customer_unique_id,
            COUNT(DISTINCT o.order_id) AS total_orders,
            SUM(p.payment_value) AS total_spent,
            AVG(p.payment_value) AS avg_order_value,
            AVG(r.review_score) AS avg_review_score,
            COUNT(DISTINCT oi.product_id) AS unique_products,
            AVG(oi.freight_value) AS avg_freight,
            SUM(oi.freight_value) AS total_freight,
            AVG(oi.price) AS avg_product_price,
            AVG(p.payment_installments) AS avg_installments,
            SUM(CASE WHEN p.payment_type = 'credit_card' THEN 1 ELSE 0 END)
                AS credit_card_orders,
            MIN(o.order_purchase_timestamp) AS first_purchase,
            MAX(o.order_purchase_timestamp) AS last_purchase,
            AVG(
                julianday(o.order_delivered_customer_date) -
                julianday(o.order_purchase_timestamp)
            ) AS avg_delivery_days
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN payments p ON o.order_id = p.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        GROUP BY c.customer_unique_id
        """
        conn = self._get_connection()
        conn.execute("DROP VIEW IF EXISTS ml_features_v2")
        conn.execute(query)
        conn.commit()
        conn.close()
        logger.info("Created 'ml_features_v2' view.")

    def load_ml_features(self, version: int = 2) -> pd.DataFrame:
        """
        Load the ML features view into a pandas DataFrame.

        Args:
            version: 1 for basic view, 2 for advanced view.

        Returns:
            DataFrame with customer-level features.
        """
        view = "ml_features_v2" if version == 2 else "ml_features"
        conn = self._get_connection()
        df = pd.read_sql(f"SELECT * FROM {view}", conn)
        conn.close()
        logger.info(f"Loaded '{view}': {df.shape}")
        return df

