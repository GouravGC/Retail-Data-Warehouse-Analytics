"""
Exploratory Data Analysis component.

Generates all visualisations used in the notebook for EDA
and saves them to the artifacts/images/eda directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.constants import paths as p
from src.logger.logger import get_logger

logger = get_logger(__name__)

# Use a consistent style
plt.style.use("default")
sns.set_style("whitegrid")


class EDAAnalysis:
    """
    Generate EDA plots and insights for the Olist datasets.
    """

    def __init__(self, output_dir: Path = p.EDA_IMAGES_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_plot(self, filename: str) -> None:
        """Save the current matplotlib figure to disk."""
        path = self.output_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot: {path}")
        plt.close()

    def monthly_order_trend(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Monthly order count trend line plot."""
        orders = orders.copy()
        orders["YearMonth"] = (
            orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
        )
        monthly = (
            orders.groupby("YearMonth").size().reset_index(name="Total Orders")
        )

        plt.figure(figsize=(14, 6))
        plt.plot(monthly["YearMonth"], monthly["Total Orders"], marker="o", linewidth=2)
        plt.xticks(rotation=90)
        plt.title("Monthly Orders Trend")
        plt.xlabel("Month")
        plt.ylabel("Number of Orders")
        plt.tight_layout()
        self._save_plot("monthly_orders.png")
        return monthly

    def order_status_distribution(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Bar plot of order status counts."""
        status = (
            orders["order_status"]
            .value_counts()
            .reset_index()
        )
        status.columns = ["Status", "Count"]

        plt.figure(figsize=(8, 5))
        sns.barplot(data=status, x="Status", y="Count")
        plt.title("Order Status Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_plot("order_status_distribution.png")
        return status

    def customers_by_state(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Bar plot of customer count per state."""
        state_counts = (
            customers["customer_state"]
            .value_counts()
            .reset_index()
        )
        state_counts.columns = ["State", "Customers"]

        plt.figure(figsize=(12, 6))
        sns.barplot(data=state_counts, x="State", y="Customers")
        plt.xticks(rotation=90)
        plt.title("Customers by State")
        plt.tight_layout()
        self._save_plot("customers_by_state.png")
        return state_counts

    def top_customer_cities(self, customers: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Bar plot of top N customer cities."""
        top_cities = (
            customers["customer_city"]
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        top_cities.columns = ["City", "Customers"]

        plt.figure(figsize=(14, 7))
        sns.barplot(data=top_cities, x="City", y="Customers")
        plt.xticks(rotation=90)
        plt.title(f"Top {top_n} Customer Cities")
        plt.tight_layout()
        self._save_plot("top_customer_cities.png")
        return top_cities

    def payment_method_distribution(self, payments: pd.DataFrame) -> pd.DataFrame:
        """Bar plot of payment type counts."""
        summary = (
            payments["payment_type"]
            .value_counts()
            .reset_index()
        )
        summary.columns = ["Payment Type", "Transactions"]

        plt.figure(figsize=(8, 5))
        sns.barplot(data=summary, x="Payment Type", y="Transactions")
        plt.title("Payment Method Distribution")
        plt.xticks(rotation=30)
        plt.tight_layout()
        self._save_plot("payment_method_distribution.png")
        return summary

    def payment_installments_distribution(self, payments: pd.DataFrame) -> pd.DataFrame:
        """Bar plot of payment installment counts."""
        installments = (
            payments["payment_installments"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        installments.columns = ["Installments", "Transactions"]

        plt.figure(figsize=(12, 5))
        sns.barplot(data=installments, x="Installments", y="Transactions")
        plt.title("Payment Installments Distribution")
        plt.tight_layout()
        self._save_plot("payment_installments_distribution.png")
        return installments

    def review_score_distribution(self, reviews: pd.DataFrame) -> pd.DataFrame:
        """Bar plot of review score counts."""
        scores = (
            reviews["review_score"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        scores.columns = ["Review Score", "Count"]

        plt.figure(figsize=(8, 5))
        sns.barplot(data=scores, x="Review Score", y="Count")
        plt.title("Customer Review Score Distribution")
        plt.tight_layout()
        self._save_plot("review_score_distribution.png")
        return scores

    def top_product_categories(self, products: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Horizontal bar plot of top product categories."""
        cat_col = "product_category_name_english"
        top = (
            products[cat_col]
            .fillna("Unknown")
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        top.columns = ["Category", "Products"]

        plt.figure(figsize=(14, 7))
        sns.barplot(data=top, x="Products", y="Category")
        plt.title(f"Top {top_n} Product Categories")
        plt.tight_layout()
        self._save_plot("top_product_categories.png")
        return top

    def seller_distribution(self, sellers: pd.DataFrame) -> pd.DataFrame:
        """Bar plot of seller count per state."""
        state_counts = (
            sellers["seller_state"]
            .value_counts()
            .reset_index()
        )
        state_counts.columns = ["State", "Sellers"]

        plt.figure(figsize=(10, 6))
        sns.barplot(data=state_counts, x="State", y="Sellers")
        plt.title("Seller Distribution by State")
        plt.tight_layout()
        self._save_plot("seller_distribution.png")
        return state_counts

    def delivery_time_distribution(self, orders: pd.DataFrame) -> pd.Series:
        """Histogram of delivery times in days."""
        delivery = orders.copy()
        delivery["order_delivered_customer_date"] = pd.to_datetime(
            delivery["order_delivered_customer_date"]
        )
        delivery["order_purchase_timestamp"] = pd.to_datetime(
            delivery["order_purchase_timestamp"]
        )
        delivery["Delivery Days"] = (
            delivery["order_delivered_customer_date"]
            - delivery["order_purchase_timestamp"]
        ).dt.days
        days = delivery["Delivery Days"].dropna()

        plt.figure(figsize=(10, 5))
        plt.hist(days, bins=40)
        plt.title("Delivery Time Distribution")
        plt.xlabel("Days")
        plt.ylabel("Orders")
        plt.tight_layout()
        self._save_plot("delivery_time_distribution.png")
        return days

    def correlation_heatmap(self, products: pd.DataFrame) -> None:
        """Correlation heatmap of numeric product features."""
        numeric = products.select_dtypes(include=np.number)
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm")
        plt.title("Product Features Correlation")
        plt.tight_layout()
        self._save_plot("correlation_heatmap.png")

