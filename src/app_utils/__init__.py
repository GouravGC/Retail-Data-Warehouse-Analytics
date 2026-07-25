"""
Application-specific utilities for the Streamlit frontend.

Provides feature metadata, input validation helpers,
and prediction history management via st.session_state.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import streamlit as st
import pandas as pd
import json


# =============================================================================
# Feature Metadata
# =============================================================================

FEATURE_METADATA = {
    "total_spent": {
        "label": "Total Spent",
        "icon": "💰",
        "description": "Total monetary amount spent by the customer across all orders.",
        "range": "0 – 10,000+",
        "typical": "~150",
        "importance": "Most important feature. Higher spending → more likely high-value.",
    },
    "avg_order_value": {
        "label": "Avg Order Value",
        "icon": "🛒",
        "description": "Average monetary value of each order placed by the customer.",
        "range": "0 – 5,000+",
        "typical": "~100–200",
        "importance": "Second most important. Customers with higher average spend per order are more valuable.",
    },
    "avg_review_score": {
        "label": "Avg Review Score",
        "icon": "⭐",
        "description": "Average customer review score across all orders (1–5).",
        "range": "1.0 – 5.0",
        "typical": "4.0–4.5",
        "importance": "Satisfied customers tend to spend more and are more likely to be high-value.",
    },
    "unique_products": {
        "label": "Unique Products",
        "icon": "📦",
        "description": "Number of distinct products the customer has purchased.",
        "range": "1 – 100+",
        "typical": "3–10",
        "importance": "Customers buying diverse products are more engaged and less likely to churn.",
    },
    "avg_freight": {
        "label": "Avg Freight Cost",
        "icon": "🚚",
        "description": "Average shipping cost paid per order.",
        "range": "0 – 500+",
        "typical": "15–30",
        "importance": "Higher freight costs may indicate rural areas or premium shipping preferences.",
    },
    "total_freight": {
        "label": "Total Freight",
        "icon": "📬",
        "description": "Total shipping cost paid across all orders.",
        "range": "0 – 2,000+",
        "typical": "50–200",
        "importance": "Combined with total_spent, helps calculate freight_ratio for value assessment.",
    },
    "avg_product_price": {
        "label": "Avg Product Price",
        "icon": "🏷️",
        "description": "Average price of products purchased.",
        "range": "0 – 3,000+",
        "typical": "50–200",
        "importance": "Third most important. Customers buying premium products tend to have higher lifetime value.",
    },
    "avg_installments": {
        "label": "Avg Installments",
        "icon": "📅",
        "description": "Average number of payment installments used.",
        "range": "1 – 24",
        "typical": "3–8",
        "importance": "More installments may indicate higher purchase values or payment preference patterns.",
    },
    "credit_card_orders": {
        "label": "Credit Card Orders",
        "icon": "💳",
        "description": "Number of orders paid using credit card.",
        "range": "0 – 50+",
        "typical": "2–5",
        "importance": "Credit card usage correlates with higher spending and purchasing convenience.",
    },
    "avg_delivery_days": {
        "label": "Avg Delivery Days",
        "icon": "📬",
        "description": "Average number of days between purchase and delivery.",
        "range": "1 – 100+",
        "typical": "8–15",
        "importance": "Faster deliveries improve satisfaction and may encourage repeat purchases.",
    },
    "recency_days": {
        "label": "Recency (Days)",
        "icon": "🕐",
        "description": "Days since the customer's last purchase.",
        "range": "0 – 365+",
        "typical": "1–60",
        "importance": "Low recency = recently active. Key RFM metric for customer engagement scoring.",
    },
    "customer_age_days": {
        "label": "Customer Age (Days)",
        "icon": "📆",
        "description": "Number of days between first and last purchase (customer tenure).",
        "range": "0 – 700+",
        "typical": "30–365",
        "importance": "Longer tenure customers are usually more loyal and have higher lifetime value.",
    },
    "purchase_frequency": {
        "label": "Purchase Frequency",
        "icon": "🔄",
        "description": "Orders placed per day (total_orders / customer_age_days).",
        "range": "0.0 – 1.0+",
        "typical": "0.01–0.10",
        "importance": "High frequency indicates strong engagement and habitual purchasing behavior.",
    },
    "products_per_order": {
        "label": "Products per Order",
        "icon": "📋",
        "description": "Average number of distinct products per order.",
        "range": "1.0 – 10.0+",
        "typical": "1.0–3.0",
        "importance": "Buying multiple items per order increases basket size and total spending.",
    },
    "freight_ratio": {
        "label": "Freight Ratio",
        "icon": "⚖️",
        "description": "Proportion of total_freight to total_spent (freight / total).",
        "range": "0.0 – 1.0+",
        "typical": "0.05–0.30",
        "importance": "Fourth most important. Low ratio = efficient shipping relative to spend; high ratio may indicate friction.",
    },
}


# =============================================================================
# Input Validation
# =============================================================================

class ValidationError(Exception):
    """Raised when user input fails validation."""
    pass


def validate_inputs(input_data: Dict[str, float]) -> List[str]:
    """
    Validate user inputs before prediction.

    Args:
        input_data: Dictionary of feature name → value.

    Returns:
        List of validation error messages (empty = all valid).
    """
    errors = []

    # Define validation rules
    rules = {
        "total_spent": (0, 1_000_000, "Total Spent cannot be negative or unrealistically large."),
        "avg_order_value": (0, 500_000, "Avg Order Value cannot be negative or unrealistically large."),
        "avg_review_score": (1.0, 5.0, "Avg Review Score must be between 1.0 and 5.0."),
        "unique_products": (0, 10_000, "Unique Products must be a non-negative integer."),
        "avg_freight": (0, 100_000, "Avg Freight Cost cannot be negative."),
        "total_freight": (0, 1_000_000, "Total Freight cannot be negative."),
        "avg_product_price": (0, 500_000, "Avg Product Price cannot be negative."),
        "avg_installments": (0, 24, "Avg Installments must be between 0 and 24."),
        "credit_card_orders": (0, 10_000, "Credit Card Orders cannot be negative."),
        "avg_delivery_days": (0, 1_000, "Avg Delivery Days must be between 0 and 1,000."),
        "recency_days": (0, 10_000, "Recency Days cannot be negative."),
        "customer_age_days": (0, 10_000, "Customer Age Days cannot be negative."),
        "purchase_frequency": (0, 1000, "Purchase Frequency cannot be negative."),
        "products_per_order": (0, 10_000, "Products per Order cannot be negative."),
        "freight_ratio": (0, 1000, "Freight Ratio cannot be negative."),
    }

    for field, (min_val, max_val, message) in rules.items():
        value = input_data.get(field)
        if value is None:
            errors.append(f"{field}: Value is required.")
        elif value < min_val or value > max_val:
            errors.append(f"{field}: {message} (got {value})")

    return errors


# =============================================================================
# Prediction History
# =============================================================================

def init_prediction_history() -> None:
    """Initialize prediction history in session state if not present."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def add_to_history(
    input_data: Dict[str, float],
    prediction: int,
    probability: float,
) -> None:
    """
    Add a prediction record to the session history.

    Args:
        input_data: The feature values used for prediction.
        prediction: Predicted class label (0 or 1).
        probability: Probability of being high-value (class 1).
    """
    init_prediction_history()
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **input_data,
        "prediction": "High-Value ✅" if prediction == 1 else "Standard",
        "probability": f"{probability:.2%}",
        "probability_raw": round(probability, 4),
    }
    st.session_state.prediction_history.append(record)


def get_history_df() -> pd.DataFrame:
    """Return prediction history as a DataFrame (empty if none)."""
    init_prediction_history()
    if not st.session_state.prediction_history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.prediction_history)


def clear_history() -> None:
    """Clear all prediction history."""
    st.session_state.prediction_history = []
