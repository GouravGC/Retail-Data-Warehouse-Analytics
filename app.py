"""
Streamlit application entry point.

Provides an interactive dashboard for:
- Model prediction (single / batch)
- Model explainability insights
- EDA visualisations
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from src.constants import paths as p
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Retail Data Warehouse Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏪 Retail Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 EDA Visualizations",
        "🤖 Prediction",
        "📈 Model Insights",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This application uses a **Tuned Random Forest** model to predict "
    "high-value customers based on their purchase behaviour."
)


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
def home_page():
    st.title("Retail Data Warehouse Analytics")
    st.markdown(
        """
        ### End-to-End SQL + Machine Learning Project

        This project analyses the **Olist Brazilian E-Commerce** dataset and builds
        a machine learning model to identify **high-value customers**.

        **Pipeline stages:**
        1. Data Ingestion & Cleaning
        2. Exploratory Data Analysis (Python + SQL)
        3. Feature Engineering (Version 2)
        4. Model Training (Logistic Regression, Random Forest, XGBoost)
        5. Hyperparameter Tuning (RandomizedSearchCV)
        6. Model Evaluation & Selection
        7. SHAP Explainability

        **Best Model:** Tuned Random Forest Classifier  
        **Saved Artifacts:** `final_model.pkl`, `scaler.pkl`, `best_random_forest.pkl`
        """
    )

    # Show dataset summary
    summary_path = p.DATASET_SUMMARY_PATH
    if summary_path.exists():
        st.subheader("Dataset Summary")
        summary_df = pd.read_csv(summary_path)
        st.dataframe(summary_df, use_container_width=True)


# ---------------------------------------------------------------------------
# EDA Visualizations page
# ---------------------------------------------------------------------------
def eda_page():
    st.title("Exploratory Data Analysis")
    st.markdown("Visual insights generated from the Olist dataset.")

    eda_images = sorted(p.EDA_IMAGES_DIR.glob("*.png"))
    if not eda_images:
        st.warning("No EDA visualisations found. Run the training pipeline first.")
        return

    cols = st.columns(2)
    for i, img_path in enumerate(eda_images):
        with cols[i % 2]:
            st.image(str(img_path), caption=img_path.stem.replace("_", " ").title())
            st.markdown("---")


# ---------------------------------------------------------------------------
# Prediction page
# ---------------------------------------------------------------------------
def prediction_page():
    st.title("Customer Value Prediction")
    st.markdown(
        "Enter customer feature values below to predict whether they are a "
        "**high-value customer** (top 25% by lifetime value)."
    )

    # Load prediction pipeline
    try:
        pipeline = PredictionPipeline()
    except Exception as e:
        st.error(f"Failed to load prediction pipeline: {e}")
        st.info("Ensure the model artifacts exist in `artifacts/models/`.")
        return

    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            total_spent = st.number_input(
                "Total Spent", min_value=0.0, value=500.0, step=10.0
            )
            avg_order_value = st.number_input(
                "Avg Order Value", min_value=0.0, value=150.0, step=5.0
            )
            avg_review_score = st.slider(
                "Avg Review Score", 1.0, 5.0, 4.0, 0.5
            )
            unique_products = st.number_input(
                "Unique Products", min_value=1, value=5, step=1
            )
            avg_freight = st.number_input(
                "Avg Freight Cost", min_value=0.0, value=20.0, step=1.0
            )

        with col2:
            total_freight = st.number_input(
                "Total Freight", min_value=0.0, value=100.0, step=5.0
            )
            avg_product_price = st.number_input(
                "Avg Product Price", min_value=0.0, value=100.0, step=5.0
            )
            avg_installments = st.number_input(
                "Avg Installments", min_value=1.0, value=4.0, step=0.5
            )
            credit_card_orders = st.number_input(
                "Credit Card Orders", min_value=0, value=3, step=1
            )
            avg_delivery_days = st.number_input(
                "Avg Delivery Days", min_value=1.0, value=12.0, step=0.5
            )

        with col3:
            recency_days = st.number_input(
                "Recency (Days since last purchase)",
                min_value=0, value=30, step=1,
            )
            customer_age_days = st.number_input(
                "Customer Age (Days)", min_value=0, value=365, step=1
            )
            purchase_frequency = st.number_input(
                "Purchase Frequency", min_value=0.0, value=0.05, step=0.01,
                format="%.4f",
            )
            products_per_order = st.number_input(
                "Products per Order", min_value=0.0, value=1.5, step=0.1
            )
            freight_ratio = st.number_input(
                "Freight Ratio", min_value=0.0, value=0.2, step=0.01,
                format="%.3f",
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build feature DataFrame
        input_data = {
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "avg_review_score": avg_review_score,
            "unique_products": unique_products,
            "avg_freight": avg_freight,
            "total_freight": total_freight,
            "avg_product_price": avg_product_price,
            "avg_installments": avg_installments,
            "credit_card_orders": credit_card_orders,
            "avg_delivery_days": avg_delivery_days,
            "recency_days": recency_days,
            "customer_age_days": customer_age_days,
            "purchase_frequency": purchase_frequency,
            "products_per_order": products_per_order,
            "freight_ratio": freight_ratio,
        }
        features_df = pd.DataFrame([input_data])

        try:
            label = pipeline.predict(features_df)
            prob = pipeline.predict_proba(features_df)

            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if label == 1:
                    st.success(
                        f"**High-Value Customer** ✅\n\n"
                        f"Probability: **{prob:.2%}**"
                    )
                else:
                    st.info(
                        f"**Standard Customer** ℹ️\n\n"
                        f"Probability: **{prob:.2%}**"
                    )
            with col_b:
                # Show a simple gauge-like bar
                st.markdown("**Confidence Level**")
                st.progress(prob)
                st.caption(f"{prob:.1%} probability of being high-value")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.exception("Prediction error")


# ---------------------------------------------------------------------------
# Model Insights page
# ---------------------------------------------------------------------------
def model_insights_page():
    st.title("Model Insights & Explainability")

    # Feature importance
    imp_path = p.FEATURE_IMPORTANCE_PATH
    if imp_path.exists():
        st.subheader("Feature Importance (Tuned Random Forest)")
        imp_df = pd.read_csv(imp_path)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp_df["Feature"], imp_df["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        st.markdown("---")

    # ML images
    ml_images = sorted(p.ML_IMAGES_DIR.glob("*.png"))
    if ml_images:
        st.subheader("Evaluation Plots")
        cols = st.columns(2)
        for i, img_path in enumerate(ml_images):
            with cols[i % 2]:
                st.image(
                    str(img_path),
                    caption=img_path.stem.replace("_", " ").title(),
                )
    else:
        st.info("No ML plots found. Run the training pipeline first.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    home_page()
elif page == "📊 EDA Visualizations":
    eda_page()
elif page == "🤖 Prediction":
    prediction_page()
elif page == "📈 Model Insights":
    model_insights_page()
