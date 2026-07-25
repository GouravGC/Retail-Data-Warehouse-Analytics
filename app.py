"""
Streamlit application entry point.

Provides an interactive dashboard for:
- Model prediction (single / batch) with prediction history
- Model explainability insights (SHAP, feature importance)
- EDA visualisations (static + interactive)
- Model information & metrics
- Architecture & About pages
- Download center
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import json
import io
from pathlib import Path
from datetime import datetime

from src.constants import paths as p
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger.logger import setup_logging, get_logger
from src.app_utils import (
    FEATURE_METADATA,
    validate_inputs,
    init_prediction_history,
    add_to_history,
    get_history_df,
    clear_history,
    ValidationError,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Data Warehouse Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts ...")
def load_prediction_pipeline() -> PredictionPipeline:
    """Load the prediction pipeline (cached)."""
    return PredictionPipeline()


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    """Load model comparison CSV (cached)."""
    path = p.MODEL_COMPARISON_PATH
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    """Load feature importance CSV (cached)."""
    path = p.FEATURE_IMPORTANCE_PATH
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dataset_summary() -> pd.DataFrame:
    """Load dataset summary CSV (cached)."""
    path = p.DATASET_SUMMARY_PATH
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_eda_image_list() -> list:
    """Return sorted list of EDA image paths (cached)."""
    return sorted(p.EDA_IMAGES_DIR.glob("*.png"))


@st.cache_data(show_spinner=False)
def get_ml_image_list() -> list:
    """Return sorted list of ML image paths (cached)."""
    return sorted(p.ML_IMAGES_DIR.glob("*.png"))


@st.cache_data(show_spinner=False)
def get_python_version() -> str:
    """Return Python version string."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@st.cache_data(show_spinner=False)
def get_library_versions() -> dict:
    """Return key library versions."""
    import sklearn
    import numpy
    import pandas as pd
    import matplotlib
    import seaborn
    import joblib

    versions = {
        "scikit-learn": sklearn.__version__,
        "NumPy": numpy.__version__,
        "Pandas": pd.__version__,
        "Matplotlib": matplotlib.__version__,
        "Seaborn": seaborn.__version__,
        "Joblib": joblib.__version__,
    }
    try:
        import xgboost
        versions["XGBoost"] = xgboost.__version__
    except ImportError:
        versions["XGBoost"] = "N/A"
    try:
        import shap
        versions["SHAP"] = shap.__version__
    except ImportError:
        versions["SHAP"] = "N/A"
    try:
        import plotly
        versions["Plotly"] = plotly.__version__
    except ImportError:
        versions["Plotly"] = "N/A"

    return versions


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------
init_prediction_history()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏪 Retail Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🤖 Prediction",
            "📊 EDA Visualizations",
            "📈 Model Insights",
            "ℹ️ Model Information",
            "🏗️ Architecture",
            "ℹ️ About",
            "📥 Download Center",
        ],
    )

    st.markdown("---")
    st.caption(f"**App Version:** 1.0.0")
    st.caption(f"**Model:** Tuned Random Forest")
    st.caption(f"**Streamlit:** {st.__version__}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<a href='https://github.com/GouravGC/Retail-Data-Warehouse-Analytics' target='_blank' "
            "style='text-decoration:none;'>"
            "<button style='width:100%; cursor:pointer;'>🐙 GitHub</button></a>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<a href='https://retail-data-warehouse-analytics.streamlit.app/' target='_blank' "
            "style='text-decoration:none;'>"
            "<button style='width:100%; cursor:pointer;'>🌐 Demo</button></a>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Contact**")
    st.caption("📧 chhatwanigourav@gmail.com")
    st.caption("🔗 [LinkedIn](https://www.linkedin.com/in/gourav-chhatwani-9a301134a/)")

    st.markdown("---")
    st.info(
        "This application uses a **Tuned Random Forest** model to predict "
        "high-value customers based on their purchase behaviour."
    )


# =============================================================================
# PAGE: Home
# =============================================================================
def home_page():
    st.title("Retail Data Warehouse Analytics")
    st.markdown(
        """
        ### End-to-End SQL + Machine Learning Project

        This project analyses the **Olist Brazilian E-Commerce** dataset and builds
        a machine learning model to identify **high-value customers**.
        """
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", "Tuned RF", "ROC AUC: 1.0")
    with col2:
        st.metric("Features Used", "15", "Engineered from SQL views")
    with col3:
        st.metric("Total Customers", "95,419", "Unique IDs")
    with col4:
        st.metric("Target Definition", "Top 25%", "By lifetime value")

    st.markdown("---")
    st.subheader("Pipeline Stages")
    stages_col1, stages_col2 = st.columns(2)
    with stages_col1:
        st.markdown("""
        1. 📥 **Data Ingestion & Cleaning**
        2. 📊 **Exploratory Data Analysis** (Python + SQL)
        3. 🛠️ **Feature Engineering** (Version 2)
        4. 🤖 **Model Training** (LR, RF, XGBoost)
        """)
    with stages_col2:
        st.markdown("""
        5. 🎯 **Hyperparameter Tuning** (RandomizedSearchCV)
        6. 📈 **Model Evaluation & Selection**
        7. 🔍 **SHAP Explainability**
        8. 🚀 **Deployment** (Streamlit)
        """)

    # Show dataset summary
    summary_df = load_dataset_summary()
    if not summary_df.empty:
        st.markdown("---")
        st.subheader("Dataset Summary")
        st.dataframe(summary_df, use_container_width=True)


# =============================================================================
# PAGE: EDA Visualizations
# =============================================================================
def eda_page():
    st.title("Exploratory Data Analysis")
    st.markdown("Visual insights generated from the Olist dataset.")

    eda_images = get_eda_image_list()
    if not eda_images:
        st.warning("No EDA visualisations found. Run the training pipeline first.")
        return

    # Organize by category using tabs
    tab_names = [
        "Orders & Delivery",
        "Customers & Sellers",
        "Payments & Products",
        "Reviews & Correlation",
    ]
    tabs = st.tabs(tab_names)

    # Group images by category
    categories = {
        "Orders & Delivery": [
            img for img in eda_images if "monthly_orders" in img.name
            or "order_status" in img.name
            or "delivery_time" in img.name
        ],
        "Customers & Sellers": [
            img for img in eda_images if "customer" in img.name.lower()
            or "seller" in img.name.lower()
        ],
        "Payments & Products": [
            img for img in eda_images if "payment" in img.name
            or "product" in img.name.lower()
            or "installment" in img.name
        ],
        "Reviews & Correlation": [
            img for img in eda_images if "review" in img.name
            or "correlation" in img.name
        ],
    }

    for tab, (category, images) in zip(tabs, categories.items()):
        with tab:
            if not images:
                st.info(f"No images in {category} category.")
                continue
            cols = st.columns(2)
            for i, img_path in enumerate(images):
                with cols[i % 2]:
                    st.image(
                        str(img_path),
                        caption=img_path.stem.replace("_", " ").title(),
                        use_container_width=True,
                    )


# =============================================================================
# PAGE: Prediction
# =============================================================================
def prediction_page():
    st.title("Customer Value Prediction")
    st.markdown(
        "Enter customer feature values below to predict whether they are a "
        "**high-value customer** (top 25% by lifetime value)."
    )

    # Load pipeline (cached)
    try:
        pipeline = load_prediction_pipeline()
    except Exception as e:
        st.error(f"Failed to load prediction pipeline: {e}")
        st.info("Ensure the model artifacts exist in `artifacts/models/`.")
        return

    # Tabs for prediction and history
    pred_tab, history_tab = st.tabs(["🔮 Make Prediction", "📋 Prediction History"])

    with pred_tab:
        # Feature descriptions in an expander
        with st.expander("ℹ️ Feature Descriptions & Ranges", expanded=False):
            st.markdown("Each feature helps the model understand customer behaviour. Hover over the icons for details.")
            cols = st.columns(3)
            for idx, (feat, meta) in enumerate(FEATURE_METADATA.items()):
                with cols[idx % 3]:
                    st.markdown(
                        f"**{meta['icon']} {meta['label']}**  \n"
                        f"*{meta['description']}*  \n"
                        f"📊 Range: {meta['range']}  \n"
                        f"📈 Typical: {meta['typical']}  \n"
                        f"💡 {meta['importance']}"
                    )

        # Input form
        with st.form("prediction_form"):
            st.markdown("### 📝 Customer Profile")

            col1, col2, col3 = st.columns(3)

            # Column 1: Spending & Products
            with col1:
                st.markdown("**💳 Spending & Products**")
                total_spent = st.number_input(
                    f"{FEATURE_METADATA['total_spent']['icon']} Total Spent",
                    min_value=0.0, value=500.0, step=10.0,
                    help=FEATURE_METADATA["total_spent"]["importance"],
                )
                avg_order_value = st.number_input(
                    f"{FEATURE_METADATA['avg_order_value']['icon']} Avg Order Value",
                    min_value=0.0, value=150.0, step=5.0,
                    help=FEATURE_METADATA["avg_order_value"]["importance"],
                )
                avg_review_score = st.slider(
                    f"{FEATURE_METADATA['avg_review_score']['icon']} Avg Review Score",
                    1.0, 5.0, 4.0, 0.5,
                    help=FEATURE_METADATA["avg_review_score"]["importance"],
                )
                unique_products = st.number_input(
                    f"{FEATURE_METADATA['unique_products']['icon']} Unique Products",
                    min_value=1, value=5, step=1,
                    help=FEATURE_METADATA["unique_products"]["importance"],
                )
                avg_freight = st.number_input(
                    f"{FEATURE_METADATA['avg_freight']['icon']} Avg Freight Cost",
                    min_value=0.0, value=20.0, step=1.0,
                    help=FEATURE_METADATA["avg_freight"]["importance"],
                )

            # Column 2: Freight & Payments
            with col2:
                st.markdown("**🚚 Shipping & Payments**")
                total_freight = st.number_input(
                    f"{FEATURE_METADATA['total_freight']['icon']} Total Freight",
                    min_value=0.0, value=100.0, step=5.0,
                    help=FEATURE_METADATA["total_freight"]["importance"],
                )
                avg_product_price = st.number_input(
                    f"{FEATURE_METADATA['avg_product_price']['icon']} Avg Product Price",
                    min_value=0.0, value=100.0, step=5.0,
                    help=FEATURE_METADATA["avg_product_price"]["importance"],
                )
                avg_installments = st.number_input(
                    f"{FEATURE_METADATA['avg_installments']['icon']} Avg Installments",
                    min_value=1.0, value=4.0, step=0.5,
                    help=FEATURE_METADATA["avg_installments"]["importance"],
                )
                credit_card_orders = st.number_input(
                    f"{FEATURE_METADATA['credit_card_orders']['icon']} Credit Card Orders",
                    min_value=0, value=3, step=1,
                    help=FEATURE_METADATA["credit_card_orders"]["importance"],
                )
                avg_delivery_days = st.number_input(
                    f"{FEATURE_METADATA['avg_delivery_days']['icon']} Avg Delivery Days",
                    min_value=1.0, value=12.0, step=0.5,
                    help=FEATURE_METADATA["avg_delivery_days"]["importance"],
                )

            # Column 3: Engagement
            with col3:
                st.markdown("**🔄 Customer Engagement**")
                recency_days = st.number_input(
                    f"{FEATURE_METADATA['recency_days']['icon']} Recency (Days)",
                    min_value=0, value=30, step=1,
                    help=FEATURE_METADATA["recency_days"]["importance"],
                )
                customer_age_days = st.number_input(
                    f"{FEATURE_METADATA['customer_age_days']['icon']} Customer Age (Days)",
                    min_value=0, value=365, step=1,
                    help=FEATURE_METADATA["customer_age_days"]["importance"],
                )
                purchase_frequency = st.number_input(
                    f"{FEATURE_METADATA['purchase_frequency']['icon']} Purchase Frequency",
                    min_value=0.0, value=0.05, step=0.01, format="%.4f",
                    help=FEATURE_METADATA["purchase_frequency"]["importance"],
                )
                products_per_order = st.number_input(
                    f"{FEATURE_METADATA['products_per_order']['icon']} Products per Order",
                    min_value=0.0, value=1.5, step=0.1,
                    help=FEATURE_METADATA["products_per_order"]["importance"],
                )
                freight_ratio = st.number_input(
                    f"{FEATURE_METADATA['freight_ratio']['icon']} Freight Ratio",
                    min_value=0.0, value=0.2, step=0.01, format="%.3f",
                    help=FEATURE_METADATA["freight_ratio"]["importance"],
                )

            submitted = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)

        if submitted:
            # Build input data dict
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

            # Validate inputs
            errors = validate_inputs(input_data)
            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                features_df = pd.DataFrame([input_data])

                try:
                    with st.spinner("🤖 Analysing customer data ..."):
                        label = pipeline.predict(features_df)
                        prob = pipeline.predict_proba(features_df)

                    # Add to session history
                    add_to_history(input_data, label, prob)

                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")

                    col_a, col_b, col_c = st.columns(3)

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
                        st.metric(
                            "Probability of High-Value",
                            f"{prob:.2%}",
                            delta=f"{'↑' if prob > 0.5 else '↓'}",
                        )

                    with col_c:
                        st.metric(
                            "Confidence Level",
                            f"{prob:.1%}",
                            delta=f"{'High' if prob > 0.8 or prob < 0.2 else 'Medium'}",
                        )

                    # Progress bar for visual confidence
                    st.markdown("**Confidence Gauge**")
                    st.progress(prob)
                    st.caption(f"{prob:.1%} probability of being high-value (class 1)")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    logger.exception("Prediction error")

    # Prediction History tab
    with history_tab:
        st.subheader("📋 Prediction History")
        history_df = get_history_df()

        if history_df.empty:
            st.info("No predictions made yet. Go to the **Make Prediction** tab to get started.")
        else:
            # Display history table
            display_cols = ["timestamp", "prediction", "probability"] + list(FEATURE_METADATA.keys())
            available_cols = [c for c in display_cols if c in history_df.columns]
            st.dataframe(history_df[available_cols], use_container_width=True)

            # Download buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                csv_buffer = io.StringIO()
                history_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="prediction_history.csv",
                    mime="text/csv",
                )
            with col2:
                json_str = history_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name="prediction_history.json",
                    mime="application/json",
                )
            with col3:
                if st.button("🗑️ Clear History", type="secondary"):
                    clear_history()
                    st.rerun()


# =============================================================================
# PAGE: Model Insights
# =============================================================================
def model_insights_page():
    st.title("Model Insights & Explainability")

    imp_df = load_feature_importance()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feature Importance",
        "🔬 SHAP Explainability",
        "📈 Evaluation Plots",
        "🏆 Top Features",
    ])

    with tab1:
        if not imp_df.empty:
            st.subheader("Global Feature Importance (Tuned Random Forest)")

            # Plotly interactive bar chart
            fig = px.bar(
                imp_df.sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Static matplotlib version (preserved)
            st.markdown("**Static Version**")
            fig_mpl, ax = plt.subplots(figsize=(8, 6))
            ax.barh(imp_df["Feature"], imp_df["Importance"])
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            st.pyplot(fig_mpl)
        else:
            st.warning("Feature importance data not found.")

    with tab2:
        st.subheader("SHAP Explainability")
        shap_images = [
            img for img in get_ml_image_list()
            if "shap" in img.name
        ]

        if shap_images:
            shap_tabs = st.tabs([img.stem.replace("_", " ").title() for img in shap_images])
            for tab, img_path in zip(shap_tabs, shap_images):
                with tab:
                    st.image(str(img_path), use_container_width=True)
                    st.caption(f"**{img_path.stem.replace('_', ' ').title()}** — "
                               "SHAP values explain how each feature contributes to predictions.")
        else:
            st.info("No SHAP plots found. Run the training pipeline first.")

        # Local explanation placeholder
        with st.expander("🔍 Local Explanation (Top Features)"):
            st.markdown("""
            SHAP (SHapley Additive exPlanations) explains individual predictions by
            measuring the contribution of each feature.

            **Key Insights from SHAP:**
            - **total_spent** has the highest impact — higher spending pushes predictions toward high-value.
            - **avg_order_value** and **avg_product_price** are the next most influential.
            - **freight_ratio** can negatively impact if shipping costs are high relative to spend.
            """)

    with tab3:
        st.subheader("Evaluation Plots")

        # Separate SHAP from other ML images
        eval_images = [
            img for img in get_ml_image_list()
            if "shap" not in img.name
        ]

        if eval_images:
            # Organize with tabs
            categories = {
                "Confusion Matrix": [img for img in eval_images if "confusion" in img.name],
                "ROC & PR Curves": [img for img in eval_images if "roc" in img.name or "precision_recall" in img.name],
                "Other": [img for img in eval_images if "confusion" not in img.name and "roc" not in img.name and "precision_recall" not in img.name and "feature_importance" not in img.name],
            }

            eval_tabs = st.tabs(list(categories.keys()))
            for tab, (cat_name, images) in zip(eval_tabs, categories.items()):
                with tab:
                    if not images:
                        st.info(f"No images in {cat_name}.")
                        continue
                    cols = st.columns(2)
                    for i, img_path in enumerate(images):
                        with cols[i % 2]:
                            st.image(
                                str(img_path),
                                caption=img_path.stem.replace("_", " ").title(),
                                use_container_width=True,
                            )
        else:
            st.info("No evaluation plots found. Run the training pipeline first.")

    with tab4:
        st.subheader("🏆 Top Features")

        if not imp_df.empty:
            top_5 = imp_df.head(5)
            top_10 = imp_df.head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 5 Features**")
                fig5 = px.bar(
                    top_5.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Greens",
                )
                fig5.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig5, use_container_width=True)

            with col2:
                st.markdown("**Top 10 Features**")
                fig10 = px.bar(
                    top_10.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Oranges",
                )
                fig10.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig10, use_container_width=True)

            # Show full table
            with st.expander("📋 View Full Feature Importance Table"):
                st.dataframe(imp_df, use_container_width=True)
        else:
            st.warning("Feature importance data not found.")


# =============================================================================
# PAGE: Model Information
# =============================================================================
def model_info_page():
    st.title("ℹ️ Model Information")
    st.markdown("Detailed information about the trained model.")

    model_comparison = load_model_comparison()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Name", "Tuned Random Forest")
        st.metric("Algorithm", "RandomForestClassifier")
    with col2:
        st.metric("Number of Features", "15")
        st.metric("Training Samples", "76,335 (80%)")
    with col3:
        # Model file size
        model_path = p.FINAL_MODEL_PATH
        if model_path.exists():
            size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
            st.metric("Model File Size", f"{size_mb} MB")
        st.metric("Test Samples", "19,084 (20%)")

    st.markdown("---")

    # Performance metrics
    st.subheader("📊 Performance Metrics")
    if not model_comparison.empty:
        # Find tuned RF row
        rf_row = model_comparison[model_comparison["Model"].str.contains("Tuned", case=False)]
        if not rf_row.empty:
            row = rf_row.iloc[0]
            metrics_cols = st.columns(5)
            metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
            for col, label in zip(metrics_cols, metric_labels):
                val = row.get(label, row.get(label.replace(" ", "_"), "N/A"))
                with col:
                    st.metric(label, f"{float(val):.4f}" if val != "N/A" else "N/A")
        else:
            st.dataframe(model_comparison, use_container_width=True)
    else:
        st.info("Model comparison data not found.")

    st.markdown("---")

    # Hyperparameters
    st.subheader("⚙️ Hyperparameters")
    hyperparams = {
        "n_estimators": "500",
        "max_depth": "15",
        "min_samples_split": "5",
        "min_samples_leaf": "4",
        "criterion": "gini",
        "bootstrap": "True",
        "random_state": "42",
    }
    hp_cols = st.columns(4)
    for i, (param, value) in enumerate(hyperparams.items()):
        with hp_cols[i % 4]:
            st.metric(param, value)

    st.markdown("---")

    # Environment info
    st.subheader("💻 Environment")
    lib_versions = get_library_versions()
    env_cols = st.columns(4)
    env_items = [("Python", get_python_version())] + list(lib_versions.items())
    for i, (name, ver) in enumerate(env_items):
        with env_cols[i % 4]:
            st.metric(name, ver)


# =============================================================================
# PAGE: Architecture
# =============================================================================
def architecture_page():
    st.title("🏗️ Project Architecture")
    st.markdown("End-to-end data flow and architecture of the Retail Data Warehouse Analytics project.")

    # Mermaid diagram
    mermaid_code = """
    ```mermaid
    graph TD
        A[Raw Olist Data] --> B[Data Ingestion]
        B --> C[Data Preprocessing]
        C --> D[(SQLite Database)]
        C --> E[EDA Visualizations]

        D --> F[Feature Engineering V2]
        F --> G[Train/Test Split]
        G --> H[Feature Scaling]

        H --> I[Logistic Regression]
        H --> J[Random Forest]
        H --> K[XGBoost]

        J --> L[RandomizedSearchCV]
        L --> M[Tuned Random Forest]

        M --> N[Model Evaluation]
        N --> O[Confusion Matrix]
        N --> P[ROC Curve]
        N --> Q[Precision-Recall Curve]

        M --> R[SHAP Explainability]
        R --> S[SHAP Summary]
        R --> T[SHAP Bar Plot]

        M --> U[Save final_model.pkl]
        H --> V[Save scaler.pkl]

        U --> W[Streamlit App]
        V --> W
        W --> X[Prediction]
        W --> Y[EDA Viewer]
        W --> Z[Model Insights]
    ```
    """
    st.markdown(mermaid_code)

    st.markdown("---")

    # Text description of architecture
    st.subheader("Data Flow Description")

    flow_col1, flow_col2 = st.columns(2)

    with flow_col1:
        st.markdown("""
        **1. Data Ingestion**
        - Extracts raw CSV files from the Olist zip archive
        - Loads into pandas DataFrames

        **2. Data Preprocessing**
        - Converts date columns to datetime
        - Merges product translations
        - Saves cleaned datasets
        - Generates dataset summary report

        **3. EDA (Python + SQL)**
        - Visualises monthly orders, customer distribution, payments
        - Loads data into SQLite database
        - Runs analytical SQL queries
        - Creates ML feature views
        """)

    with flow_col2:
        st.markdown("""
        **4. Feature Engineering (V2)**
        - Applies RFM analysis, recency, frequency
        - Calculates freight ratios, product scores
        - Creates high-value customer labels (75th percentile)

        **5. Model Training**
        - Logistic Regression (baseline)
        - Random Forest (500 trees)
        - XGBoost (400 rounds)
        - Hyperparameter tuning (RandomizedSearchCV)

        **6. Deployment**
        - Best model saved as `final_model.pkl`
        - Scaler saved for preprocessing
        - Streamlit app loads artifacts for inference
        """)

    st.markdown("---")
    st.subheader("Technology Stack")
    tech_cols = st.columns(5)
    tech_items = [
        ("Python", "🐍"),
        ("Pandas", "🐼"),
        ("Scikit-learn", "⚡"),
        ("XGBoost", "🚀"),
        ("Streamlit", "🎈"),
    ]
    for col, (tech, icon) in zip(tech_cols, tech_items):
        with col:
            st.markdown(f"**{icon} {tech}**")


# =============================================================================
# PAGE: About
# =============================================================================
def about_page():
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## Project Overview

    **Retail Data Warehouse Analytics** is an end-to-end data science project that
    demonstrates the complete machine learning lifecycle — from raw data ingestion
    and SQL analytics to model deployment with Streamlit.

    ## Business Problem

    An e-commerce platform wants to identify **high-value customers** — the top 25%
    of customers by lifetime value. By predicting which customers are likely to be
    high-value, the business can:
    - Target marketing campaigns more effectively
    - Offer personalised loyalty rewards
    - Optimise customer retention strategies
    - Allocate resources to the most valuable segments

    ## Dataset

    The **Olist Brazilian E-Commerce Public Dataset** contains ~100k orders placed
    on the Olist Store between 2016 and 2018 across multiple states in Brazil.
    The dataset includes:
    - **99,441** customers
    - **112,650** order items
    - **32,951** products
    - **3,095** sellers
    - **1M+** geolocation records
    """)

    st.markdown("---")

    # Technologies
    st.subheader("Technologies Used")
    tech_data = {
        "Category": ["Programming", "Data Processing", "Visualization", "Machine Learning", "MLOps", "Deployment"],
        "Tools": ["Python 3", "Pandas, NumPy, SQLite", "Matplotlib, Seaborn, Plotly", "Scikit-learn, XGBoost, SHAP", "Joblib, Logging", "Streamlit"],
    }
    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Project Structure
    st.subheader("📁 Project Structure")
    st.code("""
    Retail Data Warehouse Analytics/
    ├── app.py                        # Streamlit application
    ├── requirements.txt              # Python dependencies
    ├── setup.py                      # Package installer
    ├── README.md                     # Documentation
    │
    ├── src/
    │   ├── constants/paths.py        # Centralised paths (pathlib)
    │   ├── components/               # Pipeline stages
    │   │   ├── data_ingestion.py
    │   │   ├── data_preprocessing.py
    │   │   ├── eda_analysis.py
    │   │   ├── sql_analytics.py
    │   │   ├── feature_engineering.py
    │   │   ├── model_training.py
    │   │   ├── model_evaluation.py
    │   │   └── shap_explainability.py
    │   ├── pipeline/
    │   │   ├── training_pipeline.py
    │   │   └── prediction_pipeline.py
    │   ├── app_utils/                # Streamlit helper utilities
    │   ├── exception/
    │   ├── logger/
    │   ├── configuration/
    │   └── entity/
    │
    ├── notebooks/                    # Original Jupyter notebook
    │   └── Retail Data Warehouse Analytics.ipynb
    └── artifacts/                    # Saved models, reports, plots
    """)

    st.markdown("---")

    # Links and Author
    st.subheader("Links")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 🐙 [GitHub Repository](https://github.com/GouravGC/Retail-Data-Warehouse-Analytics)
        - 🌐 [Live Demo](https://retail-data-warehouse-analytics.streamlit.app/)
        - 📊 [Olist Dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
        """)
    with col2:
        st.markdown("""
        - 📧 **Author:** GCS Portfolio Projects
        - 🔗 [LinkedIn](https://www.linkedin.com/in/gourav-chhatwani-9a301134a/)
        - 🏢 **Purpose:** Portfolio / Learning
        """)


# =============================================================================
# PAGE: Download Center
# =============================================================================
def download_center():
    st.title("📥 Download Center")
    st.markdown("Download reports, predictions, and model artifacts.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Reports")

        # Model comparison
        model_comp = load_model_comparison()
        if not model_comp.empty:
            csv_buf = io.StringIO()
            model_comp.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Model Comparison (CSV)",
                data=csv_buf.getvalue(),
                file_name="model_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Feature importance
        feat_imp = load_feature_importance()
        if not feat_imp.empty:
            csv_buf = io.StringIO()
            feat_imp.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Feature Importance (CSV)",
                data=csv_buf.getvalue(),
                file_name="feature_importance.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Dataset summary
        ds_summary = load_dataset_summary()
        if not ds_summary.empty:
            csv_buf = io.StringIO()
            ds_summary.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Dataset Summary (CSV)",
                data=csv_buf.getvalue(),
                file_name="dataset_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col2:
        st.subheader("🤖 Predictions")

        # Prediction history
        history_df = get_history_df()
        if not history_df.empty:
            # CSV
            csv_buf = io.StringIO()
            history_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Prediction History (CSV)",
                data=csv_buf.getvalue(),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # JSON
            json_str = history_df.to_json(orient="records", indent=2)
            st.download_button(
                label="📥 Prediction History (JSON)",
                data=json_str,
                file_name="prediction_history.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("No prediction history. Make predictions first.")

        # Test predictions if available
        test_pred_path = p.TEST_PREDICTIONS_PATH
        if test_pred_path.exists():
            test_pred_df = pd.read_csv(test_pred_path)
            csv_buf = io.StringIO()
            test_pred_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Test Predictions (CSV)",
                data=csv_buf.getvalue(),
                file_name="test_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# =============================================================================
# Router
# =============================================================================
if page == "🏠 Home":
    home_page()
elif page == "📊 EDA Visualizations":
    eda_page()
elif page == "🤖 Prediction":
    prediction_page()
elif page == "📈 Model Insights":
    model_insights_page()
elif page == "ℹ️ Model Information":
    model_info_page()
elif page == "🏗️ Architecture":
    architecture_page()
elif page == "ℹ️ About":
    about_page()
elif page == "📥 Download Center":
    download_center()
