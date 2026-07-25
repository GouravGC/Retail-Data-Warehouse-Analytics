# Retail Data Warehouse Analytics

**End-to-End SQL + Machine Learning Project**

An end-to-end retail analytics pipeline using the **Olist Brazilian E-Commerce** dataset. The project covers data exploration, cleaning, relational database creation, advanced SQL analytics, feature engineering, supervised machine learning, model explainability, and deployment readiness.

---

## Live Demo

🚀 **Streamlit Application:** https://your-app-name.streamlit.app

---

## Structure

```
├── app.py                            # Streamlit application
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installer
├── README.md                         # This file
│
├── src/
│   ├── __init__.py
│   ├── constants/
│   │   └── paths.py                  # Centralised artifact paths (pathlib)
│   ├── components/
│   │   ├── data_ingestion.py         # Extract zip & load raw CSVs
│   │   ├── data_preprocessing.py     # Clean, convert dtypes, missing-value analysis
│   │   ├── eda_analysis.py           # All EDA visualisations
│   │   ├── sql_analytics.py          # SQLite DB setup, queries, ML views
│   │   ├── feature_engineering.py    # Feature engineering (V2 logic)
│   │   ├── model_training.py         # Model training & hyperparameter tuning
│   │   ├── model_evaluation.py       # Metrics, plots, reports
│   │   └── shap_explainability.py    # SHAP value computation & plots
│   ├── pipeline/
│   │   ├── training_pipeline.py      # Orchestrates the full training pipeline
│   │   └── prediction_pipeline.py    # Loads artifacts and runs inference
│   ├── utils/
│   │   └── helpers.py                # Shared utility functions
│   ├── exception/
│   │   └── custom_exception.py       # Custom exception handling
│   ├── logger/
│   │   └── logger.py                 # Logging configuration
│   ├── configuration/
│   │   └── configuration.py          # Configuration manager
│   └── entity/
│       └── config_entity.py          # Dataclass-based configuration
│
├── notebooks/
│   └── Retail Data Warehouse Analytics.ipynb   # Original notebook
│
└── artifacts/                         # Saved models, reports, plots, data
    ├── models/                        # final_model.pkl, scaler.pkl, ...
    ├── reports/                       # CSV reports
    ├── images/                        # EDA & ML plots
    ├── database/                      # SQLite database
    └── data/                          # Raw & processed CSVs
```

---

## Key Features

- **SQL + Python Analytics**: Both Python (pandas) and SQL (SQLite) analyses
- **Feature Engineering (V2)**: Recency, frequency, monetary (RFM), freight ratios, product scores
- **Model Training**: Logistic Regression, Random Forest (with hyperparameter tuning), XGBoost
- **Best Model**: Tuned Random Forest (ROC AUC ≈ 1.0 on test set)
- **Model Explainability**: SHAP summary and bar plots
- **Streamlit App**: Interactive prediction and visualization dashboard
- **Modular Codebase**: Clean separation of concerns, reusable components

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/retail-data-warehouse-analytics.git
cd retail-data-warehouse-analytics
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 5. Run the full training pipeline (optional)

```bash
python -m src.pipeline.training_pipeline
```

---

## Usage

### Prediction

The Streamlit app provides a **Prediction** page where you can input customer features (total spent, order value, review scores, etc.) and get:

- **High-value customer** classification (yes / no)
- **Probability score** (0–100%)
- **Confidence level** visualisation

### Model Insights

View feature importance plots, confusion matrix, ROC curve, Precision-Recall curve, and SHAP explainability plots.

### EDA

Explore the Olist dataset through visualisations: monthly orders, customer distribution, payment methods, review scores, and more.

---

## Dataset

The **Olist Brazilian E-Commerce Public Dataset** contains ~100k orders from 2016–2018 across multiple states in Brazil. It includes:

- Customers & geolocation
- Orders, items, payments
- Product catalogue
- Seller information
- Customer reviews

---

## License

This project is for educational and portfolio purposes. The Olist dataset is provided by Olist Store and is available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
