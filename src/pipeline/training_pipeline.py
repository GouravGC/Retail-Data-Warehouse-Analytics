"""
Training pipeline.

Orchestrates the end-to-end training process:
1. Data ingestion
2. Preprocessing & cleaning
3. EDA (optional)
4. SQL database setup & feature views
5. Feature engineering (V2)
6. Model training, tuning, evaluation
7. SHAP explainability
8. Save all artifacts
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.eda_analysis import EDAAnalysis
from src.components.sql_analytics import SQLAnalytics
from src.components.feature_engineering import FeatureEngineering
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation
from src.components.shap_explainability import SHAPExplainer
from src.configuration.configuration import ConfigurationManager
from src.logger.logger import setup_logging, get_logger
from src.utils.helpers import create_directories, save_csv
from src.constants import paths as p

setup_logging()
logger = get_logger(__name__)


class TrainingPipeline:
    """
    Run the complete ML pipeline from data ingestion to SHAP explainability.
    """

    def __init__(self):
        self.config = ConfigurationManager()

    def run(self) -> None:
        """Execute every stage of the pipeline."""
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        # ---- 0. Ensure directories exist ----
        create_directories([
            p.RAW_DATA_DIR,
            p.PROCESSED_DATA_DIR,
            p.DATABASE_DIR,
            p.MODELS_DIR,
            p.REPORTS_DIR,
            p.EDA_IMAGES_DIR,
            p.ML_IMAGES_DIR,
            p.SHAP_DIR,
            p.LOGS_DIR,
        ])

        # ---- 1. Data Ingestion ----
        logger.info("--- STAGE 1: Data Ingestion ---")
        ingestion = DataIngestion()
        ingestion.extract_raw_data()
        datasets = ingestion.load_datasets()

        # ---- 2. Data Preprocessing ----
        logger.info("--- STAGE 2: Data Preprocessing ---")
        preprocessor = DataPreprocessing()
        summary = preprocessor.generate_dataset_summary(datasets)
        save_csv(summary, p.DATASET_SUMMARY_PATH)
        datasets = preprocessor.convert_date_columns(datasets)
        datasets = preprocessor.merge_translation(datasets)
        preprocessor.save_cleaned_datasets(datasets)

        # ---- 3. EDA (optional - can be skipped in production) ----
        logger.info("--- STAGE 3: EDA ---")
        eda = EDAAnalysis()
        eda.monthly_order_trend(datasets["Orders"])
        eda.order_status_distribution(datasets["Orders"])
        eda.customers_by_state(datasets["Customers"])
        eda.top_customer_cities(datasets["Customers"])
        eda.payment_method_distribution(datasets["Payments"])
        eda.payment_installments_distribution(datasets["Payments"])
        eda.review_score_distribution(datasets["Reviews"])
        eda.top_product_categories(datasets["Products"])
        eda.seller_distribution(datasets["Sellers"])
        eda.delivery_time_distribution(datasets["Orders"])
        eda.correlation_heatmap(datasets["Products"])
        logger.info("EDA plots saved.")

        # ---- 4. SQL Database & Views ----
        logger.info("--- STAGE 4: SQL Analytics ---")
        sql = SQLAnalytics()
        sql.load_data_to_db()
        sql.create_ml_features_view()
        sql.create_ml_features_v2_view()

        # ---- 5. Feature Engineering (V2) ----
        logger.info("--- STAGE 5: Feature Engineering ---")
        ml_df_v2 = sql.load_ml_features(version=2)
        fe = FeatureEngineering()
        ml_df_engineered = fe.engineer_features(ml_df_v2)
        fe.save_ml_dataset(ml_df_engineered)
        X, y = fe.prepare_features_and_target(ml_df_engineered)

        # ---- 6. Model Training ----
        logger.info("--- STAGE 6: Model Training ---")
        train_config = self.config.get_model_training_config()
        trainer = ModelTraining(train_config)
        trainer.split_data(X, y)
        trainer.scale_features()

        # Train models
        trainer.train_logistic_regression()
        trainer.train_random_forest()
        search_config = self.config.get_randomized_search_config()
        trainer.tune_random_forest(search_config)
        trainer.train_xgboost()

        # Cross-validate tuned model
        trainer.cross_validate(trainer.best_rf, X, y)

        # Save all models
        trainer.save_artifacts()

        # ---- 7. Model Evaluation ----
        logger.info("--- STAGE 7: Model Evaluation ---")
        evaluator = ModelEvaluation()

        models = {
            "Logistic Regression": trainer.lr,
            "Random Forest": trainer.rf,
            "Tuned Random Forest": trainer.best_rf,
            "XGBoost": trainer.xgb,
        }
        results_df = evaluator.evaluate_all(
            models,
            trainer.X_test,
            trainer.y_test,
            trainer.X_test_scaled,
        )
        save_csv(results_df, p.MODEL_COMPARISON_PATH)

        # Best model predictions
        best_model = trainer.best_rf
        y_pred = best_model.predict(trainer.X_test)
        y_prob = best_model.predict_proba(trainer.X_test)[:, 1]

        # Plot evaluation figures
        evaluator.plot_confusion_matrix(trainer.y_test, y_pred)
        evaluator.plot_roc_curve(trainer.y_test, y_prob)
        evaluator.plot_precision_recall_curve(trainer.y_test, y_prob)
        evaluator.plot_feature_importance(
            best_model, trainer.X_test.columns.tolist()
        )

        # Save predictions
        evaluator.save_predictions(
            trainer.X_test, trainer.y_test, y_pred, y_prob
        )
        logger.info(classification_report(trainer.y_test, y_pred))

        # ---- 8. SHAP Explainability ----
        logger.info("--- STAGE 8: SHAP Explainability ---")
        explainer = SHAPExplainer()
        explainer.explain(best_model, trainer.X_test)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)


def classification_report(y_true, y_pred):
    """Return a one-line summary of classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return (
        f"Accuracy: {accuracy_score(y_true, y_pred):.4f}, "
        f"Precision: {precision_score(y_true, y_pred):.4f}, "
        f"Recall: {recall_score(y_true, y_pred):.4f}, "
        f"F1: {f1_score(y_true, y_pred):.4f}"
    )


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
