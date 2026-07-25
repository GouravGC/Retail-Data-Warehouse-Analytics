"""
Model evaluation component.

Computes classification metrics, generates evaluation plots,
and saves reports to disk — matching the notebook's evaluation logic.
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import paths as p
from src.logger.logger import get_logger
from src.utils.helpers import save_csv

logger = get_logger(__name__)


class ModelEvaluation:
    """
    Evaluate trained models and persist reports / plots.
    """

    def __init__(self, reports_dir: Path = p.REPORTS_DIR, images_dir: Path = p.ML_IMAGES_DIR):
        self.reports_dir = reports_dir
        self.images_dir = images_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _save_plot(self, filename: str) -> None:
        """Save the current matplotlib figure."""
        path = self.images_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot: {path}")
        plt.close()

    def evaluate_all(
        self,
        models: Dict[str, object],
        X_test,
        y_test,
        X_test_scaled=None,
    ) -> pd.DataFrame:
        """
        Compute Accuracy, Precision, Recall, F1, ROC AUC for all models.

        Args:
            models: Dict of model name → fitted estimator.
            X_test: Test features (unscaled for tree models).
            y_test: Test target.
            X_test_scaled: Scaled test features (for LogisticRegression).

        Returns:
            DataFrame of metrics.
        """
        logger.info("Evaluating all models ...")
        results = []

        for name, model in models.items():
            # Determine correct feature set
            if "Logistic" in name:
                eval_data = X_test_scaled if X_test_scaled is not None else X_test
            else:
                eval_data = X_test

            pred = model.predict(eval_data)
            prob = model.predict_proba(eval_data)[:, 1]

            results.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred),
                    "Recall": recall_score(y_test, pred),
                    "F1": f1_score(y_test, pred),
                    "ROC_AUC": roc_auc_score(y_test, prob),
                }
            )

        results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
        logger.info(f"Evaluation complete:\n{results_df.to_string()}")
        return results_df

    def plot_confusion_matrix(self, y_test, y_pred) -> None:
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        self._save_plot("confusion_matrix.png")

    def plot_roc_curve(self, y_test, y_prob) -> None:
        """Plot and save ROC curve."""
        fig, ax = plt.subplots(figsize=(6, 6))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        plt.title("ROC Curve")
        plt.tight_layout()
        self._save_plot("roc_curve.png")

    def plot_precision_recall_curve(self, y_test, y_prob) -> None:
        """Plot and save Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(6, 6))
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
        plt.title("Precision Recall Curve")
        plt.tight_layout()
        self._save_plot("precision_recall_curve.png")

    def plot_feature_importance(self, model, feature_names: list) -> pd.DataFrame:
        """
        Bar plot of feature importance from a tree-based model.

        Args:
            model: Fitted model with ``feature_importances_`` attribute.
            feature_names: List of feature column names.

        Returns:
            DataFrame of feature importances sorted descending.
        """
        importance = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        plt.figure(figsize=(8, 6))
        plt.barh(importance["Feature"], importance["Importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance")
        plt.tight_layout()
        self._save_plot("feature_importance.png")
        logger.info("Feature importance plot saved.")
        return importance

    def save_predictions(
        self, X_test, y_test, y_pred, y_prob, path: Path = None
    ) -> pd.DataFrame:
        """
        Save test predictions alongside actual values and probabilities.

        Args:
            X_test: Test feature DataFrame.
            y_test: Actual target values.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities.
            path: Output CSV path.

        Returns:
            DataFrame with predictions.
        """
        path = path or p.TEST_PREDICTIONS_PATH
        predictions = X_test.copy()
        predictions["Actual"] = y_test.values
        predictions["Predicted"] = y_pred
        predictions["Probability"] = y_prob
        save_csv(predictions, path)
        return predictions

