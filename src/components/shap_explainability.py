"""
SHAP explainability component.

Generates SHAP summary and bar plots for the best model
to explain feature contributions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import shap

from src.constants import paths as p
from src.logger.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    Compute and visualise SHAP values for the trained model.

    Only works with tree-based models (TreeExplainer).
    """

    def __init__(self, images_dir: Path = p.ML_IMAGES_DIR):
        self.images_dir = images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _save_plot(self, filename: str) -> None:
        """Save the current matplotlib figure."""
        path = self.images_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved SHAP plot: {path}")
        plt.close()

    def explain(self, model, X_sample):
        """
        Compute SHAP values using a TreeExplainer and generate plots.

        Args:
            model: Fitted tree-based model (RandomForest, XGBoost, etc.).
            X_sample: Feature DataFrame (typically X_test).

        Returns:
            Computed SHAP values array.
        """
        logger.info("Computing SHAP values ...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        logger.info(f"SHAP values shape: {len(shap_values)}")

        # ---- Summary plot (dot) ----
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        self._save_plot("shap_summary.png")

        # ---- Summary plot (bar) ----
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        self._save_plot("shap_bar.png")

        return shap_values

