"""
Model training component.

Trains Logistic Regression, Random Forest (with hyperparameter tuning),
and XGBoost classifiers using the same settings as the notebook (Version 2).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import paths as p
from src.entity.config_entity import (
    ModelTrainingConfig,
    RandomizedSearchConfig,
)
from src.logger.logger import get_logger

logger = get_logger(__name__)


class ModelTraining:
    """
    Train and tune classification models for high-value customer prediction.

    Follows the exact same logic as the notebook (Version 2).
    """

    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.models_dir = config.models_dir
        self.random_state = config.random_state
        self.test_size = config.test_size

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated during training
        self.scaler = None
        self.lr = None
        self.rf = None
        self.best_rf = None
        self.xgb = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Perform train/test split (80/20, stratified).

        Args:
            X: Feature DataFrame.
            y: Target Series.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(
            f"Train/Test split: Train {self.X_train.shape}, Test {self.X_test.shape}"
        )

    def scale_features(self) -> None:
        """
        Fit StandardScaler on training data and transform both splits.
        """
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        logger.info("Feature scaling completed.")

    def train_logistic_regression(self) -> LogisticRegression:
        """
        Train LogisticRegression baseline model.

        Returns:
            Fitted LogisticRegression instance.
        """
        self.lr = LogisticRegression(
            random_state=self.random_state, max_iter=self.config.lr_max_iter
        )
        self.lr.fit(self.X_train_scaled, self.y_train)
        logger.info("Logistic Regression trained.")
        return self.lr

    def train_random_forest(self) -> RandomForestClassifier:
        """
        Train RandomForest baseline model (n_estimators=500, defaults from config).

        Returns:
            Fitted RandomForestClassifier instance.
        """
        self.rf = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf.fit(self.X_train, self.y_train)
        logger.info("Random Forest trained.")
        return self.rf

    def tune_random_forest(
        self, search_config: RandomizedSearchConfig
    ) -> RandomForestClassifier:
        """
        Perform RandomizedSearchCV on RandomForest to find best hyperparameters.

        Args:
            search_config: Configuration for the search.

        Returns:
            Best estimator from the search.
        """
        logger.info("Starting RandomizedSearchCV for Random Forest ...")

        param_grid = {
            "n_estimators": search_config.n_estimators,
            "max_depth": search_config.max_depth,
            "min_samples_split": search_config.min_samples_split,
            "min_samples_leaf": search_config.min_samples_leaf,
        }

        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state),
            param_distributions=param_grid,
            n_iter=search_config.n_iter,
            cv=search_config.cv,
            scoring=search_config.scoring,
            n_jobs=-1,
            random_state=self.random_state,
        )
        search.fit(self.X_train, self.y_train)

        self.best_rf = search.best_estimator_
        logger.info(f"Best RF params: {search.best_params_}")
        return self.best_rf

    def train_xgboost(self) -> XGBClassifier:
        """
        Train XGBoost classifier (Version 2 parameters).

        Returns:
            Fitted XGBClassifier instance.
        """
        self.xgb = XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            learning_rate=self.config.xgb_learning_rate,
            max_depth=self.config.xgb_max_depth,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.xgb.fit(self.X_train, self.y_train)
        logger.info("XGBoost trained.")
        return self.xgb

    def cross_validate(self, model, X, y) -> list:
        """
        Run 5-fold cross-validation with ROC AUC scoring.

        Args:
            model: Any sklearn-compatible estimator.
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            List of ROC AUC scores per fold.
        """
        scores = cross_val_score(
            model, X, y, cv=5, scoring="roc_auc", n_jobs=-1
        )
        logger.info(f"CV ROC AUC scores: {scores}")
        logger.info(f"Mean CV ROC AUC: {scores.mean():.4f}")
        return scores

    def save_artifacts(self) -> None:
        """
        Save all trained models and the scaler to disk.
        """
        joblib.dump(self.scaler, self.models_dir / "scaler.pkl")
        joblib.dump(self.lr, self.models_dir / "logistic_regression.pkl")
        joblib.dump(self.rf, self.models_dir / "random_forest.pkl")
        joblib.dump(self.best_rf, self.models_dir / "best_random_forest.pkl")
        joblib.dump(self.xgb, self.models_dir / "xgboost.pkl")
        joblib.dump(self.best_rf, self.models_dir / "final_model.pkl")
        logger.info("All models and scaler saved.")

