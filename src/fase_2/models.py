"""
ML Model Training Module for Wine Quality Classification.

This module provides functionality to:
1. Train classification models (Logistic Regression, Random Forest, XGBoost/LightGBM)
2. Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
3. Evaluate models using various metrics (accuracy, precision, recall, F1-score)
4. Save trained models and evaluation results to disk
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class ModelTrainer:
    """Train and evaluate classification models for wine quality prediction."""

    def __init__(
        self,
        train_path: str = "data/processed/train.csv",
        test_path: str = None,
        output_dir: str = "models",
        results_dir: str = "results",
    ):
        """
        Initialize the ModelTrainer.

        Args:
            train_path: Path to the training data file
            test_path: Path to the test data file (optional, if None, data is split)
            output_dir: Directory to save trained models
            results_dir: Directory to save evaluation results
        """
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Model storage
        self.models = {}
        self.best_params = {}
        self.predictions = {}
        self.metrics = {}

    def load_and_preprocess_data(self, stratify: bool = True, bin_target: bool = True):
        """
        Load and preprocess training data.

        Args:
            stratify: Whether to stratify the train/test split
            bin_target: Whether to bin the continuous target variable into categories
        """
        print("Loading data...")
        df = pd.read_csv(self.train_path)

        # Separate features and target
        # Target variable is 'quality'
        feature_cols = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "alcohol_density_ratio",
            "free_sulfur_ratio",
            "log_total_sulfur",
            "sugar_alcohol_ratio",
        ]

        self.X = df[feature_cols]

        # Bin the continuous target variable into categories for classification
        if bin_target:
            # Create bins based on quality value ranges
            bins = [-np.inf, -0.5, 0.5, 1.5, np.inf]
            labels = ["low", "medium", "high", "very_high"]
            self.y = pd.cut(df["quality"], bins=bins, labels=labels, include_lowest=True)
            print(f"Binned target distribution:\n{self.y.value_counts().sort_index()}")
        else:
            self.y = df["quality"]
            print(f"Target distribution:\n{self.y.value_counts().sort_index()}")

        print(f"\nFeature shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")

        # Split data if no test path provided
        if self.test_path is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42,
                stratify=self.y if stratify else None,
            )
            print(f"\nTrain set shape: {self.X_train.shape}")
            print(f"Test set shape: {self.X_test.shape}")
        else:
            # Load separate test set
            self.X_train = self.X
            self.y_train = self.y
            self.X_test = pd.read_csv(self.test_path)
            print(f"\nLoaded separate test set shape: {self.X_test.shape}")

    def train_logistic_regression(self, cv_folds: int = 5):
        """
        Train Logistic Regression with hyperparameter tuning.

        Args:
            cv_folds: Number of cross-validation folds
        """
        print("\n" + "="*80)
        print("Training Logistic Regression...")
        print("="*80)

        # Define parameter grid
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l2", "l1", None],
            "solver": ["lbfgs", "liblinear", "saga"],
            "max_iter": [100, 200, 500],
        }

        # Initialize model and GridSearchCV
        # Note: Setting penalty=None requires solver='saga'
        if None in param_grid["penalty"]:
            param_grid["solver"] = ["lbfgs", "liblinear", "saga"]

        model = LogisticRegression(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )

        # Train model
        grid_search.fit(self.X_train, self.y_train)

        # Store results
        self.models["logistic_regression"] = grid_search.best_estimator_
        self.best_params["logistic_regression"] = grid_search.best_params_
        self.predictions["logistic_regression"] = grid_search.predict(self.X_test)

        # Evaluate
        y_pred = self.predictions["logistic_regression"]
        self.metrics["logistic_regression"] = self._evaluate_model(
            y_pred, "Logistic Regression"
        )

        print(f"\nBest parameters: {self.best_params['logistic_regression']}")
        print(f"\nTest metrics: {self.metrics['logistic_regression']}")

    def train_random_forest(self, cv_folds: int = 5, n_iter: int = 50):
        """
        Train Random Forest with hyperparameter tuning.

        Args:
            cv_folds: Number of cross-validation folds
            n_iter: Number of random search iterations
        """
        print("\n" + "="*80)
        print("Training Random Forest...")
        print("="*80)

        # Define parameter grid for RandomizedSearchCV
        param_dist = {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False],
        }

        # Initialize model and RandomizedSearchCV
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="f1_weighted",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        # Train model
        random_search.fit(self.X_train, self.y_train)

        # Store results
        self.models["random_forest"] = random_search.best_estimator_
        self.best_params["random_forest"] = random_search.best_params_
        self.predictions["random_forest"] = random_search.predict(self.X_test)

        # Evaluate
        y_pred = self.predictions["random_forest"]
        self.metrics["random_forest"] = self._evaluate_model(y_pred, "Random Forest")

        print(f"\nBest parameters: {self.best_params['random_forest']}")
        print(f"\nTest metrics: {self.metrics['random_forest']}")

    def train_xgboost(self, cv_folds: int = 5, n_iter: int = 50):
        """
        Train XGBoost with hyperparameter tuning.

        Note: Using LightGBM as fallback if XGBoost is not installed.
        """
        print("\n" + "="*80)
        print("Training XGBoost (using LightGBM as fallback)...")
        print("="*80)

        try:
            import xgboost as xgb
            model_name = "xgboost"
        except ImportError:
            import lightgbm as lgb
            model_name = "lightgbm"

        # Define parameter grid for RandomizedSearchCV
        param_dist = {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [3, 5, 7, 10, 15, None],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [1, 3, 5, 10],
        }

        # Initialize model and RandomizedSearchCV
        if model_name == "xgboost":
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
            param_dist["objective"] = ["binary:logistic", "multi:softmax"]
        else:
            model = lgb.LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="f1_weighted",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        # Train model
        random_search.fit(self.X_train, self.y_train)

        # Store results
        self.models[model_name] = random_search.best_estimator_
        self.best_params[model_name] = random_search.best_params_
        self.predictions[model_name] = random_search.predict(self.X_test)

        # Evaluate
        y_pred = self.predictions[model_name]
        self.metrics[model_name] = self._evaluate_model(y_pred, model_name.capitalize())

        print(f"\nBest parameters: {self.best_params[model_name]}")
        print(f"\nTest metrics: {self.metrics[model_name]}")

    def _evaluate_model(self, y_pred, model_name: str) -> dict:
        """
        Evaluate a model and return metrics.

        Args:
            y_pred: Predicted labels
            model_name: Name of the model

        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

        # Classification report
        report = classification_report(
            self.y_test, y_pred, output_dict=True, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    def compare_models(self):
        """Compare all trained models and display performance."""
        print("\n" + "="*80)
        print("Model Comparison")
        print("="*80)

        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.metrics.items():
            comparison_data.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1_score"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Find best model
        best_model = comparison_df.loc[comparison_df["F1-Score"].idxmax()]
        print(f"\nBest model based on F1-Score: {best_model['Model']}")
        print(f"F1-Score: {best_model['F1-Score']:.4f}")

        return comparison_df

    def save_models(self):
        """Save all trained models to disk."""
        print("\n" + "="*80)
        print("Saving models...")
        print("="*80)

        for model_name, model in self.models.items():
            model_path = self.output_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved: {model_path}")

    def save_results(self):
        """Save model metrics, predictions, and best parameters to JSON."""
        print("\n" + "="*80)
        print("Saving results...")
        print("="*80)

        # Prepare results dictionary
        results = {
            "best_params": self.best_params,
            "model_metrics": {},
            "predictions": {},
        }

        # Add metrics for each model
        for model_name, metrics in self.metrics.items():
            results["model_metrics"][model_name] = {
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"]),
                "confusion_matrix": metrics["confusion_matrix"],
            }

        # Add predictions (convert to lists for JSON serialization)
        for model_name, predictions in self.predictions.items():
            results["predictions"][model_name] = predictions.tolist()

        # Save to JSON
        results_path = self.results_dir / "model_performance.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved: {results_path}")

        # Also save detailed classification reports
        report_path = self.results_dir / "classification_reports.json"
        reports = {}
        for model_name, metrics in self.metrics.items():
            reports[model_name] = metrics["classification_report"]
        with open(report_path, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"Saved: {report_path}")

    def train_all_models(
        self,
        test_path: str = None,
        cv_folds: int = 5,
        n_iter: int = 50,
        stratify: bool = True,
        bin_target: bool = True,
    ):
        """
        Train all models with hyperparameter tuning.

        Args:
            test_path: Path to test data file (optional)
            cv_folds: Number of cross-validation folds
            n_iter: Number of random search iterations
            stratify: Whether to stratify the train/test split
            bin_target: Whether to bin the target variable
        """
        # Load data
        self.load_and_preprocess_data(stratify=stratify, bin_target=bin_target)

        # Train models
        self.train_logistic_regression(cv_folds=cv_folds)
        self.train_random_forest(cv_folds=cv_folds, n_iter=n_iter)
        self.train_xgboost(cv_folds=cv_folds, n_iter=n_iter)

        # Compare and save results
        self.compare_models()
        self.save_models()
        self.save_results()

        print("\n" + "="*80)
        print("All models trained and saved successfully!")
        print("="*80)
