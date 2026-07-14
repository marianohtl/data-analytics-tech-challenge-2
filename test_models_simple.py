#!/usr/bin/env python3
"""
Simple test script for model training to verify basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.fase_2.models import ModelTrainer


def main():
    """Train and evaluate all models."""
    print("="*80)
    print("Wine Quality Classification - Model Training (Simple Test)")
    print("="*80)

    # Initialize ModelTrainer
    trainer = ModelTrainer(
        train_path="data/processed/train.csv",
        output_dir="models",
        results_dir="results",
    )

    # Load data
    print("\nLoading data...")
    trainer.load_and_preprocess_data(stratify=True, bin_target=True)

    # Train Logistic Regression (simplified)
    print("\n" + "="*80)
    print("Training Logistic Regression (simplified)...")
    print("="*80)

    # Use simpler parameter grid to avoid convergence issues
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "saga"],
        "max_iter": [200, 500],
    }

    model = LogisticRegression(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # Fewer folds
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(trainer.X_train, trainer.y_train)
    trainer.models["logistic_regression"] = grid_search.best_estimator_
    trainer.best_params["logistic_regression"] = grid_search.best_params_
    trainer.predictions["logistic_regression"] = grid_search.predict(trainer.X_test)
    trainer.metrics["logistic_regression"] = trainer._evaluate_model(
        trainer.predictions["logistic_regression"], "Logistic Regression"
    )

    print(f"\nBest parameters: {trainer.best_params['logistic_regression']}")
    print(f"\nTest metrics: {trainer.metrics['logistic_regression']}")

    # Train Random Forest
    print("\n" + "="*80)
    print("Training Random Forest...")
    print("="*80)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt"],
        "bootstrap": [True],
    }

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,  # Fewer iterations
        cv=3,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(trainer.X_train, trainer.y_train)
    trainer.models["random_forest"] = random_search.best_estimator_
    trainer.best_params["random_forest"] = random_search.best_params_
    trainer.predictions["random_forest"] = random_search.predict(trainer.X_test)
    trainer.metrics["random_forest"] = trainer._evaluate_model(
        trainer.predictions["random_forest"], "Random Forest"
    )

    print(f"\nBest parameters: {trainer.best_params['random_forest']}")
    print(f"\nTest metrics: {trainer.metrics['random_forest']}")

    # Train LightGBM
    print("\n" + "="*80)
    print("Training LightGBM...")
    print("="*80)

    import lightgbm as lgb

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "learning_rate": [0.1, 0.2],
        "subsample": [0.8, 1.0],
    }

    model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(trainer.X_train, trainer.y_train)
    trainer.models["lightgbm"] = random_search.best_estimator_
    trainer.best_params["lightgbm"] = random_search.best_params_
    trainer.predictions["lightgbm"] = random_search.predict(trainer.X_test)
    trainer.metrics["lightgbm"] = trainer._evaluate_model(
        trainer.predictions["lightgbm"], "LightGBM"
    )

    print(f"\nBest parameters: {trainer.best_params['lightgbm']}")
    print(f"\nTest metrics: {trainer.metrics['lightgbm']}")

    # Compare and save
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)

    for model_name, metrics in trainer.metrics.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

    # Save models
    print("\n" + "="*80)
    print("Saving models...")
    print("="*80)

    for model_name, model in trainer.models.items():
        import joblib
        model_path = trainer.output_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")

    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)

    import json
    from pathlib import Path

    results = {
        "best_params": trainer.best_params,
        "model_metrics": {},
        "predictions": {},
    }

    for model_name, metrics in trainer.metrics.items():
        results["model_metrics"][model_name] = {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
            "confusion_matrix": metrics["confusion_matrix"],
        }

    for model_name, predictions in trainer.predictions.items():
        results["predictions"][model_name] = predictions.tolist()

    results_path = trainer.results_dir / "model_performance.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {results_path}")

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
