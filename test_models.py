#!/usr/bin/env python3
"""
Test script for model training and evaluation.

This script demonstrates how to use the ModelTrainer class to:
1. Train multiple classification models
2. Perform hyperparameter tuning
3. Evaluate model performance
4. Save results and models
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
    print("Wine Quality Classification - Model Training")
    print("="*80)

    # Initialize ModelTrainer
    trainer = ModelTrainer(
        train_path="data/processed/train.csv",
        output_dir="models",
        results_dir="results",
    )

    # Train all models with hyperparameter tuning
    trainer.train_all_models(
        test_path=None,  # Will split from training data
        cv_folds=5,      # Cross-validation folds
        n_iter=50,       # Random search iterations
        stratify=True,   # Stratified split
    )

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


def test_with_existing_test_data():
    """Test with existing test data (if available)."""
    print("\n" + "="*80)
    print("Testing with existing test data...")
    print("="*80)

    # Initialize ModelTrainer
    trainer = ModelTrainer(
        train_path="data/processed/train.csv",
        test_path="data/processed/validation.csv",  # Assuming validation.csv exists
        output_dir="models",
        results_dir="results",
    )

    # Train all models
    trainer.train_all_models(
        test_path="data/processed/validation.csv",
        cv_folds=5,
        n_iter=50,
        stratify=True,
    )


if __name__ == "__main__":
    # Train models using training data only (splits into train/test)
    main()

    # Uncomment below to test with existing test data
    # test_with_existing_test_data()
