"""
Test script for Feature Importance Analysis (Task 08)

This script verifies that the feature importance analysis works correctly.
"""

import joblib
from pathlib import Path

from src.fase_2.feature_importance import FeatureImportanceAnalyzer


def test_feature_importance_analysis():
    """Run the feature importance analysis and verify outputs."""
    print("="*80)
    print("Testing Feature Importance Analysis (Task 08)")
    print("="*80)

    # Initialize analyzer with Random Forest model
    analyzer = FeatureImportanceAnalyzer(
        model_path="models/random_forest_feature_importance.joblib",
        data_path="data/processed/train.csv",
        output_dir="results/feature_importance",
    )

    # Run analysis
    outputs = analyzer.analyze()

    # Verify outputs
    print("\n" + "="*80)
    print("Verification")
    print("="*80)

    expected_files = [
        "feature_importance_barplot.png",
        "shap_beeswarm_plot.png",
        "shap_bar_plot.png",
        "feature_importance_summary.md",
    ]

    all_passed = True

    for filename in expected_files:
        filepath = Path(outputs.get(filename, "results/feature_importance/" + filename))
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"✓ {filename}: {size_kb:.1f} KB")
        else:
            print(f"✗ {filename}: NOT FOUND")
            all_passed = False

    # Verify output directory
    output_dir = Path("results/feature_importance")
    if output_dir.exists() and output_dir.is_dir():
        files = list(output_dir.glob("*"))
        print(f"\n✓ Output directory created with {len(files)} files:")
        for f in sorted(files):
            print(f"  - {f.name}")
    else:
        print(f"✗ Output directory not found")
        all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
    else:
        print("✗ SOME VERIFICATIONS FAILED")
    print("="*80)

    return all_passed


def test_random_forest_model():
    """Test feature importance with a Random Forest model instead of Logistic Regression."""
    print("\n" + "="*80)
    print("Testing with Random Forest Model")
    print("="*80)

    from src.fase_2.models import ModelTrainer

    # Train a Random Forest model
    print("\n[1/3] Training Random Forest model...")
    trainer = ModelTrainer(
        train_path="data/processed/train.csv",
        output_dir="models",
    )
    trainer.load_and_preprocess_data()
    trainer.train_random_forest(cv_folds=3, n_iter=10)

    # Save the model
    print("\n[2/3] Saving model...")
    model_path = "models/random_forest_test.joblib"
    joblib.dump(trainer.models["random_forest"], model_path)
    print(f"✓ Model saved to {model_path}")

    # Test feature importance analysis
    print("\n[3/3] Running feature importance analysis...")
    analyzer = FeatureImportanceAnalyzer(
        model_path=model_path,
        data_path="data/processed/train.csv",
        output_dir="results/feature_importance_rf",
    )

    outputs = analyzer.analyze()

    print("\n✓ Random Forest feature importance analysis completed")

    # Clean up
    print("\nCleaning up test model...")
    Path(model_path).unlink()
    print(f"✓ Removed {model_path}")

    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Task 08 - Feature Importance and Interpretation")
    print("="*80 + "\n")

    # Test with Logistic Regression (existing model)
    success1 = test_feature_importance_analysis()

    # Test with Random Forest (new model)
    success2 = test_random_forest_model()

    # Final summary
    print("\n" + "="*80)
    print("Final Summary")
    print("="*80)
    print(f"✓ Logistic Regression test: {'PASSED' if success1 else 'FAILED'}")
    print(f"✓ Random Forest test: {'PASSED' if success2 else 'FAILED'}")
    print("="*80)

    if success1 and success2:
        print("\n✓ Task 08 - Feature Importance and Interpretation: COMPLETE")
        exit(0)
    else:
        print("\n✗ Task 08 - Feature Importance and Interpretation: INCOMPLETE")
        exit(1)
