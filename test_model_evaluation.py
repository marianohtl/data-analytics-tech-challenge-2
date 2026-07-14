#!/usr/bin/env python3
"""
Test script for ModelEvaluator.

This script demonstrates how to use the ModelEvaluator class to:
1. Load trained models
2. Evaluate all models
3. Generate comparison metrics and plots
4. Select best model and print recommendation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.fase_2.model_evaluator import ModelEvaluator


def main():
    """Demonstrate model evaluation and comparison."""
    print("=" * 80)
    print("Model Evaluation Test")
    print("=" * 80)

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_dir="models",
        test_data_path="data/processed/train.csv",  # Using train.csv for evaluation
        output_dir="results/model_comparison",
    )

    # Step 1: Load test data
    print("\n[Step 1/5] Loading test data...")
    X_test, y_test = evaluator.load_test_data()
    print(f"✓ Loaded {X_test.shape[0]} samples with {X_test.shape[1]} features")

    # Step 2: Load models
    print("\n[Step 2/5] Loading trained models...")
    models = evaluator.load_models()
    print(f"✓ Loaded {len(models)} model(s)")

    # Step 3: Evaluate models
    print("\n[Step 3/5] Evaluating models...")
    metrics = evaluator.evaluate_models()
    print(f"✓ Generated evaluation metrics for {len(metrics)} model(s)")

    # Step 4: Create visualizations
    print("\n[Step 4/5] Creating visualizations...")
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_all_metrics_comparison()
    print("✓ Saved all visualization plots")

    # Step 5: Compare models and generate summary
    print("\n[Step 5/5] Comparing models and generating summary...")
    comparison_df, best_model = evaluator.compare_models()
    report = evaluator.generate_summary_report()

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print(f"\n📊 Generated Files:")
    print(f"   - Visualizations: {evaluator.output_dir}/")
    print(f"   - Summary Report: {evaluator.output_dir.parent}/model_comparison_summary.md")
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")


def test_with_existing_test_file():
    """Test with existing test data file (if available)."""
    print("\n" + "=" * 80)
    print("Testing with existing test data...")
    print("=" * 80)

    # Check if test.csv exists
    test_path = Path("data/processed/test.csv")
    if not test_path.exists():
        print(f"⚠️  Test file not found: {test_path}")
        print("   Using train.csv instead.")
        evaluator = ModelEvaluator(
            model_dir="models",
            test_data_path="data/processed/train.csv",
            output_dir="results/model_comparison",
        )
    else:
        evaluator = ModelEvaluator(
            model_dir="models",
            test_data_path="data/processed/test.csv",
            output_dir="results/model_comparison",
        )

    # Run evaluation
    evaluator.load_test_data()
    evaluator.load_models()
    evaluator.evaluate_models()
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_all_metrics_comparison()
    evaluator.compare_models()
    evaluator.generate_summary_report()


if __name__ == "__main__":
    main()

    # Uncomment below to test with existing test file
    # test_with_existing_test_file()
