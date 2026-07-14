# Model Evaluator - Phase 2

Comprehensive model evaluation and comparison tool for wine quality classification.

## Overview

The `ModelEvaluator` module provides a complete pipeline for loading trained models, evaluating their performance, comparing multiple models, and generating visualization and summary reports.

## Features

- ✅ Load trained models from disk (using joblib)
- ✅ Evaluate models with multiple metrics (accuracy, precision, recall, F1-score)
- ✅ Generate confusion matrices for each model
- ✅ Create ROC-AUC curves for model discrimination
- ✅ Compare all models side-by-side
- ✅ Select best model with justification
- ✅ Save visualizations as PNG files
- ✅ Generate comprehensive summary reports
- ✅ Support for multiclass classification

## Installation

No additional dependencies required. Uses existing project dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

### Quick Start

```python
from src.fase_2.model_evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(
    model_dir="models",
    test_data_path="data/processed/train.csv",
    output_dir="results/model_comparison"
)

# Load data and models
evaluator.load_test_data()
evaluator.load_models()

# Evaluate models
evaluator.evaluate_models()

# Generate visualizations
evaluator.plot_confusion_matrices()
evaluator.plot_roc_curves()
evaluator.plot_all_metrics_comparison()

# Compare models and select best
evaluator.compare_models()
report = evaluator.generate_summary_report()
```

### Running the Test Script

```bash
# Using uv
uv run python test_model_evaluation.py

# Using Python directly
python test_model_evaluation.py
```

### Using in Jupyter Notebook

```python
# Import the evaluator
from src.fase_2.model_evaluator import ModelEvaluator

# Initialize and run as shown above
evaluator = ModelEvaluator(
    model_dir="models",
    test_data_path="data/processed/train.csv",
    output_dir="results/model_comparison"
)
evaluator.load_test_data()
evaluator.load_models()
evaluator.evaluate_models()
evaluator.plot_confusion_matrices()
evaluator.plot_roc_curves()
evaluator.plot_all_metrics_comparison()
evaluator.compare_models()
report = evaluator.generate_summary_report()
```

## Module Reference

### Class: `ModelEvaluator`

#### Constructor

```python
ModelEvaluator(
    model_dir: str = "models",
    test_data_path: str = "data/processed/test.csv",
    output_dir: str = "results/model_comparison"
)
```

**Parameters:**
- `model_dir`: Directory containing trained model files (default: "models")
- `test_data_path`: Path to test data CSV file (default: "data/processed/test.csv")
- `output_dir`: Directory to save comparison plots and reports (default: "results/model_comparison")

#### Methods

##### `load_test_data()`
Load test data from CSV file. Returns `(X_test, y_test)`.

##### `load_models()`
Load all trained models from model directory. Returns dictionary of loaded models.

##### `evaluate_models()`
Evaluate all loaded models using test data. Returns dictionary of evaluation metrics.

##### `plot_confusion_matrices()`
Create confusion matrices for all models and save as PNG file.

##### `plot_roc_curves()`
Create ROC-AUC curves for all models and save as PNG file (uses One-vs-Rest strategy).

##### `plot_all_metrics_comparison()`
Create comparison bar charts for all metrics and save as PNG file.

##### `compare_models()`
Compare all models and select the best one based on F1-Score. Returns `(comparison_df, best_model)`.

##### `generate_summary_report()`
Generate a comprehensive summary report and save as Markdown file. Returns report as string.

## Output Files

### Visualizations

Generated in `results/model_comparison/`:

1. **`confusion_matrices.png`** - Confusion matrices for all models
2. **`roc_curves.png`** - ROC-AUC curves showing discrimination performance
3. **`metrics_comparison.png`** - Bar chart comparison of all metrics

### Summary Report

Saved in `results/model_comparison_summary.md`:

- Executive summary
- Model performance table
- Best model selection with justification
- Practical recommendations
- Next steps for deployment
- Detailed classification reports

## Model Performance Metrics

The evaluator calculates the following metrics for each model:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall

## Example Output

```
================================================================================
Model Comparison
================================================================================

Model Performance Summary:
--------------------------------------------------------------------------------
     Model  Accuracy  Precision   Recall  F1-Score
test_model  0.663808   0.665162 0.663808  0.659538
--------------------------------------------------------------------------------

🏆 Best Model: test_model
   F1-Score: 0.6595
   Accuracy: 0.6638
   Precision: 0.6652
   Recall: 0.6638
```

## Testing

Run the automated test script:

```bash
uv run python test_model_evaluation.py
```

Expected output:
1. Load test data (583 samples, 15 features)
2. Load trained model(s)
3. Evaluate each model with metrics
4. Generate 3 visualization plots
5. Compare models and select best
6. Save summary report

## Requirements

- Python 3.10+
- Project dependencies (pandas, numpy, scikit-learn, matplotlib, seaborn)

## Integration with Project

This module integrates seamlessly with:

- **ModelTrainer**: Used to train models (`src/fase_2/models.py`)
- **Preprocessor**: Used to preprocess data (`src/fase_2/preprocessor.py`)
- **Data Loader**: Used to load datasets (`src/fase_2/data_loader.py`)

## Troubleshooting

### No models found
- Ensure models are trained first using `ModelTrainer`
- Check that models are saved in the `models/` directory as `.joblib` files

### ROC-AUC error
- The module handles multiclass classification using One-vs-Rest strategy
- If error persists, ensure sklearn version >= 0.22

### Memory issues with large datasets
- The module uses standard sklearn metrics which are memory efficient
- Consider downsampling for very large datasets

## License

Part of Phase 2 Project - Machine Learning for Wine Quality Classification
