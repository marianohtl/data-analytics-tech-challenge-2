# Model Training Module

This module provides comprehensive machine learning model training, evaluation, and serialization functionality for wine quality classification.

## Quick Start

```python
from src.fase_2.models import ModelTrainer

# Initialize and train
trainer = ModelTrainer(
    train_path="data/processed/train.csv",
    output_dir="models",
    results_dir="results"
)
trainer.train_all_models(cv_folds=5, n_iter=50)

# Or use the test scripts
# uv run python test_models.py
# uv run python test_models_simple.py
```

## Features

### Three Classification Models
1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble tree-based model
3. **LightGBM/XGBoost** - Gradient boosting (auto-fallback)

### Hyperparameter Tuning
- GridSearchCV for Logistic Regression (exhaustive)
- RandomizedSearchCV for Random Forest and XGBoost (efficient)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score (weighted)
- Classification report with per-class metrics
- Confusion matrix

## Usage

### Basic Training
```python
trainer = ModelTrainer(
    train_path="data/processed/train.csv",
    output_dir="models",
    results_dir="results"
)

# Train all models with default parameters
trainer.train_all_models(
    cv_folds=5,
    n_iter=50,
    stratify=True,
    bin_target=True
)
```

### Individual Model Training
```python
# Load data
trainer.load_and_preprocess_data(stratify=True, bin_target=True)

# Train Logistic Regression
trainer.train_logistic_regression(cv_folds=5)

# Train Random Forest
trainer.train_random_forest(cv_folds=5, n_iter=50)

# Train XGBoost/LightGBM
trainer.train_xgboost(cv_folds=5, n_iter=50)
```

### Model Comparison
```python
# Compare all trained models
comparison = trainer.compare_models()

# Best model based on F1-score
best_model = comparison['Model'].iloc[comparison['F1-Score'].idxmax()]
```

## Output

### Saved Models
```bash
models/
├── logistic_regression.joblib
├── random_forest.joblib
└── lightgbm.joblib
```

### Results Files
```bash
results/
├── model_performance.json
└── classification_reports.json
```

## Model Performance

Typical performance ranges:
- Logistic Regression: ~65-70%
- Random Forest: ~70-75%
- LightGBM/XGBoost: ~72-78%

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- lightgbm
- joblib

## License

Internal project use only.
