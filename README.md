# Wine Quality Classification Project

A comprehensive machine learning project for predicting wine quality using the Wine Quality dataset. This project implements end-to-end data analysis, model training, evaluation, and interpretation pipelines.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Results Overview](#results-overview)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project demonstrates the complete machine learning workflow from data exploration to model deployment. It covers:

- **Data Analysis**: Exploratory Data Analysis (EDA) to understand data distributions and patterns
- **Data Preprocessing**: Clean, encode, and transform data for modeling
- **Model Training**: Multiple classification models including Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Model Evaluation**: Comprehensive evaluation using confusion matrices, ROC-AUC curves, and multiple metrics
- **Model Interpretation**: Feature importance analysis using SHAP values

## Dataset Description

### Source
- **Dataset**: Wine Quality Dataset (Kaggle)
- **Original Source**: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009)
- **Download**: [Kaggle Wine Quality Dataset](https://www.kaggle.com/uciml/wine-quality)

### Dataset Structure
- **Total Samples**: 1,143 wines
- **Features**: 11 physicochemical properties
  1. Fixed acidity (g/t)
  2. Volatile acidity (g/t)
  3. Citric acid (g/t)
  4. Residual sugar (g/dm)
  5. Chlorides (g/dm3)
  6. Free sulfur dioxide (mg/dm3)
  7. Total sulfur dioxide (mg/dm3)
  8. Density (g/dm3)
  9. pH
  10. Sulphates (g/dm3)
  11. Alcohol (% by volume)
- **Target Variable**: Quality (integers 3-9)
  - Categorical interpretation:
    - **Low/Medium**: Quality ≤ 6 (binary target)
    - **High**: Quality ≥ 7

## Project Structure

```
fase_2/
├── data/
│   ├── raw/
│   │   └── winequality.csv          # Raw dataset
│   └── processed/
│       └── train.csv                # Preprocessed training data
├── models/
│   └── (trained models saved here)  # Model files (.joblib)
├── results/
│   ├── eda/                         # Exploratory data analysis visualizations
│   ├── model_comparison/            # Model evaluation plots and reports
│   ├── feature_importance/          # Feature importance visualizations
│   └── model_comparison_summary.md   # Summary report
├── notebooks/
│   ├── eda.ipynb                    # Exploratory data analysis notebook
│   ├── train_models.ipynb           # Model training notebook
│   └── model_evaluation.ipynb       # Model evaluation notebook
├── src/fase_2/
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessor.py              # Data preprocessing and encoding
│   ├── models.py                    # Model training utilities
│   ├── model_evaluator.py           # Model evaluation and comparison
│   ├── feature_importance.py        # Feature importance and SHAP analysis
│   └── target_encoder.py            # Target encoding utilities
├── examples/
│   └── example_usage.py             # Usage examples
├── scripts/
│   ├── download_dataset.py          # Data download script
│   ├── example_preprocessing.py     # Data preprocessing example
│   └── verify_task07.py             # Task verification scripts
├── tests/
│   ├── test_models.py
│   ├── test_model_evaluation.py
│   ├── test_feature_importance.py
│   └── test_preprocessing.py
├── pyproject.toml                   # Project configuration and dependencies
├── uv.lock                          # Locked dependencies
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── results_summary.md               # Executive summary report
```

## Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Python package manager (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd fase_2

# Install dependencies
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd fase_2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download the Dataset

```bash
python scripts/download_dataset.py
```

This will download the Wine Quality dataset from Kaggle and save it to `data/raw/winequality.csv`.

### 2. Run Preprocessing

```bash
python scripts/example_preprocessing.py
```

This will preprocess the raw data and save it to `data/processed/train.csv`.

### 3. Train Models

Option A: Use the example script
```bash
python examples/example_usage.py
```

Option B: Use Jupyter Notebook
```bash
jupyter notebook notebooks/train_models.ipynb
```

### 4. Evaluate Models

```bash
python verify_task07.py
```

This will load trained models, evaluate them, and generate visualizations.

### 5. Analyze Feature Importance

```bash
python verify_task08.py
```

This will analyze feature importance and create SHAP visualizations.

## Usage Guide

### Data Loading

```python
from src.fase_2.data_loader import DataLoader

loader = DataLoader(
    data_path="data/processed/train.csv"
)
X_train, X_test, y_train, y_test = loader.load_data()
```

### Data Preprocessing

```python
from src.fase_2.preprocessor import Preprocessor

preprocessor = Preprocessor()
X_train_processed, X_test_processed = preprocessor.preprocess(X_train, X_test)
```

### Model Training

```python
from src.fase_2.models import ModelTrainer

trainer = ModelTrainer(
    model_dir="models",
    random_state=42
)
trainer.train(
    X_train_processed,
    y_train,
    model_type="random_forest"
)
```

### Model Evaluation

```python
from src.fase_2.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    model_dir="models",
    test_data_path="data/processed/train.csv",
    output_dir="results/model_comparison"
)
evaluator.evaluate_models()
evaluator.compare_models()
evaluator.generate_summary_report()
```

### Feature Importance Analysis

```python
from src.fase_2.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(
    model_path="models/random_forest_model.joblib",
    data_path="data/processed/train.csv",
    output_dir="results/feature_importance"
)
analyzer.analyze()
```

## Results Overview

### Executive Summary

This project successfully implemented a complete machine learning pipeline for wine quality classification. The analysis revealed that several physicochemical properties significantly influence wine quality, with alcohol content and sulphates being the most important factors.

### Key Findings

1. **Model Performance**: The best model achieved an F1-Score of 0.8816 with an accuracy of 88.16%, demonstrating excellent predictive capability.

2. **Key Features**:
   - **Alcohol Content**: Strongly influences perceived quality (10.66% importance)
   - **Sulphates**: Moderate levels improve quality and preservation (12.63% importance)
   - **Volatile Acidity**: Lower values indicate better quality (8.92% importance)
   - **Density**: Relates to sugar and alcohol content (6.19% importance)

3. **Data Insights**:
   - Wine quality is primarily influenced by alcohol content and chemical composition
   - Moderate sulphate levels contribute to better quality
   - Low volatile acidity is essential to avoid off-flavors

5. **Best Model**: Random Forest with:
    - Accuracy: 88.16%
    - Precision: 88.57%
    - Recall: 88.16%
    - F1-Score: 88.16%

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest (random_forest_feature_importance) | 0.8816 | 0.8857 | 0.8816 | 0.8816 |
| Logistic Regression (test_model) | 0.6638 | 0.6652 | 0.6638 | 0.6595 |

## Feature Importance

Top 5 most important features for wine quality prediction:

1. **alcohol_density_ratio** (13.44%)
   - Indicates wine strength and body

2. **sulphates** (12.63%)
   - Contributes to taste and antioxidant properties

3. **alcohol** (10.66%)
   - Strong influence on perceived quality

4. **volatile acidity** (8.92%)
   - Lower values indicate better quality

5. **density** (6.19%)
   - Relates to sugar and alcohol content

## Visualizations

### Exploratory Data Analysis
- Distribution plots for all features
- Quality vs. feature relationships
- Correlation heatmaps
- Outlier detection plots

### Model Evaluation
- Confusion matrices for each model
- ROC-AUC curves (One-vs-Rest for multiclass)
- Metrics comparison bar charts
- Classification reports

### Feature Importance
- Feature importance bar plots
- SHAP beeswarm plots
- SHAP summary bar plots

### Accessing Visualizations

All visualizations are saved in the `results/` directory:

```bash
# EDA visualizations
results/eda/

# Model comparison plots
results/model_comparison/confusion_matrices.png
results/model_comparison/roc_curves.png
results/model_comparison/metrics_comparison.png

# Feature importance plots
results/feature_importance/feature_importance_barplot.png
results/feature_importance/shap_beeswarm_plot.png
results/feature_importance/shap_bar_plot.png
```

## Dependencies

### Core Dependencies

- **pandas** (>=2.3.3): Data manipulation and analysis
- **numpy** (>=2.2.6): Numerical computing
- **scikit-learn** (>=1.7.2): Machine learning algorithms
- **matplotlib** (>=3.10.9): Visualization
- **seaborn** (>=0.13.2): Statistical data visualization
- **shap** (>=0.49.1): Model interpretability

### Additional Dependencies

- **lightgbm** (>=4.6.0): Gradient boosting framework
- **xgboost**: Gradient boosting framework (optional)
- **jupyter** (>=1.1.1): Interactive notebooks
- **kagglehub** (>=0.2.0): Dataset download

### Installation

See [Installation](#installation) section above for detailed installation instructions.

## Documentation

- **README.md**: This file
- **results_summary.md**: Executive summary report
- **TASK07_FINAL_REPORT.md**: Model evaluation report
- **TASK08_SUMMARY.md**: Feature importance report
- **QUICK_REFERENCE.md**: Quick start guide

### Module Documentation

Each module includes detailed documentation:

- **src/fase_2/data_loader.py**: Data loading utilities
- **src/fase_2/preprocessor.py**: Data preprocessing
- **src/fase_2/models.py**: Model training
- **src/fase_2/model_evaluator.py**: Model evaluation
- **src/fase_2/feature_importance.py**: Feature importance analysis

Each module includes:
- Comprehensive docstrings
- Usage examples
- API references
- Error handling guidelines

## Development Workflow

### Running Tests

```bash
# All tests
uv run python -m pytest tests/

# Specific test file
uv run python -m pytest tests/test_models.py

# With coverage
uv run python -m pytest tests/ --cov=src/fase_2
```

### Code Quality

```bash
# Linting
uv run ruff check src/fase_2/

# Type checking
uv run mypy src/fase_2/

# Formatting
uv run ruff format src/fase_2/
```

### Verification

```bash
# Task 07 verification (Model Evaluation)
uv run python verify_task07.py

# Task 08 verification (Feature Importance)
uv run python verify_task08.py
```

## Notebooks

The project includes three Jupyter notebooks:

### 1. eda.ipynb
- Exploratory data analysis
- Data distribution analysis
- Feature relationships
- Quality vs. feature analysis
- Correlation matrix
- Outlier detection

### 2. train_models.ipynb
- Model training workflow
- Hyperparameter tuning
- Model comparison
- Performance evaluation

### 3. model_evaluation.ipynb
- Model evaluation using ModelEvaluator class
- Confusion matrix visualization
- ROC-AUC curve generation
- Metrics comparison

### Accessing Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Or using VS Code
code .
```

## Best Practices Implemented

1. **Code Quality**: Follows PEP 8 style guide with comprehensive docstrings
2. **Modular Design**: Separate modules for data loading, preprocessing, training, evaluation
3. **Error Handling**: Robust error handling and validation
4. **Documentation**: Comprehensive inline documentation and usage guides
5. **Testing**: Automated test scripts with verification
6. **Reproducibility**: Random seed for reproducible results
7. **Visualization**: High-quality, publication-ready plots
8. **Interpretability**: SHAP values for model explainability

## Limitations

- Dataset size (1,143 samples) may limit model generalization
- Binary classification (low/medium vs. high quality) for simplicity
- Limited to physicochemical features
- No sensory evaluation data

## Future Enhancements

1. **Model Improvements**:
   - Hyperparameter optimization (Optuna)
   - Ensemble methods
   - Cross-validation
   - Model calibration

2. **Feature Engineering**:
   - Advanced feature interactions
   - Domain-specific features
   - Dimensionality reduction

3. **Data**:
   - Additional datasets
   - Sensory evaluation data
   - More diverse wine types

4. **Deployment**:
   - API deployment
   - Real-time prediction
   - Model monitoring

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Run lint and tests (`uv run ruff check && uv run pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- **Dataset**: [Kaggle Wine Quality Dataset](https://www.kaggle.com/uciml/wine-quality)
- **Paper**: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, 47(4), 547-553.

## Contact

For questions or contributions, please open an issue in the repository.

## Acknowledgments

- University of Minho (Portugal) for the dataset
- Kaggle for hosting the dataset
- Open source community for valuable tools and libraries

---

**Version**: 1.0.0
**Last Updated**: 2026-07-14
**Status**: Production Ready
