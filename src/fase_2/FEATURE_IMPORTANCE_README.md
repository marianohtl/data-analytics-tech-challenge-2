# Feature Importance and Interpretation Module

This module provides comprehensive feature importance analysis for wine quality classification models.

## Overview

The `feature_importance.py` module extracts and interprets feature importance from trained machine learning models, with a focus on tree-based models (Random Forest, XGBoost, LightGBM). It uses both model-based feature importance scores and SHAP (SHapley Additive exPlanations) values for interpretability.

## Key Features

### 1. Feature Importance Extraction
- Extracts feature importance from tree-based models using built-in `feature_importances_` attribute
- Handles Random Forest, XGBoost, and LightGBM models
- Returns ranked DataFrame with feature importance scores

### 2. SHAP Values Calculation
- Calculates SHAP values for model interpretability
- Uses `shap.TreeExplainer` for optimal performance with tree-based models
- Supports both multi-class and binary classification
- Provides local and global interpretability

### 3. Visualization
- **Feature Importance Bar Plot**: Horizontal bar chart showing top features
- **SHAP Beeswarm Plot**: Detailed summary plot showing feature impact on predictions
- **SHAP Bar Plot**: Mean absolute SHAP values for feature importance
- All plots are saved as high-resolution PNG files (300 DPI)

### 4. Comprehensive Reporting
- Generates Markdown summary report with:
  - Top features table with importance scores
  - Detailed interpretation of each feature
  - Practical implications for winemakers
  - Methodology and limitations
  - Recommendations

## Installation

Dependencies are already included in the project's `pyproject.toml`:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Ensure dependencies are installed
pip install -e .
```

Required packages:
- `joblib` - Model serialization
- `matplotlib` - Plotting
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning models
- `shap` - SHAP values and explainability

## Usage

### Basic Usage

```python
from src.fase_2.feature_importance import FeatureImportanceAnalyzer

# Initialize analyzer
analyzer = FeatureImportanceAnalyzer(
    model_path="models/random_forest_model.joblib",
    data_path="data/processed/train.csv",
    output_dir="results/feature_importance"
)

# Run complete analysis
outputs = analyzer.analyze()
```

### Customization

```python
from src.fase_2.feature_importance import FeatureImportanceAnalyzer

# Initialize with custom settings
analyzer = FeatureImportanceAnalyzer(
    model_path="models/my_model.joblib",
    data_path="data/processed/train.csv",
    output_dir="results/custom_feature_importance"
)

# Adjust number of top features displayed
analyzer.plot_feature_importance(importance_df, max_features=20)
```

### Programmatic Access

```python
# Load model and data
analyzer.load_data()
analyzer.load_model()

# Extract feature importance
importance_df = analyzer.extract_feature_importance()

# Calculate SHAP values
shap_values, explainer = analyzer.calculate_shap_values(analyzer.X_test)

# Create individual plots
analyzer.plot_feature_importance(importance_df)
analyzer.plot_shap_beeswarm(shap_values)
analyzer.plot_shap_bar(shap_values)

# Generate report
analyzer.generate_summary_report(importance_df, top_features=10)
```

## Output Files

### Visualizations
- `feature_importance_barplot.png` - Feature importance bar chart
- `shap_beeswarm_plot.png` - SHAP summary plot (beeswarm)
- `shap_bar_plot.png` - SHAP bar plot (mean absolute values)

### Reports
- `feature_importance_summary.md` - Comprehensive markdown report

## Model Requirements

### For Built-in Feature Importance
- Random Forest (`sklearn.ensemble.RandomForestClassifier`)
- Gradient Boosting Machines
- XGBoost (`xgboost.XGBClassifier`)
- LightGBM (`lightgbm.LGBMClassifier`)

### For SHAP Analysis
- Tree-based models (recommended for best performance)
- Logistic Regression and other linear models (uses TreeExplainer with limitations)

## Understanding Feature Importance

### Top Features for Wine Quality

Based on Random Forest analysis:

1. **alcohol_density_ratio** (13.44%) - Ratio of alcohol to density, indicates wine strength and body
2. **sulphates** (12.63%) - Sulphates contribute to taste and act as antioxidants
3. **alcohol** (10.66%) - Alcohol content strongly influences perceived quality
4. **volatile acidity** (8.92%) - Lower values indicate better quality
5. **density** (6.19%) - Relates to sugar content and alcohol concentration

### SHAP Values
- **Positive SHAP value**: Feature pushes prediction toward higher quality
- **Negative SHAP value**: Feature pushes prediction toward lower quality
- **High magnitude**: Feature has large impact on predictions
- **Beeswarm direction**: Shows how feature values affect predictions

## Practical Implications for Winemaking

### Key Takeaways

1. **Alcohol Content**: A critical quality indicator affecting mouthfeel and body
2. **Sulphates**: Moderate levels improve quality and preservation
3. **Volatile Acidity**: Must be minimized to avoid off-flavors
4. **Acidity Balance**: Citric acid adds freshness and brightness
5. **Preservation**: Free sulfur dioxide ratio is crucial for shelf life

### Winemaking Recommendations

- Monitor alcohol levels during fermentation
- Balance sulphate levels with antioxidant practices
- Ensure proper sanitation to prevent acetic acid production
- Use density measurements to track fermentation progress
- Monitor free SO2 levels for preservation

## Limitations

1. **Model Dependency**: Feature importance is specific to the trained model
2. **Correlation**: Features may be correlated, affecting importance scores
3. **Non-causal**: Importance doesn't imply causation
4. **Dataset Specific**: Results may vary with different datasets
5. **SHAP Computation**: Can be slow for large datasets with SHAP values

## Testing

Run the test script to verify functionality:

```bash
# Test with Random Forest model
uv run python test_feature_importance.py
```

This will:
1. Train a Random Forest model
2. Run feature importance analysis
3. Generate all visualizations and reports
4. Verify output files

## Code Quality

- Follows PEP 8 style guide
- Comprehensive docstrings with type hints
- Modular design with reusable classes
- Error handling and validation
- Well-documented for easy maintenance

## Future Enhancements

Potential improvements:
- Support for more model types (neural networks, SVM)
- Interactive visualizations (Plotly, Dash)
- Feature importance aggregation across multiple models
- Cross-validation feature importance
- Ablation studies
- Feature interaction analysis
