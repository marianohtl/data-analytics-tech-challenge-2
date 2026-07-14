#!/usr/bin/env python
# coding: utf-8

# # Model Evaluation and Comparison - Notebook
# 
# This notebook demonstrates how to use the `ModelEvaluator` class to evaluate, compare, and select the best model for wine quality prediction.
# 
# ## Table of Contents
# 1. [Setup and Imports](#setup)
# 2. [Load and Evaluate Models](#load)
# 3. [Generate Comparisons](#compare)
# 4. [Visualize Results](#visualize)
# 5. [Select Best Model](#best)

# ## Setup and Imports <a name="setup"></a>

# In[ ]:


import sys
from pathlib import Path

# Add project root to path
project_root = Path(".")
sys.path.insert(0, str(project_root))

from src.fase_2.model_evaluator import ModelEvaluator


# ## Load and Evaluate Models <a name="load"></a>

# In[ ]:


# Initialize evaluator
evaluator = ModelEvaluator(
    model_dir="models",
    test_data_path="data/processed/train.csv",
    output_dir="results/model_comparison"
)

# Load test data
X_test, y_test = evaluator.load_test_data()
print(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Load models
models = evaluator.load_models()
print(f"\nLoaded {len(models)} model(s): {list(models.keys())}")

# Evaluate models
metrics = evaluator.evaluate_models()
print(f"\nGenerated metrics for {len(metrics)} model(s)")


# ## Generate Comparisons <a name="compare"></a>

# In[ ]:


# Compare all models
comparison_df, best_model = evaluator.compare_models()

# Display comparison table
display(comparison_df)


# ## Visualize Results <a name="visualize"></a>

# In[ ]:


# Generate all visualizations
evaluator.plot_confusion_matrices()
evaluator.plot_roc_curves()
evaluator.plot_all_metrics_comparison()

print("\nVisualizations saved to: results/model_comparison/")


# ## Select Best Model <a name="best"></a>

# In[ ]:


# Print detailed best model information
print("=" * 80)
print("BEST MODEL SELECTION")
print("=" * 80)
print(f"\nModel: {best_model['Model']}")
print(f"F1-Score: {best_model['F1-Score']:.4f}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall: {best_model['Recall']:.4f}")

# Generate and display summary report
report = evaluator.generate_summary_report()

# Display a snippet of the report
print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)
print(report)


# ## Additional Analysis
# 
# ### Inspect Confusion Matrix
# 
# Let's look at the confusion matrix for the best model:

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get best model metrics
best_model_name = best_model['Model']
best_metrics = metrics[best_model_name]
cm = np.array(best_metrics['confusion_matrix'])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.show()

print(f"\nConfusion Matrix for {best_model_name}:")
print(pd.DataFrame(cm, index=['low', 'medium', 'high', 'very_high'], columns=['low', 'medium', 'high', 'very_high']))

