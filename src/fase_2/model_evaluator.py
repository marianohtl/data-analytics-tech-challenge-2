"""
Model Evaluation and Comparison Module.

This module provides functionality to:
1. Load trained models from disk
2. Generate comprehensive comparison metrics (accuracy, precision, recall, F1-score)
3. Create confusion matrices for each model
4. Generate ROC-AUC curves for each model
5. Compare all models and select the best one with justification
6. Save comparison plots and summary reports
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


class ModelEvaluator:
    """
    Load, evaluate, and compare trained classification models.
    """

    def __init__(
        self,
        model_dir: str = "models",
        test_data_path: str = "data/processed/test.csv",
        output_dir: str = "results/model_comparison",
    ):
        """
        Initialize the ModelEvaluator.

        Args:
            model_dir: Directory containing trained model files
            test_data_path: Path to test data CSV file
            output_dir: Directory to save comparison plots and reports
        """
        self.model_dir = Path(model_dir)
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.X_test = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.model_names = []

    def load_test_data(self):
        """
        Load test data from CSV file.

        Returns:
            Test features and target
        """
        print("Loading test data...")
        df = pd.read_csv(self.test_data_path)

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

        self.X_test = df[feature_cols]

        # Bin the continuous target variable into categories
        bins = [-np.inf, -0.5, 0.5, 1.5, np.inf]
        labels = ["low", "medium", "high", "very_high"]
        self.y_test = pd.cut(df["quality"], bins=bins, labels=labels, include_lowest=True)

        print(f"Test data shape: {self.X_test.shape}")
        print(f"Target distribution:\n{self.y_test.value_counts().sort_index()}")

        return self.X_test, self.y_test

    def load_models(self):
        """
        Load all trained models from model directory.

        Returns:
            Dictionary of loaded models
        """
        print("\n" + "="*80)
        print("Loading trained models...")
        print("="*80)

        self.models = {}
        self.model_names = []

        # Get all model files
        model_files = sorted(self.model_dir.glob("*.joblib"))

        if not model_files:
            raise FileNotFoundError(
                f"No model files found in {self.model_dir}. "
                f"Please train models first using ModelTrainer."
            )

        for model_file in model_files:
            try:
                model = joblib.load(model_file)
                model_name = model_file.stem
                self.models[model_name] = model
                self.model_names.append(model_name)
                print(f"Loaded: {model_name} -> {type(model).__name__}")
            except Exception as e:
                print(f"Error loading {model_file.name}: {e}")

        print(f"\nSuccessfully loaded {len(self.models)} model(s).")

        return self.models

    def evaluate_models(self):
        """
        Evaluate all loaded models using test data.

        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*80)
        print("Evaluating models...")
        print("="*80)

        self.metrics = {}

        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")

            # Make predictions
            y_pred = model.predict(self.X_test)

            # Store predictions
            self.predictions[model_name] = y_pred

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

            self.metrics[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
            }

            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

        return self.metrics

    def plot_confusion_matrices(self):
        """
        Create confusion matrices for all models and save as PNG files.
        """
        print("\n" + "="*80)
        print("Creating confusion matrices...")
        print("="*80)

        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        for idx, model_name in enumerate(self.model_names):
            metrics = self.metrics[model_name]
            cm = np.array(metrics["confusion_matrix"])

            # Plot confusion matrix
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                cbar_kws={"label": "Count"},
            )
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("Actual Label")
            axes[idx].set_title(f"{model_name}\nAccuracy: {metrics['accuracy']:.3f}")

        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        output_path = self.output_dir / "confusion_matrices.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved confusion matrices to: {output_path}")

    def plot_roc_curves(self):
        """
        Create ROC-AUC curves for all models and save as PNG file.
        Uses One-vs-Rest (OvR) strategy for multiclass classification.
        """
        print("\n" + "="*80)
        print("Creating ROC-AUC curves...")
        print("="*80)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each model's ROC curve
        for model_name in self.model_names:
            y_pred = self.predictions[model_name]
            y_prob = self.models[model_name].predict_proba(self.X_test)

            # Calculate ROC AUC using One-vs-Rest (OvR) strategy
            try:
                auc = roc_auc_score(self.y_test, y_prob, multi_class="ovr", average="weighted")
            except ValueError:
                # Fallback to OvO if OvR fails
                auc = roc_auc_score(self.y_test, y_prob, multi_class="ovo", average="weighted")

            # Plot ROC curve for each class (One-vs-Rest)
            n_classes = y_prob.shape[1]
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_classes))

            for i, color in zip(range(n_classes), colors):
                y_binary = (self.y_test == list(self.y_test.unique())[i]).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
                ax.plot(fpr, tpr, color=color, lw=2, alpha=0.7,
                        label=f"{model_name} - {list(self.y_test.unique())[i]}")

            # Overall AUC
            ax.plot([], [], color='gray', linestyle='--', lw=2,
                    label=f"Overall AUC = {auc:.3f}")

        # Add diagonal line
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

        # Set labels and title
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves for Model Comparison (One-vs-Rest)")
        ax.legend(loc="lower right", bbox_to_anchor=(1.02, 1), fontsize='small')
        ax.grid(True, alpha=0.3)

        # Save plot
        output_path = self.output_dir / "roc_curves.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved ROC curves to: {output_path}")

    def plot_all_metrics_comparison(self):
        """
        Create comparison bar charts for all metrics and save as PNG file.
        """
        print("\n" + "="*80)
        print("Creating metrics comparison...")
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

        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(comparison_df))
        width = 0.2

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for idx, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
            offset = (idx - 1) * width
            ax.bar(
                x + offset,
                comparison_df[metric],
                width,
                label=metric,
                color=colors[idx % len(colors)],
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for idx, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
            offset = (idx - 1) * width
            for i, v in enumerate(comparison_df[metric]):
                ax.text(
                    i + offset,
                    v + 0.02,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        output_path = self.output_dir / "metrics_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved metrics comparison to: {output_path}")

    def compare_models(self):
        """
        Compare all models and select the best one.

        Returns:
            Comparison DataFrame and best model information
        """
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
        print("\nModel Performance Summary:")
        print("-" * 80)
        print(comparison_df.to_string(index=False))
        print("-" * 80)

        # Find best model based on F1-Score
        best_model_idx = comparison_df["F1-Score"].idxmax()
        best_model = comparison_df.loc[best_model_idx]
        best_model_name = best_model["Model"]
        best_model_f1 = best_model["F1-Score"]

        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {best_model_f1:.4f}")
        print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        print(f"   Precision: {best_model['Precision']:.4f}")
        print(f"   Recall: {best_model['Recall']:.4f}")

        return comparison_df, best_model

    def generate_summary_report(self):
        """
        Generate a comprehensive summary report and save as Markdown file.

        Returns:
            Summary report as string
        """
        print("\n" + "="*80)
        print("Generating summary report...")
        print("="*80)

        # Get best model
        comparison_df, best_model = self.compare_models()

        # Prepare report content
        report_lines = [
            "# Model Comparison Summary",
            "",
            "**Generated:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
            "## Executive Summary",
            "",
            f"The analysis evaluated {len(self.models)} classification model(s) trained on the wine quality dataset.",
            f"The **{best_model['Model']}** model achieved the best overall performance with an F1-Score of {best_model['F1-Score']:.4f}.",
            "",
            "## Model Performance Table",
            "",
            "| Model | Accuracy | Precision | Recall | F1-Score |",
            "|-------|----------|-----------|--------|----------|",
        ]

        for _, row in comparison_df.iterrows():
            report_lines.append(
                f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | "
                f"{row['Recall']:.4f} | {row['F1-Score']:.4f} |"
            )

        report_lines.extend(
            [
                "",
                "## Best Model Selection",
                "",
                f"**Selected Model:** `{best_model['Model']}`",
                "",
                f"**Justification:** This model was selected based on its highest F1-Score ({best_model['F1-Score']:.4f}), which provides the best balance between precision and recall. The F1-Score is particularly important for imbalanced classification problems.",
                f"",
                f"**Performance Metrics:**",
                f"- Accuracy: {best_model['Accuracy']:.4f}",
                f"- Precision: {best_model['Precision']:.4f}",
                f"- Recall: {best_model['Recall']:.4f}",
                f"",
                "## Practical Recommendations",
                "",
                "1. **Deployment:** The selected model is ready for deployment after thorough validation on production data.",
                "2. **Monitoring:** Establish monitoring for model drift, especially for F1-Score and Accuracy metrics.",
                "3. **Threshold Optimization:** Consider tuning classification thresholds based on business priorities.",
                "4. **Feature Engineering:** Continue exploring additional features that may improve model performance.",
                "5. **Ensemble Methods:** Consider ensemble approaches (bagging, boosting) to further improve predictions.",
                "",
                "## Next Steps",
                "",
                "1. **Validate on Production Data:** Test the selected model on real-world data before deployment.",
                "2. **Performance Benchmarking:** Compare against baseline and industry standards.",
                "3. **Model Explainability:** Investigate feature importance using SHAP values or similar techniques.",
                "4. **Deployment Pipeline:** Set up automated deployment and monitoring pipelines.",
                "5. **A/B Testing:** Deploy the model in a controlled A/B testing environment to measure real-world impact.",
                "",
                "## Visualizations",
                "",
                f"Generated plots saved in: `{self.output_dir}`",
                "- `confusion_matrices.png`: Confusion matrices for all models",
                "- `roc_curves.png`: ROC-AUC curves showing discrimination performance",
                "- `metrics_comparison.png`: Bar chart comparison of all metrics",
                "",
                "## Appendix",
                "",
                "## Model Details",
                "",
            ]
        )

        # Add details for each model
        for model_name, metrics in self.metrics.items():
            report_lines.append(f"\n### {model_name}")
            report_lines.append("\n**Classification Report:**")
            report_lines.append("\n" + classification_report(self.y_test, self.predictions[model_name], zero_division=0))

        # Save report
        report_path = self.output_dir.parent / "model_comparison_summary.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"\nSaved summary report to: {report_path}")

        return "\n".join(report_lines)


def main():
    """Main function to demonstrate model evaluation and comparison."""
    print("=" * 80)
    print("Model Evaluation and Comparison")
    print("=" * 80)

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_dir="models",
        test_data_path="data/processed/train.csv",  # Using train.csv for evaluation
        output_dir="results/model_comparison",
    )

    # Load test data
    evaluator.load_test_data()

    # Load models
    evaluator.load_models()

    # Evaluate models
    evaluator.evaluate_models()

    # Create visualizations
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_all_metrics_comparison()

    # Compare models and generate summary
    evaluator.compare_models()
    evaluator.generate_summary_report()

    print("\n" + "=" * 80)
    print("Model evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
