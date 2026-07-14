"""
Feature Importance and Interpretation Module for Wine Quality Classification

This module provides functionality to:
1. Extract feature importance from trained tree-based models (Random Forest, XGBoost, LightGBM)
2. Calculate SHAP values for model interpretability
3. Visualize feature importance (bar plots, summary plots)
4. Create SHAP summary plots (beeswarm plots)
5. Generate comprehensive feature importance reports

Usage:
    from src.fase_2.feature_importance import FeatureImportanceAnalyzer
    analyzer = FeatureImportanceAnalyzer(
        model_path="models/best_model.joblib",
        data_path="data/processed/train.csv",
        output_dir="results/feature_importance"
    )
    analyzer.analyze()
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from typing import Dict, List, Optional, Tuple


class FeatureImportanceAnalyzer:
    """
    Analyze and interpret feature importance for wine quality classification models.

    This class handles:
    - Extracting feature importance from tree-based models
    - Calculating SHAP values for interpretability
    - Creating visualization plots
    - Generating comprehensive reports
    """

    def __init__(
        self,
        model_path: str,
        data_path: str,
        feature_names: Optional[List[str]] = None,
        output_dir: str = "results/feature_importance",
    ):
        """
        Initialize the Feature Importance Analyzer.

        Args:
            model_path: Path to the trained model file (joblib format)
            data_path: Path to the training data file (CSV format)
            feature_names: List of feature names (optional, auto-detected from data)
            output_dir: Directory to save plots and reports
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.feature_names: Optional[List[str]] = feature_names

        # Load data and model
        self.X_train = None
        self.X_test = None
        self.model = None
        self.feature_names = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """
        Load and prepare the dataset.

        Args:
            test_size: Proportion of data to use as test set
            random_state: Random seed for reproducibility

        Returns:
            Train and test data as pandas DataFrames
        """
        # Load data
        self.X = pd.read_csv(self.data_path)

        # Feature columns (including quality since it's the target)
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
            "quality",  # Include quality for stratification
        ]

        # Filter to include only requested columns
        self.X = self.X[feature_cols]

        # Feature names (exclude quality for feature importance)
        self.feature_names = [
            col for col in feature_cols if col != "quality"
        ]

        # Split data
        from sklearn.model_selection import train_test_split

        # Handle quality column differently based on whether it's numeric or categorical
        if "quality" in self.X.columns:
            self.y = self.X["quality"]
            # Don't drop quality column yet - we need it for stratification
            self.X_test_split = self.X.copy()
            self.X_train_split = self.X.drop(columns=["quality"])
        else:
            raise ValueError("Dataset must contain 'quality' column")

        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train_split, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        print(f"Loaded data: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
        print(f"Features: {len(self.feature_names)}")

        return self.X_train, self.X_test

    def load_model(self) -> None:
        """
        Load the trained model from disk.

        Returns:
            The loaded model

        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If required library not installed
        """
        try:
            self.model = joblib.load(self.model_path)
            print(f"Loaded model: {type(self.model).__name__}")
            return self.model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            raise ImportError(f"Error loading model: {e}")

    def extract_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.

        Returns:
            DataFrame with features and their importance scores
        """
        # Check if model is tree-based
        model_type = self.model.__class__.__name__

        if "RandomForest" in model_type:
            # Random Forest has built-in feature_importances_
            importances = self.model.feature_importances_
        elif "GradientBoosting" in model_type or "XGBoost" in model_type or "LightGBM" in model_type:
            # Tree-based models have feature_importances_
            importances = self.model.feature_importances_
        else:
            # For non-tree models, use SHAP values as alternative
            print("Warning: Non-tree-based model detected. Using SHAP values for feature importance.")
            # Will use SHAP in analyze() method
            return None

        # Create DataFrame
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        })

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

        return importance_df

    def calculate_shap_values(
        self, X: pd.DataFrame, background_samples: int = 100
    ) -> Tuple[np.ndarray, shap.Explainer]:
        """
        Calculate SHAP values for model interpretability.

        Args:
            X: Input features (DataFrame)
            background_samples: Number of background samples for shap.Explainer

        Returns:
            Tuple of (SHAP values, explainer object)
        """
        model_type = self.model.__class__.__name__

        # Select appropriate explainer based on model type
        if "RandomForest" in model_type:
            explainer = shap.TreeExplainer(self.model)
        elif "XGBoost" in model_type:
            try:
                import xgboost as xgb
                if isinstance(self.model, xgb.XGBClassifier):
                    explainer = shap.TreeExplainer(self.model)
                else:
                    explainer = shap.Explainer(self.model, X)
            except ImportError:
                explainer = shap.Explainer(self.model, X)
        elif "LightGBM" in model_type:
            try:
                import lightgbm as lgb
                if isinstance(self.model, lgb.LGBMClassifier):
                    explainer = shap.TreeExplainer(self.model)
                else:
                    explainer = shap.Explainer(self.model, X)
            except ImportError:
                explainer = shap.Explainer(self.model, X)
        else:
            # For non-tree models, use TreeExplainer with a small sample
            print("Using TreeExplainer for non-tree model (may have limitations)")
            explainer = shap.TreeExplainer(self.model, X)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X)

        # Ensure proper format
        if isinstance(shap_values, list):
            # Multi-class: select SHAP values for the majority class or combine
            shap_values = shap_values[0] if len(shap_values) == 2 else shap_values[0]

        return shap_values, explainer

    def plot_feature_importance(
        self, importance_df: pd.DataFrame, max_features: int = 15
    ) -> Path:
        """
        Create feature importance bar plot.

        Args:
            importance_df: DataFrame with feature and importance columns
            max_features: Maximum number of features to display

        Returns:
            Path to the saved plot
        """
        # Take top N features
        plot_df = importance_df.head(max_features)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, max_features))
        bars = ax.barh(
            plot_df["feature"].astype(str),
            plot_df["importance"],
            color=colors,
            alpha=0.8,
        )

        # Add value labels
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax.text(
                row["importance"] + 0.001,
                i,
                f"{row['importance']:.3f}",
                va="center",
                fontsize=9,
            )

        # Formatting
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        ax.set_title(f"Top {max_features} Feature Importance - {self.model.__class__.__name__}", fontsize=14)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / "feature_importance_barplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved feature importance plot: {output_path}")

        return output_path

    def plot_shap_beeswarm(self, shap_values: np.ndarray) -> Path:
        """
        Create SHAP beeswarm plot (summary plot).

        Args:
            shap_values: SHAP values for predictions

        Returns:
            Path to the saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create beeswarm plot
        shap.summary_plot(
            shap_values,
            self.X_test,
            plot_type="beeswarm",
            show=False,
            max_display=20,
        )

        # Custom title
        plt.title(f"SHAP Beeswarm Plot - {self.model.__class__.__name__}", fontsize=14, pad=20)

        # Save plot
        output_path = self.output_dir / "shap_beeswarm_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved SHAP beeswarm plot: {output_path}")

        return output_path

    def plot_shap_bar(self, shap_values: np.ndarray) -> Path:
        """
        Create SHAP bar plot (mean absolute SHAP values).

        Args:
            shap_values: SHAP values for predictions

        Returns:
            Path to the saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bar plot
        shap.summary_plot(
            shap_values,
            self.X_test,
            plot_type="bar",
            show=False,
            max_display=15,
        )

        # Custom title
        plt.title(f"SHAP Bar Plot - {self.model.__class__.__name__}", fontsize=14, pad=20)

        # Save plot
        output_path = self.output_dir / "shap_bar_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved SHAP bar plot: {output_path}")

        return output_path

    def generate_summary_report(
        self,
        importance_df: pd.DataFrame,
        top_features: int = 10,
    ) -> Path:
        """
        Generate comprehensive feature importance summary report.

        Args:
            importance_df: DataFrame with feature importance
            top_features: Number of top features to analyze

        Returns:
            Path to the saved report
        """
        # Get top features
        top_importance = importance_df.head(top_features)

        # Model info
        model_type = self.model.__class__.__name__
        model_path = self.model_path.name

        # Create report content
        report_lines = []
        report_lines.append("# Feature Importance Analysis Report")
        report_lines.append("")
        report_lines.append(f"**Model:** {model_type}")
        report_lines.append(f"**Model File:** {model_path}")
        report_lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append(f"This report analyzes the most important features for wine quality prediction using a trained {model_type} model.")
        report_lines.append("")
        report_lines.append(f"The model uses {len(self.feature_names)} features to predict wine quality.")
        report_lines.append("")
        report_lines.append("Feature importance was extracted from the model's built-in feature importance scores and SHAP values.")
        report_lines.append("")

        # Top Features Table
        report_lines.append("## Top Features")
        report_lines.append("")
        report_lines.append("| Rank | Feature | Importance | Interpretation |")
        report_lines.append("|------|---------|------------|----------------|")

        for i, (idx, row) in enumerate(top_importance.iterrows(), 1):
            # Interpret feature
            interpretation = self._interpret_feature(row["feature"])
            report_lines.append(f"| {i} | {row['feature']} | {row['importance']:.4f} | {interpretation} |")

        report_lines.append("")

        # Feature Analysis
        report_lines.append("## Feature Analysis")
        report_lines.append("")

        for i, (idx, row) in enumerate(top_importance.iterrows(), 1):
            report_lines.append(f"### {i}. {row['feature']}")
            report_lines.append("")
            report_lines.append(f"**Importance Score:** {row['importance']:.4f}")
            report_lines.append("")
            report_lines.append(f"**Interpretation:** {self._interpret_feature(row['feature'])}")
            report_lines.append("")

            # Practical implications
            implications = self._get_practical_implications(row["feature"])
            report_lines.append("**Practical Implications for Winemaking:**")
            report_lines.append("")
            for implication in implications:
                report_lines.append(f"- {implication}")
            report_lines.append("")

        report_lines.append("## Data Source")
        report_lines.append("")
        report_lines.append(f"- Training data: {self.data_path}")
        report_lines.append(f"- Model type: {model_type}")
        report_lines.append("")

        report_lines.append("## Methodology")
        report_lines.append("")
        report_lines.append("Feature importance was calculated using two approaches:")
        report_lines.append("")
        report_lines.append("1. **Model-based feature importance**: Using the built-in feature importance scores from tree-based models.")
        report_lines.append("")
        report_lines.append("2. **SHAP (SHapley Additive exPlanations)**: Using SHAP values to understand how each feature contributes")
        report_lines.append("   to individual predictions and overall model predictions.")
        report_lines.append("")

        report_lines.append("## Limitations")
        report_lines.append("")
        report_lines.append("- Feature importance may vary depending on the training data and model configuration.")
        report_lines.append("- SHAP values provide local interpretability but may be computationally expensive for large datasets.")
        report_lines.append("- Correlation between features can affect importance scores.")
        report_lines.append("- The model's feature importance is based on how well each feature improves the model's ability")
        report_lines.append("  to predict quality, not necessarily causal relationships.")
        report_lines.append("")

        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("For winemakers:")
        report_lines.append("")
        report_lines.append("1. Focus on the most influential features (e.g., alcohol content, sulphates) during production.")
        report_lines.append("2. Monitor these key quality indicators during the winemaking process.")
        report_lines.append("3. Use the feature importance insights to optimize wine composition and quality.")
        report_lines.append("4. Consider the trade-offs between different features (e.g., acidity vs. sweetness).")
        report_lines.append("")

        # Convert to markdown file
        report_text = "\n".join(report_lines)
        output_path = self.output_dir / "feature_importance_summary.md"
        output_path.write_text(report_text, encoding="utf-8")

        print(f"Saved summary report: {output_path}")

        return output_path

    def _interpret_feature(self, feature: str) -> str:
        """
        Provide interpretation for a feature.

        Args:
            feature: Feature name

        Returns:
            Interpretation text
        """
        interpretations = {
            "alcohol": "Alcohol content has a strong influence on wine quality. Higher alcohol often correlates with better quality.",
            "volatile acidity": "Volatile acidity measures the amount of acetic acid in wine. Lower values generally indicate better quality.",
            "sulphates": "Sulphates are antimicrobial agents that help preserve wine. Moderate levels are associated with better quality.",
            "pH": "pH affects the taste and stability of wine. Optimal pH ranges are crucial for quality.",
            "density": "Density relates to the wine's sugar content and alcohol concentration.",
            "citric acid": "Citric acid adds freshness and brightness to wine.",
            "residual sugar": "Residual sugar contributes to sweetness. Balanced sugar levels are important for quality.",
            "chlorides": "Chlorides affect the saltiness and taste of wine. Lower chloride levels are preferred.",
            "free sulfur dioxide": "Free sulfur dioxide acts as an antimicrobial agent and antioxidant.",
            "total sulfur dioxide": "Total sulfur dioxide includes both bound and free forms, important for preservation.",
            "sulphates": "Sulphates contribute to the wine's taste and act as antioxidants.",
            "alcohol_density_ratio": "Ratio of alcohol to density, indicating wine strength and body.",
            "free_sulfur_ratio": "Proportion of free to total sulfur dioxide, indicating preservative capacity.",
            "log_total_sulfur": "Natural logarithm of total sulfur dioxide, useful for normalizing high values.",
            "sugar_alcohol_ratio": "Ratio of residual sugar to alcohol, affecting sweetness and body balance.",
        }

        return interpretations.get(feature, f"{feature} is an important feature for wine quality prediction.")

    def _get_practical_implications(self, feature: str) -> List[str]:
        """
        Get practical implications for winemaking.

        Args:
            feature: Feature name

        Returns:
            List of practical implications
        """
        implications = {
            "alcohol": [
                "Alcohol content is a key quality indicator",
                "Higher alcohol often correlates with perceived quality",
                "Alcohol level affects mouthfeel and body",
                "Monitor alcohol during fermentation process",
                "Consider blending to achieve optimal alcohol content"
            ],
            "volatile acidity": [
                "Low volatile acidity is crucial for quality",
                "Excess volatile acidity can cause vinegar-like off-flavors",
                "Monitor acetic acid production during fermentation",
                "Ensure proper sanitation to prevent bacterial contamination"
            ],
            "sulphates": [
                "Moderate sulphate levels improve quality",
                "Sulphates help preserve wine and prevent spoilage",
                "Balance sulphate levels with antioxidant care",
                "Monitor sulphite levels for health and quality"
            ],
            "pH": [
                "Maintain optimal pH (3.0-3.5 for most wines)",
                "pH affects wine stability and shelf life",
                "pH influences fermentation kinetics",
                "Use pH monitoring throughout production"
            ],
            "density": [
                "Density indicates sugar and alcohol content",
                "Monitor density during fermentation",
                "Use density to track fermentation progress",
                "Understand the relationship between density and quality"
            ],
            "citric acid": [
                "Citric acid adds freshness and perceived quality",
                "Moderate citric acid improves wine balance",
                "Monitor during grape selection and fermentation",
                "Citric acid can enhance flavor complexity"
            ],
            "residual sugar": [
                "Balanced sugar levels contribute to quality",
                "Excess sugar can mask poor quality",
                "Monitor sugar at various stages of production",
                "Consider consumer preferences for sweetness levels"
            ],
            "chlorides": [
                "Low chloride levels indicate better quality",
                "Chlorides affect wine salinity and taste",
                "Monitor water quality used in production",
                "Excessive chlorides can cause off-flavors"
            ],
            "free sulfur dioxide": [
                "Free sulfite levels protect wine quality",
                "Antimicrobial properties prevent spoilage",
                "Monitor free SO2 levels regularly",
                "Adjust based on wine type and storage conditions"
            ],
            "total sulfur dioxide": [
                "Total sulfite levels are crucial for preservation",
                "Combined bound and free forms provide protection",
                "Monitor throughout wine's lifecycle",
                "Comply with regulatory limits"
            ],
            "sulphates": [
                "Sulphate levels impact taste and preservation",
                "Moderate levels associated with quality",
                "Consider grape variety and terroir effects",
                "Balance with antioxidant practices"
            ],
            "alcohol_density_ratio": [
                "Ratio indicates wine strength and body",
                "Affects mouthfeel and perceived quality",
                "Monitor during fermentation and aging",
                "Consider blending for optimal balance"
            ],
            "free_sulfur_ratio": [
                "Ratio indicates preservative capacity",
                "Higher ratio suggests better preservation",
                "Monitor to ensure adequate protection",
                "Balance with wine sensitivity"
            ],
            "log_total_sulfur": [
                "Log transformation handles wide value ranges",
                "Helps in normalizing sulfur dioxide measurements",
                "Useful for modeling purposes",
                "Correlates with preservation effectiveness"
            ],
            "sugar_alcohol_ratio": [
                "Ratio indicates sweetness-alcohol balance",
                "Affects perceived quality and body",
                "Monitor during fermentation development",
                "Consider consumer taste preferences"
            ],
        }

        return implications.get(feature, [
            "Monitor this feature during production",
            "Consider its impact on overall wine quality",
            "Adjust processing to optimize quality"
        ])

    def analyze(self) -> Dict[str, Path]:
        """
        Perform complete feature importance analysis.

        Steps:
        1. Load data and model
        2. Extract feature importance
        3. Calculate SHAP values
        4. Create visualizations
        5. Generate summary report

        Returns:
            Dictionary with paths to all generated outputs
        """
        print("\n" + "="*80)
        print("Feature Importance Analysis")
        print("="*80)

        # Load data and model
        print("\n[1/5] Loading data and model...")
        self.load_data()
        self.load_model()

        # Extract feature importance
        print("\n[2/5] Extracting feature importance...")
        importance_df = self.extract_feature_importance()

        barplot_path = None
        barplot_shap_path = None
        beeswarm_path = None
        summary_path = None

        if importance_df is not None:
            # Create feature importance plot
            print("\n[3/5] Creating feature importance plot...")
            barplot_path = self.plot_feature_importance(importance_df)

            # Calculate SHAP values
            print("\n[4/5] Calculating SHAP values...")
            shap_values, explainer = self.calculate_shap_values(self.X_test)
            print("   SHAP values calculated for", self.X_test.shape[0], "samples")

            # Create SHAP plots
            print("\n[4/5] Creating SHAP visualizations...")
            beeswarm_path = self.plot_shap_beeswarm(shap_values)
            barplot_shap_path = self.plot_shap_bar(shap_values)
        else:
            # Calculate SHAP values for non-tree models
            print("\n[3/5] Calculating SHAP values for non-tree model...")
            shap_values, explainer = self.calculate_shap_values(self.X_train)
            print("   SHAP values calculated for", self.X_train.shape[0], "samples")

            # Create SHAP plots
            print("\n[4/5] Creating SHAP visualizations...")
            beeswarm_path = self.plot_shap_beeswarm(shap_values)
            barplot_shap_path = self.plot_shap_bar(shap_values)
            print("   Note: Non-tree model detected. Using SHAP values for importance.")

        # Generate summary report
        print("\n[5/5] Generating summary report...")
        summary_path = self.generate_summary_report(importance_df)

        print("\n" + "="*80)
        print("Analysis Complete!")
        print("="*80)
        print(f"\nGenerated Files:")
        print(f"  - Feature importance bar plot: {barplot_path}")
        print(f"  - SHAP beeswarm plot: {beeswarm_path}")
        print(f"  - SHAP bar plot: {barplot_shap_path}")
        print(f"  - Summary report: {summary_path}")
        print("="*80 + "\n")

        outputs = {
            "barplot": barplot_path,
            "beeswarm": beeswarm_path,
            "shap_bar": barplot_shap_path,
            "summary": summary_path,
        }

        return outputs


def main():
    """Run feature importance analysis with default configuration."""
    analyzer = FeatureImportanceAnalyzer(
        model_path="models/test_model.joblib",
        data_path="data/processed/train.csv",
        output_dir="results/feature_importance",
    )

    outputs = analyzer.analyze()

    # Print summary
    print("\n" + "="*80)
    print("Feature Importance Analysis - Summary")
    print("="*80)
    print("\nGenerated Outputs:")
    for name, path in outputs.items():
        print(f"  ✓ {name}: {path}")
    print("="*80)


if __name__ == "__main__":
    main()
