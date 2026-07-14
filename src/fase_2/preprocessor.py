"""
Preprocessing Pipeline Module for Wine Quality Dataset

This module provides a comprehensive preprocessing pipeline for the wine quality dataset,
including:
- Missing value handling
- Outlier detection and removal
- Feature scaling
- Feature engineering
- Data splitting for train/validation/test sets
- Data persistence using Parquet format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Constants
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_IQR_THRESHOLD = 1.5
DEFAULT_OUTLIERS_TO_REMOVE = "all"


class Preprocessor:
    """
    A comprehensive preprocessing pipeline for wine quality dataset.

    This class handles the complete preprocessing workflow including:
    - Missing value imputation
    - Outlier removal using IQR method
    - Feature scaling
    - Feature engineering
    - Data splitting

    Attributes:
        random_state (int): Random seed for reproducibility (default: 42)
        scaler (StandardScaler or MinMaxScaler): Feature scaler
        feature_engineering (bool): Whether to apply feature engineering (default: True)
        iqr_threshold (float): IQR multiplier for outlier detection (default: 1.5)
        outliers_to_remove (str): How to handle outliers ('all', 'train', 'validation', 'test')
    """

    def __init__(
        self,
        random_state: int = DEFAULT_RANDOM_STATE,
        scaler_type: str = "standard",
        feature_engineering: bool = True,
        iqr_threshold: float = DEFAULT_IQR_THRESHOLD,
        outliers_to_remove: str = DEFAULT_OUTLIERS_TO_REMOVE,
    ):
        """
        Initialize the Preprocessor.

        Args:
            random_state: Random seed for reproducibility
            scaler_type: Type of scaler ('standard' or 'minmax')
            feature_engineering: Whether to apply feature engineering
            iqr_threshold: IQR multiplier for outlier detection
            outliers_to_remove: How to handle outliers ('all', 'train', 'validation', 'test')
        """
        self.random_state = random_state
        self.feature_engineering = feature_engineering
        self.iqr_threshold = iqr_threshold
        self.outliers_to_remove = outliers_to_remove

        # Initialize scaler
        if scaler_type.lower() == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        # Store metadata
        self.metadata: Dict[str, Any] = {}
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.feature_engineering_info: List[str] = []

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "mean",
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.

        Args:
            df: Input DataFrame
            strategy: Strategy for missing values ('mean', 'median', 'mode', 'drop')

        Returns:
            DataFrame with missing values handled

        Raises:
            ValueError: If strategy is not recognized
        """
        print("\n" + "="*60)
        print("HANDLING MISSING VALUES")
        print("="*60)

        missing_before = df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")

        if missing_before == 0:
            print("No missing values found")
            return df.copy()

        df = df.copy()

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col in numeric_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strategy == "drop":
                    df = df.dropna(subset=[col])
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                print(f"  Filled {missing} missing values in '{col}' using '{strategy}'")

        for col in categorical_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                if strategy == "drop":
                    df = df.dropna(subset=[col])
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")
                print(f"  Handled {missing} missing values in categorical column '{col}'")

        print(f"Missing values after: {df.isnull().sum().sum()}")
        print("="*60)

        return df

    def detect_outliers(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect outliers using IQR method.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (Dataframe without outliers, outlier statistics)
        """
        print("\n" + "="*60)
        print("OUTLIER DETECTION (IQR METHOD)")
        print("="*60)

        outliers_info = {}
        df_cleaned = df.copy()

        for col in self.numeric_columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_threshold * IQR
            upper_bound = Q3 + self.iqr_threshold * IQR

            outliers = df_cleaned[
                (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            ]
            outlier_count = len(outliers)
            outlier_percentage = outlier_count / len(df_cleaned) * 100 if len(df_cleaned) > 0 else 0.0

            outliers_info[col] = {
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_percentage,
            }

            print(f"  Column: {col}")
            print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"    Outliers: {outlier_count} ({outlier_percentage:.2f}%)")

        print("="*60)

        return df_cleaned, outliers_info

    def remove_outliers(
        self,
        df: pd.DataFrame,
        outliers_info: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Remove outliers from the dataframe.

        Args:
            df: Input DataFrame
            outliers_info: Optional pre-computed outlier information

        Returns:
            DataFrame without outliers
        """
        print("\n" + "="*60)
        print(f"REMOVING OUTLIERS ({self.outliers_to_remove.upper()})")
        print("="*60)

        df_cleaned = df.copy()

        if outliers_info is None:
            df_cleaned, outliers_info = self.detect_outliers(df_cleaned)

        # Determine which columns to apply outlier removal to
        columns_to_process = self.numeric_columns if self.numeric_columns else None

        total_outliers = 0  # Initialize to 0

        if outliers_info:
            total_outliers = sum(info.get("outlier_count", 0) for info in outliers_info.values())
            print(f"Total outliers detected: {total_outliers}")

            if self.outliers_to_remove == "all":
                # Remove all outliers
                for col, info in outliers_info.items():
                    lower_bound = info.get("lower_bound", float('-inf'))
                    upper_bound = info.get("upper_bound", float('inf'))
                    df_cleaned = df_cleaned[
                        (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
                    ]
                print(f"Removed all {total_outliers} outliers")
            elif self.outliers_to_remove == "train":
                # Only remove from training data
                pass  # Outliers handled during split
            elif self.outliers_to_remove in ["validation", "test"]:
                # Remove from validation/test sets
                for col, info in outliers_info.items():
                    lower_bound = info.get("lower_bound", float('-inf'))
                    upper_bound = info.get("upper_bound", float('inf'))
                    df_cleaned = df_cleaned[
                        (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
                    ]
                print(f"Removed {total_outliers} outliers from data")
        else:
            print("No outliers detected")

        df_cleaned.reset_index(drop=True, inplace=True)
        print(f"DataFrame shape after outlier removal: {df_cleaned.shape}")
        print("="*60)

        self.metadata["outliers_removed"] = total_outliers
        return df_cleaned

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features through feature engineering.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        if not self.feature_engineering:
            return df.copy()

        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)

        df_eng = df.copy()

        # Feature engineering ideas based on wine chemistry
        if 'alcohol' in df_eng.columns and 'density' in df_eng.columns:
            # Alcohol-to-density ratio (inverse relationship)
            # This can indicate the concentration of ethanol
            df_eng['alcohol_density_ratio'] = df_eng['alcohol'] / df_eng['density']
            print("  ✓ Created: alcohol_density_ratio (alcohol/density)")
            self.feature_engineering_info.append("alcohol_density_ratio")

        if 'free sulfur dioxide' in df_eng.columns and 'total sulfur dioxide' in df_eng.columns:
            # Ratio of free to total sulfur dioxide
            # Indicates how much of the sulfur dioxide is in free form vs bound
            df_eng['free_sulfur_ratio'] = df_eng['free sulfur dioxide'] / df_eng['total sulfur dioxide']
            print("  ✓ Created: free_sulfur_ratio (free/total sulfur dioxide)")
            self.feature_engineering_info.append("free_sulfur_ratio")

        if 'total sulfur dioxide' in df_eng.columns:
            # Log transform of total sulfur dioxide (often has skewed distribution)
            df_eng['log_total_sulfur'] = np.log1p(df_eng['total sulfur dioxide'])
            print("  ✓ Created: log_total_sulfur (log1p of total sulfur dioxide)")
            self.feature_engineering_info.append("log_total_sulfur")

        if 'residual sugar' in df_eng.columns and 'alcohol' in df_eng.columns:
            # Sugar-to-alcohol ratio
            df_eng['sugar_alcohol_ratio'] = df_eng['residual sugar'] / df_eng['alcohol']
            print("  ✓ Created: sugar_alcohol_ratio (residual sugar/alcohol)")
            self.feature_engineering_info.append("sugar_alcohol_ratio")

        # Create quality categories for categorical target
        if 'quality' in df_eng.columns:
            # Create categories based on quality score
            df_eng['quality_category'] = pd.cut(
                df_eng['quality'],
                bins=[2, 5, 7, 10],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
            print("  ✓ Created: quality_category (categorical target)")
            self.feature_engineering_info.append("quality_category")

        print("="*60)
        print(f"Created {len(self.feature_engineering_info)} new features")
        print("="*60)

        return df_eng

    def scale_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Scale numeric features using the configured scaler.

        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (default: True)

        Returns:
            DataFrame with scaled features
        """
        if not self.numeric_columns:
            return df.copy()

        print("\n" + "="*60)
        print(f"SCALING FEATURES ({self.scaler.__class__.__name__})")
        print("="*60)

        df_scaled = df.copy()
        scaled_columns = []

        for col in self.numeric_columns:
            if col in df_scaled.columns:
                df_scaled[col] = self.scaler.fit_transform(
                    df_scaled[[col]] if fit else df_scaled[[col]]
                )
                scaled_columns.append(col)
                print(f"  ✓ Scaled: {col}")

        print(f"Total columns scaled: {len(scaled_columns)}")
        print("="*60)

        return df_scaled

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: Optional[float] = None,
        validation_size: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame
            test_size: Test set size (default: 0.15)
            validation_size: Validation set size (default: 0.15)

        Returns:
            Tuple of (train_df, validation_df, test_df, split_info)

        Raises:
            ValueError: If sizes don't add up to 1.0 or are invalid
        """
        print("\n" + "="*60)
        print("DATA SPLITTING")
        print("="*60)

        # Set default sizes if not provided
        if test_size is None:
            test_size = DEFAULT_TEST_RATIO
        if validation_size is None:
            validation_size = DEFAULT_VALIDATION_RATIO

        # Validate ratios
        total_size = test_size + validation_size
        if total_size > 1.0:
            raise ValueError(
                f"Validation ({validation_size:.1%}) + Test ({test_size:.1%}) "
                f"cannot exceed 100% (sum: {total_size:.1%})"
            )
        if total_size >= 1.0:
            raise ValueError(
                "Validation + Test sizes must be less than 1.0"
            )

        # Calculate train size
        train_size = 1.0 - total_size
        print(f"Split ratios: Train={train_size:.1%}, Validation={validation_size:.1%}, Test={test_size:.1%}")

        # First split: separate test
        stratify = df.get('quality', None)
        if stratify is not None and len(stratify.unique()) > 1 and len(stratify) > 2:
            try:
                temp_df, test_df = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=self.random_state,
                    stratify=stratify,
                )
            except ValueError:
                # Fall back to non-stratified split
                temp_df, test_df = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=self.random_state,
                )
        else:
            # Fall back to non-stratified split
            temp_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=self.random_state,
            )

        # Second split: separate validation
        stratify_val = temp_df.get('quality', None)
        validation_size_actual = validation_size / (validation_size + train_size)

        if stratify_val is not None and len(stratify_val.unique()) > 1 and len(stratify_val) > 2:
            try:
                train_df, validation_df = train_test_split(
                    temp_df,
                    test_size=validation_size_actual,
                    random_state=self.random_state,
                    stratify=stratify_val,
                )
            except ValueError:
                # Fall back to non-stratified split
                train_df, validation_df = train_test_split(
                    temp_df,
                    test_size=validation_size_actual,
                    random_state=self.random_state,
                )
        else:
            # Fall back to non-stratified split
            train_df, validation_df = train_test_split(
                temp_df,
                test_size=validation_size_actual,
                random_state=self.random_state,
            )

        # Print split statistics
        print(f"\nTrain set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation set: {len(validation_df)} samples ({len(validation_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

        # Print quality distribution in each split
        for name, split_df in [('Train', train_df), ('Validation', validation_df), ('Test', test_df)]:
            if 'quality' in split_df.columns:
                print(f"\n{name} set quality distribution:")
                print(split_df['quality'].value_counts().sort_index().to_string())

        split_info = {
            'train_size': len(train_df),
            'validation_size': len(validation_df),
            'test_size': len(test_df),
            'train_ratio': len(train_df) / len(df),
            'validation_ratio': len(validation_df) / len(df),
            'test_ratio': len(test_df) / len(df),
        }

        print("="*60)
        print(f"✓ Data split complete")
        print("="*60)

        return train_df, validation_df, test_df, split_info

    def fit_transform(
        self,
        df: pd.DataFrame,
        test_size: Optional[float] = None,
        validation_size: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline: fit and transform.

        This method applies the entire preprocessing workflow including:
        1. Handle missing values
        2. Detect and remove outliers
        3. Engineer features
        4. Scale features
        5. Split data

        Args:
            df: Input DataFrame
            test_size: Test set size (default: 0.15)
            validation_size: Validation set size (default: 0.15)

        Returns:
            Tuple of (train_df, validation_df, test_df, split_info)
        """
        print("\n" + "="*60)
        print("INITIATING PREPROCESSING PIPELINE")
        print("="*60)

        # Step 1: Handle missing values
        df = self.handle_missing_values(df)

        # Step 2: Identify column types
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        print(f"\nNumeric columns: {self.numeric_columns}")
        print(f"Categorical columns: {self.categorical_columns}")

        # Step 3: Detect and remove outliers
        df = self.remove_outliers(df)

        # Step 4: Feature engineering
        df = self.engineer_features(df)

        # Step 5: Scale features
        df = self.scale_features(df)

        # Step 6: Split data
        train_df, validation_df, test_df, split_info = self.split_data(
            df, test_size, validation_size
        )

        # Store metadata
        self.metadata.update({
            'original_shape': df.shape,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'feature_engineering_info': self.feature_engineering_info,
            **split_info,
        })

        print(f"\n{'='*60}")
        print("✓ Preprocessing pipeline completed successfully!")
        print(f"{'='*60}")

        return train_df, validation_df, test_df, split_info

    def save_to_parquet(
        self,
        train_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame],
        test_df: pd.DataFrame,
        output_dir: str = "data/processed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Save processed dataframes to Parquet files.

        Args:
            train_df: Training DataFrame
            validation_df: Validation DataFrame (can be None)
            test_df: Test DataFrame
            output_dir: Output directory path
            metadata: Additional metadata to save

        Returns:
            Dictionary mapping filenames to their paths
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save train data
        train_file = output_path / "train_processed.parquet"
        train_df.to_parquet(train_file, index=False)
        saved_files['train'] = str(train_file)
        print(f"\n✓ Saved train data: {train_file}")

        # Save validation data (if provided)
        if validation_df is not None:
            validation_file = output_path / "validation_processed.parquet"
            validation_df.to_parquet(validation_file, index=False)
            saved_files['validation'] = str(validation_file)
            print(f"✓ Saved validation data: {validation_file}")

        # Save test data
        test_file = output_path / "test_processed.parquet"
        test_df.to_parquet(test_file, index=False)
        saved_files['test'] = str(test_file)
        print(f"✓ Saved test data: {test_file}")

        # Save metadata (if provided)
        if metadata is not None:
            metadata_file = output_path / "metadata.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = str(metadata_file)
            print(f"✓ Saved metadata: {metadata_file}")

        # Combine metadata
        combined_metadata = {
            **self.metadata,
            **(metadata or {}),
        }
        if combined_metadata:
            combined_metadata_file = output_path / "preprocessing_metadata.json"
            import json
            with open(combined_metadata_file, 'w') as f:
                json.dump(combined_metadata, f, indent=2)
            saved_files['combined_metadata'] = str(combined_metadata_file)
            print(f"✓ Saved combined metadata: {combined_metadata_file}")

        print(f"\n{'='*60}")
        print(f"✓ All processed data saved successfully!")
        print(f"{'='*60}")

        return saved_files

    def load_from_parquet(self, filepath: str) -> pd.DataFrame:
        """
        Load a Parquet file.

        Args:
            filepath: Path to Parquet file

        Returns:
            DataFrame loaded from Parquet file
        """
        print(f"\n{'='*60}")
        print(f"Loading data from: {filepath}")
        print(f"{'='*60}")

        df = pd.read_parquet(filepath)
        print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        return df


def main():
    """Example usage of the Preprocessor."""
    from pathlib import Path
    from src.fase_2.data_loader import load_raw_data

    # Load raw data
    print("Loading raw data...")
    df = load_raw_data()

    # Initialize preprocessor
    preprocessor = Preprocessor(
        random_state=42,
        scaler_type="standard",
        feature_engineering=True,
        iqr_threshold=1.5,
        outliers_to_remove="all",
    )

    # Run preprocessing pipeline
    train_df, val_df, test_df, split_info = preprocessor.fit_transform(
        df,
        test_size=0.15,
        validation_size=0.15,
    )

    # Save processed data
    output_dir = "data/processed"
    saved_files = preprocessor.save_to_parquet(
        train_df,
        val_df,
        test_df,
        output_dir,
    )

    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE EXAMPLE COMPLETED")
    print("="*60)
    print("\nProcessed data files:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
