#!/usr/bin/env python
"""
Example usage of the preprocessing pipeline.

This script demonstrates how to use the Preprocessor class to preprocess
the wine quality dataset.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fase_2.preprocessor import Preprocessor
from fase_2.data_loader import load_raw_data


def main():
    """Run the preprocessing pipeline example."""
    print("="*70)
    print("PREPROCESSING PIPELINE EXAMPLE")
    print("="*70)

    # Load raw data
    print("\n1. Loading raw data...")
    df = load_raw_data()
    print(f"   Loaded {df.shape[0]} samples with {df.shape[1]} features\n")

    # Initialize preprocessor
    print("2. Initializing preprocessor...")
    preprocessor = Preprocessor(
        random_state=42,
        scaler_type="standard",      # Use StandardScaler
        feature_engineering=True,     # Enable feature engineering
        iqr_threshold=1.5,            # IQR multiplier for outlier detection
        outliers_to_remove="all",    # Remove all outliers
    )
    print("   ✓ Preprocessor initialized with custom settings\n")

    # Run full preprocessing pipeline
    print("3. Running preprocessing pipeline...")
    train_df, val_df, test_df, split_info = preprocessor.fit_transform(
        df,
        test_size=0.15,
        validation_size=0.15,
    )
    print(f"   ✓ Pipeline completed")
    print(f"   Train: {len(train_df)} ({split_info['train_ratio']:.1%})")
    print(f"   Validation: {len(val_df)} ({split_info['validation_ratio']:.1%})")
    print(f"   Test: {len(test_df)} ({split_info['test_ratio']:.1%})\n")

    # Display new features created
    if preprocessor.feature_engineering_info:
        print("4. Features created:")
        for feat in preprocessor.feature_engineering_info:
            print(f"   - {feat}")
        print()

    # Save processed data
    print("5. Saving processed data...")
    output_dir = Path("data/processed")
    saved_files = preprocessor.save_to_parquet(
        train_df,
        val_df,
        test_df,
        output_dir=str(output_dir),
    )
    print(f"   ✓ Data saved to {output_dir}")
    print()
    for name, path in saved_files.items():
        print(f"   {name}: {path}")

    # Display metadata
    print("\n6. Processing metadata:")
    print(f"   Original shape: {preprocessor.metadata['original_shape']}")
    print(f"   Features after preprocessing: {len(preprocessor.numeric_columns) + len(preprocessor.categorical_columns)}")
    print(f"   Feature engineering applied: {preprocessor.feature_engineering}")
    print(f"   Outliers removed: {preprocessor.metadata.get('outliers_removed', 0)}")

    print("\n" + "="*70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
