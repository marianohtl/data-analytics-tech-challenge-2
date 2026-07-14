"""
Data Loader Module for Wine Quality Dataset

This module provides functionality to load and validate the wine quality dataset.
Dataset source: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
Target variable: 'quality' (integer rating from 3 to 9)
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw wine quality dataset from CSV file.

    Args:
        path: Path to CSV file. If None, looks for 'data/raw/winequality.csv'
              or 'data/raw/winequality-red.csv' in that order.

    Returns:
        DataFrame with wine quality data

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file format is invalid
    """
    if path is None:
        # Try multiple possible filenames
        possible_paths = [
            "data/raw/winequality.csv",
            "data/raw/winequality-red.csv",
            "data/raw/winequality-white.csv",
        ]

        for p in possible_paths:
            if Path(p).exists():
                path = p
                break

        if path is None:
            raise FileNotFoundError(
                "No wine quality dataset found. "
                "Please download from Kaggle or provide a valid path."
            )

    # Load the CSV file
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}")

    # Display basic information
    print(f"\n{'='*60}")
    print(f"DATASET LOADED: {path}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    print(f"\nData types:")
    print(df.dtypes.to_string())
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0].to_string())
    else:
        print("No missing values")

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with dataset statistics
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "quality_distribution": df.get('quality', pd.Series()).value_counts().to_dict(),
    }


if __name__ == "__main__":
    # Test the loader
    df = load_raw_data()
    print(f"\n{'='*60}")
    print("✓ Data loader test completed successfully!")
    print(f"{'='*60}")
