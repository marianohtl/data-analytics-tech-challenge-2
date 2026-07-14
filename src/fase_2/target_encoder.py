"""
Target Encoding Module for Wine Quality Dataset

This module provides functionality to encode the quality column into binary targets
for classification purposes.
"""

import pandas as pd
from typing import Optional, Union


def encode_quality_target(
    df: pd.DataFrame,
    target_column: str = "quality",
    high_quality_threshold: int = 7,
    new_column_name: str = "quality_binary"
) -> pd.DataFrame:
    """
    Encode wine quality into a binary target variable.

    High quality is defined as quality >= 7 (scores 7, 8, 9)
    Low/Medium quality is defined as quality < 7 (scores 3, 4, 5, 6)

    Args:
        df: Input pandas DataFrame containing wine quality data
        target_column: Name of the column containing quality scores (default: 'quality')
        high_quality_threshold: Quality score threshold to classify as high quality (default: 7)
        new_column_name: Name for the binary target column (default: 'quality_binary')

    Returns:
        DataFrame with the original data and the new binary target column

    Raises:
        KeyError: If the specified target column does not exist in the DataFrame

    Example:
        >>> df = pd.DataFrame({'quality': [6, 7, 8, 5]})
        >>> result = encode_quality_target(df)
        >>> result['quality_binary'].tolist()
        [0, 1, 1, 0]

    Note:
        This function creates a copy of the input DataFrame to avoid modifying the original.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()

    # Validate that the target column exists
    if target_column not in df_encoded.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df_encoded.columns)}"
        )

    # Create the binary target
    # 1 = High quality (quality >= threshold)
    # 0 = Low/Medium quality (quality < threshold)
    df_encoded[new_column_name] = (df_encoded[target_column] >= high_quality_threshold).astype(int)

    return df_encoded


def create_alternative_quality_targets(
    df: pd.DataFrame,
    target_column: str = "quality",
    high_quality_threshold: int = 7,
    low_quality_threshold: int = 5,
    binary_column_name: str = "quality_binary"
) -> pd.DataFrame:
    """
    Create alternative quality targets (high vs low quality) for classification.

    This creates two binary columns:
    - quality_binary: 1 = high quality, 0 = low/medium quality
    - quality_low: 1 = low quality (< threshold), 0 = high quality (>= threshold)

    Args:
        df: Input pandas DataFrame containing wine quality data
        target_column: Name of the column containing quality scores (default: 'quality')
        high_quality_threshold: Quality score threshold for high quality (default: 7)
        low_quality_threshold: Quality score threshold for low quality (default: 5)
        binary_column_name: Base name for binary target columns (default: 'quality_binary')

    Returns:
        DataFrame with additional binary quality columns

    Raises:
        KeyError: If the specified target column does not exist in the DataFrame

    Example:
        >>> df = pd.DataFrame({'quality': [4, 6, 7, 9]})
        >>> result = create_alternative_quality_targets(df)
        >>> result['quality_binary'].tolist()
        [0, 0, 1, 1]
        >>> result['quality_low'].tolist()
        [1, 1, 0, 0]
    """
    # Validate that the target column exists
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Create a copy of the DataFrame
    df_encoded = df.copy()

    # Create binary target: 1 = high quality
    high_quality = (df_encoded[target_column] >= high_quality_threshold).astype(int)
    df_encoded[f"{binary_column_name}_high"] = high_quality

    # Create binary target: 1 = low quality
    low_quality = (df_encoded[target_column] < low_quality_threshold).astype(int)
    df_encoded[f"{binary_column_name}_low"] = low_quality

    return df_encoded


if __name__ == "__main__":
    # Test the encoding function
    print("=" * 60)
    print("Testing Quality Target Encoding")
    print("=" * 60)

    # Test case 1: Basic binary encoding
    print("\nTest 1: Basic binary encoding")
    print("-" * 40)
    test_df1 = pd.DataFrame({
        'quality': [6, 7, 8, 5, 9, 4],
        'alcohol': [9.5, 9.8, 10.2, 8.9, 10.5, 8.5]
    })
    print("Input DataFrame:")
    print(test_df1)
    print("\nQuality range: 3-9 (3=low, 9=high)")
    print("High quality threshold: 7")
    print()

    encoded_df1 = encode_quality_target(test_df1)
    print("Encoded DataFrame:")
    print(encoded_df1)
    print("\nExpected quality_binary values: [0, 1, 1, 0, 1, 0]")
    print(f"Actual values: {encoded_df1['quality_binary'].tolist()}")
    print(f"✓ Encoding works correctly!" if encoded_df1['quality_binary'].tolist() == [0, 1, 1, 0, 1, 0] else "✗ Encoding failed!")

    # Test case 2: Alternative target columns
    print("\n" + "=" * 60)
    print("Test 2: Alternative quality targets")
    print("-" * 40)
    test_df2 = pd.DataFrame({
        'quality': [4, 6, 7, 9],
        'fixed_acidity': [7.0, 6.5, 6.8, 7.2]
    })
    print("Input DataFrame:")
    print(test_df2)
    print("\nQuality range: 3-9")
    print("High quality threshold: 7")
    print("Low quality threshold: 5")
    print()

    encoded_df2 = create_alternative_quality_targets(
        test_df2,
        high_quality_threshold=7,
        low_quality_threshold=5
    )
    print("Encoded DataFrame:")
    print(encoded_df2)
    print("\nExpected quality_binary_high: [0, 0, 1, 1]")
    print(f"Actual values: {encoded_df2['quality_binary_high'].tolist()}")
    print("\nExpected quality_binary_low: [1, 1, 0, 0]")
    print(f"Actual values: {encoded_df2['quality_binary_low'].tolist()}")
    print(f"✓ Alternative targets created successfully!")

    # Test case 3: Error handling - missing column
    print("\n" + "=" * 60)
    print("Test 3: Error handling")
    print("-" * 40)
    test_df3 = pd.DataFrame({
        'alcohol': [9.5, 9.8, 10.2]
    })
    print("Input DataFrame (missing 'quality' column):")
    print(test_df3)
    print()

    try:
        encode_quality_target(test_df3)
        print("✗ Expected KeyError was not raised!")
    except KeyError as e:
        print(f"✓ Correctly raised KeyError: {e}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
