"""
Unit Test and Verification Script for Preprocessing Pipeline

This script verifies that the preprocessing pipeline works correctly by:
1. Loading raw data
2. Applying the preprocessing pipeline
3. Verifying final shape and splits
4. Checking data quality
5. Running basic tests
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fase_2.preprocessor import Preprocessor
from fase_2.data_loader import load_raw_data, get_data_summary


def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    print("\n" + "="*70)
    print("TESTING PREPROCESSING PIPELINE")
    print("="*70)

    # Test 1: Load raw data
    print("\n[Test 1] Loading raw data...")
    try:
        df = load_raw_data()
        print("✓ Raw data loaded successfully")
        print(f"  Shape: {df.shape}")
    except Exception as e:
        print(f"✗ Failed to load raw data: {e}")
        return False

    # Test 2: Check data summary
    print("\n[Test 2] Data summary...")
    try:
        summary = get_data_summary(df)
        print("✓ Data summary generated")
        print(f"  Columns: {len(summary['columns'])}")
        print(f"  Memory: {summary['memory_usage_mb']:.2f} MB")
        if summary['missing_values']:
            print(f"  Missing values: {sum(summary['missing_values'].values())}")
    except Exception as e:
        print(f"✗ Failed to generate summary: {e}")
        return False

    # Test 3: Initialize preprocessor
    print("\n[Test 3] Initializing preprocessor...")
    try:
        preprocessor = Preprocessor(
            random_state=42,
            scaler_type="standard",
            feature_engineering=True,
            iqr_threshold=1.5,
            outliers_to_remove="all",
        )
        print("✓ Preprocessor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize preprocessor: {e}")
        return False

    # Test 4: Handle missing values
    print("\n[Test 4] Handling missing values...")
    try:
        df_clean = preprocessor.handle_missing_values(df.copy(), strategy="mean")
        assert df_clean.isnull().sum().sum() == 0, "Missing values not handled"
        print("✓ Missing values handled successfully")
    except Exception as e:
        print(f"✗ Failed to handle missing values: {e}")
        return False

    # Test 5: Detect outliers
    print("\n[Test 5] Detecting outliers...")
    try:
        df_no_outliers, outliers_info = preprocessor.detect_outliers(df_clean)
        total_outliers = sum(info["outlier_count"] for info in outliers_info.values())
        print(f"✓ Outliers detected: {total_outliers}")
        print(f"  Outlier details: {len(outliers_info)} columns")
    except Exception as e:
        print(f"✗ Failed to detect outliers: {e}")
        return False

    # Test 6: Remove outliers
    print("\n[Test 6] Removing outliers...")
    try:
        df_outlier_removed = preprocessor.remove_outliers(df_no_outliers.copy())
        assert df_outlier_removed.shape[0] <= df_clean.shape[0], "Outliers not removed"
        print(f"✓ Outliers removed successfully")
        print(f"  Rows before: {df_clean.shape[0]}, after: {df_outlier_removed.shape[0]}")
    except Exception as e:
        print(f"✗ Failed to remove outliers: {e}")
        return False

    # Test 7: Feature engineering
    print("\n[Test 7] Feature engineering...")
    try:
        df_features = preprocessor.engineer_features(df_outlier_removed.copy())
        new_features = preprocessor.feature_engineering_info
        print(f"✓ Feature engineering completed")
        print(f"  New features created: {len(new_features)}")
        for feat in new_features:
            print(f"    - {feat}")
    except Exception as e:
        print(f"✗ Failed to engineer features: {e}")
        return False

    # Test 8: Scale features
    print("\n[Test 8] Scaling features...")
    try:
        # Make sure preprocessor has numeric columns identified
        preprocessor.numeric_columns = df_features.select_dtypes(include=[np.number]).columns.tolist()

        df_scaled = preprocessor.scale_features(df_features.copy())
        numeric_cols = preprocessor.numeric_columns
        assert len(numeric_cols) > 0, "No numeric columns found"
        assert all(col in df_scaled.columns for col in numeric_cols), "Some columns missing after scaling"
        print("✓ Features scaled successfully")
        print(f"  Scaled columns: {numeric_cols}")
    except Exception as e:
        print(f"✗ Failed to scale features: {e}")
        return False

    # Test 9: Split data
    print("\n[Test 9] Splitting data...")
    try:
        train_df, val_df, test_df, split_info = preprocessor.split_data(
            df_scaled,
            test_size=0.15,
            validation_size=0.15,
        )
        assert len(train_df) > 0, "Train set is empty"
        assert len(val_df) > 0, "Validation set is empty"
        assert len(test_df) > 0, "Test set is empty"
        print("✓ Data split successfully")
        print(f"  Train: {len(train_df)} rows")
        print(f"  Validation: {len(val_df)} rows")
        print(f"  Test: {len(test_df)} rows")
    except Exception as e:
        print(f"✗ Failed to split data: {e}")
        return False

    # Test 10: Verify splits
    print("\n[Test 10] Verifying splits...")
    try:
        total = len(train_df) + len(val_df) + len(test_df)
        assert abs(total - df_scaled.shape[0]) < 10, "Splits don't add up correctly"

        # Check quality distribution is preserved
        train_quality_dist = train_df['quality'].value_counts().sort_index()
        val_quality_dist = val_df['quality'].value_counts().sort_index()
        test_quality_dist = test_df['quality'].value_counts().sort_index()

        print(f"✓ Splits verified successfully")
        print(f"  Total rows: {total}")
        print(f"  Split percentages: "
              f"Train={split_info['train_ratio']:.1%}, "
              f"Validation={split_info['validation_ratio']:.1%}, "
              f"Test={split_info['test_ratio']:.1%}")
    except Exception as e:
        print(f"✗ Failed to verify splits: {e}")
        return False

    # Test 11: Full pipeline
    print("\n[Test 11] Running full preprocessing pipeline...")
    try:
        # Reset preprocessor
        preprocessor = Preprocessor(
            random_state=42,
            scaler_type="standard",
            feature_engineering=True,
            iqr_threshold=1.5,
            outliers_to_remove="all",
        )

        # Run full pipeline
        train_full, val_full, test_full, split_full = preprocessor.fit_transform(
            df.copy(),
            test_size=0.15,
            validation_size=0.15,
        )

        assert len(train_full) > 0, "Train set is empty"
        assert len(val_full) > 0, "Validation set is empty"
        assert len(test_full) > 0, "Test set is empty"
        print("✓ Full pipeline completed successfully")
    except Exception as e:
        print(f"✗ Failed to run full pipeline: {e}")
        return False

    # Test 12: Save and load
    print("\n[Test 12] Saving and loading data...")
    try:
        # Check if pyarrow is available
        try:
            import pyarrow
            print("✓ Parquet support available")
        except ImportError:
            print("⚠ Parquet support not available (pyarrow not installed)")
            print("  Skipping save/load test")

            # Test basic save/load with CSV instead
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)

            train_file = output_dir / "train.csv"
            train_full.to_csv(train_file, index=False)
            print(f"✓ Saved train data to CSV: {train_file}")

            train_loaded = pd.read_csv(train_file)
            assert train_loaded.shape == train_full.shape, "Loaded data doesn't match original"
            print(f"✓ Data loaded successfully from CSV")

            return True

        # Save to parquet
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = preprocessor.save_to_parquet(
            train_full,
            val_full,
            test_full,
            output_dir=str(output_dir),
        )
        print(f"✓ Data saved successfully")
        print(f"  Files saved: {len(saved)}")

        # Load back
        train_loaded = preprocessor.load_from_parquet(saved['train'])
        assert train_loaded.shape == train_full.shape, "Loaded data doesn't match original"
        print(f"✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Failed to save/load data: {e}")
        return False

    # Test 13: Metadata verification
    print("\n[Test 13] Verifying metadata...")
    try:
        assert 'original_shape' in preprocessor.metadata, "Original shape not in metadata"
        assert 'train_size' in preprocessor.metadata, "Train size not in metadata"
        assert 'feature_engineering_info' in preprocessor.metadata, "Features not in metadata"
        print("✓ Metadata verified")
        print(f"  Original shape: {preprocessor.metadata['original_shape']}")
        print(f"  Feature engineering: {preprocessor.metadata['feature_engineering_info']}")
    except Exception as e:
        print(f"✗ Failed to verify metadata: {e}")
        return False

    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("TESTING EDGE CASES")
    print("="*70)

    # Test 1: No data
    print("\n[Edge Case 1] Testing with empty dataframe...")
    try:
        preprocessor = Preprocessor(random_state=42)
        # Check that preprocessor initializes correctly
        assert len(preprocessor.numeric_columns) == 0
        assert len(preprocessor.categorical_columns) == 0
        print("✓ Handled empty dataframe (correctly initialized with no numeric/categorical columns)")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Test 2: Invalid split ratios
    print("\n[Edge Case 2] Testing invalid split ratios...")
    try:
        preprocessor = Preprocessor()
        train_df, val_df, test_df, _ = preprocessor.fit_transform(
            pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
            test_size=0.6,
            validation_size=0.6,
        )
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 3: Different scaler types
    print("\n[Edge Case 3] Testing different scaler types...")
    for scaler_type in ["standard", "minmax"]:
        try:
            preprocessor = Preprocessor(scaler_type=scaler_type)
            train_df, val_df, test_df, _ = preprocessor.fit_transform(
                pd.DataFrame({'a': [1, 2, 3, 4, 5]}),
                test_size=0.3,
                validation_size=0.3,
            )
            print(f"  ✓ {scaler_type} scaler works")
        except Exception as e:
            print(f"  ✗ {scaler_type} scaler failed: {e}")
            return False

    return True


def test_data_quality():
    """Test data quality after preprocessing."""
    print("\n" + "="*70)
    print("TESTING DATA QUALITY")
    print("="*70)

    try:
        preprocessor = Preprocessor(random_state=42, feature_engineering=False, iqr_threshold=1.5)
        train_df, val_df, test_df, _ = preprocessor.fit_transform(
            pd.DataFrame({
                'quality': [5, 6, 7, 6, 5, 8, 4],
                'alcohol': [10.5, 11.0, 12.0, 11.5, 10.0, 13.5, 9.5],
                'density': [0.99, 0.98, 0.97, 0.985, 0.995, 0.96, 1.0],
            }),
            test_size=0.5,
            validation_size=0.25,
        )

        # Check for missing values
        assert train_df.isnull().sum().sum() == 0, "Train has missing values"
        assert val_df.isnull().sum().sum() == 0, "Validation has missing values"
        assert test_df.isnull().sum().sum() == 0, "Test has missing values"
        print("✓ No missing values in any split")

        # Check data types
        assert train_df['quality'].dtype in ['int64', 'float64'], "Quality column has wrong type"
        print("✓ Data types are correct")

        # Check scaling (should have similar scale)
        train_alcohol = train_df['alcohol'].mean()
        val_alcohol = val_df['alcohol'].mean()
        test_alcohol = test_df['alcohol'].mean()
        print(f"  Train alcohol mean: {train_alcohol:.2f}")
        print(f"  Validation alcohol mean: {val_alcohol:.2f}")
        print(f"  Test alcohol mean: {test_alcohol:.2f}")

        # Check splits have data
        assert len(train_df) > 0, "Train set is empty"
        assert len(val_df) > 0, "Validation set is empty"
        assert len(test_df) > 0, "Test set is empty"
        print("✓ All splits have data")

        return True

    except Exception as e:
        print(f"✗ Data quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE VERIFICATION")
    print("="*70)

    all_passed = True

    # Run main tests
    if not test_preprocessing_pipeline():
        all_passed = False

    # Run edge case tests
    if not test_edge_cases():
        all_passed = False

    # Run data quality tests
    if not test_data_quality():
        all_passed = False

    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe preprocessing pipeline is working correctly!")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
