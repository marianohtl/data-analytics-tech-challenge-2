"""Task 02: Download & Verify Wine Quality Dataset

This script downloads the Wine Quality dataset from Kaggle and verifies it loads correctly.
Dataset: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import shutil
from pathlib import Path

def download_and_verify():
    """
    Downloads the wine quality dataset and verifies its structure.
    """
    print("Downloading wine quality dataset from Kaggle...")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
        print(f"Dataset downloaded to: {path}")
        
        # Load the dataset - WineQT.csv contains both red and white wine
        wineqt_path = f"{path}/WineQT.csv"
        
        # Load dataset
        df = pd.read_csv(wineqt_path)
        print(f"\n{'='*60}")
        print("WINE QUALITY DATASET LOADED (WineQT)")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nColumn names:")
        print(df.columns.tolist())
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        # Verify dataset integrity
        print(f"\n{'='*60}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total rows: {len(df)}, columns: {len(df.columns)}")
        
        # Save to raw data directory
        raw_data_dir = Path("data/raw")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(wineqt_path, raw_data_dir / "winequality.csv")
        print(f"\n✓ Dataset saved to: {raw_data_dir}/winequality.csv")
        
        # Also create separate files for red and white wine if needed
        # Check if there's a quality column that indicates red vs white
        if 'type' in df.columns:
            print(f"\n✓ Dataset includes 'type' column with values: {df['type'].unique()}")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    download_and_verify()
