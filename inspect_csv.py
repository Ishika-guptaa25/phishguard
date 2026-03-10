#!/usr/bin/env python3
"""
Simple CSV Inspector - Shows exactly what's in your dataset_full.csv
"""

import pandas as pd
import os


def inspect_csv():
    if not os.path.exists('dataset_full.csv'):
        print("❌ dataset_full.csv not found!")
        print("Download from: https://www.kaggle.com/datasets/shashwatwork/phishing-website-dataset")
        return

    print("=" * 70)
    print("INSPECTING: dataset_full.csv")
    print("=" * 70)

    try:
        # Read just the header
        df = pd.read_csv('dataset_full.csv', nrows=5)

        print(f"\nFile size: {os.path.getsize('dataset_full.csv') / (1024 * 1024):.1f} MB")
        print(f"Shape: {df.shape}")
        print(f"\nColumn names ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:3}. {col}")

        print("\n" + "=" * 70)
        print("FIRST ROW DATA")
        print("=" * 70)
        for col in df.columns[:10]:  # Show first 10 columns
            val = str(df[col].iloc[0])[:80]
            print(f"{col:20} = {val}")

        print("\n" + "=" * 70)
        print("DATA TYPES")
        print("=" * 70)
        for col in df.columns[:10]:
            print(f"{col:20} -> {df[col].dtype}")

        # Check for URL-like columns
        print("\n" + "=" * 70)
        print("SEARCHING FOR URL COLUMN")
        print("=" * 70)

        found_url = False
        for col in df.columns:
            val = str(df[col].iloc[0]).lower()
            if 'http' in val or val.startswith('www') or (val.count('.') > 1 and len(val) > 10):
                print(f"✓ FOUND: '{col}'")
                print(f"  Sample: {val[:80]}")
                found_url = True
                break

        if not found_url:
            print("❌ Could not find URL column")
            print("   Check if column 1 or 2 looks like URLs")

        # Check for label-like columns
        print("\n" + "=" * 70)
        print("SEARCHING FOR LABEL COLUMN")
        print("=" * 70)

        found_label = False
        for col in df.columns:
            try:
                unique_vals = df[col].unique()
                if len(unique_vals) <= 5:
                    print(f"Checking '{col}': {unique_vals.tolist()}")
                    if all(v in [0, 1, '0', '1', True, False] for v in unique_vals):
                        print(f"✓ FOUND: '{col}' (values: {unique_vals.tolist()})")
                        found_label = True
                        break
            except:
                pass

        if not found_label:
            print("❌ Could not find label column")
            print("   Should contain only 0 and 1")

        print("\n" + "=" * 70)
        print("WHAT TO DO")
        print("=" * 70)

        if found_url and found_label:
            print("✓ Columns found! Run: python fix_dataset.py")
        else:
            print("❌ Unable to find required columns")
            print("Check the column names above and tell me:")
            print("  1. Which column contains URLs?")
            print("  2. Which column contains labels (0/1)?")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    inspect_csv()