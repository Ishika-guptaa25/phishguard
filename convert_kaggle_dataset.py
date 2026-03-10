# convert_kaggle_dataset.py - Fixed version for feature-based datasets

import pandas as pd
import json
import os


def convert_kaggle_dataset():
    print("=" * 60)
    print("KAGGLE PHISHING DATASET CONVERTER")
    print("=" * 60)

    print("\nReading dataset_full.csv...")
    df = pd.read_csv("dataset_full.csv")

    cols = list(df.columns)
    print(f"Columns: {cols[:5]}...")
    print(f"Total columns: {len(cols)}")

    # Detect the label column (phishing/legitimate indicator)
    label_col = None
    for candidate in ['phishing', 'label', 'class', 'target', 'status']:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        # Try last column as label
        label_col = cols[-1]
        print(f"Assuming last column '{label_col}' is the label.")

    print(f"\nLabel column: '{label_col}'")
    print(f"Label distribution:\n{df[label_col].value_counts()}")

    # Feature columns = everything except the label
    feature_cols = [c for c in cols if c != label_col]

    # Convert to the format your system expects
    # Each row becomes a dict of features + label
    records = []
    for _, row in df.iterrows():
        record = {
            "features": {col: row[col] for col in feature_cols},
            "label": int(row[label_col])  # 1 = phishing, 0 = legitimate
        }
        records.append(record)

    # Save as JSONL for your detection system
    output_path = "converted_dataset.jsonl"
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\n✅ Converted {len(records)} records → {output_path}")
    print(f"   Phishing: {df[label_col].sum()}")
    print(f"   Legitimate: {len(df) - df[label_col].sum()}")

    # Also save a simple CSV with just features + label
    df.to_csv("dataset_converted.csv", index=False)
    print(f"✅ Also saved clean CSV → dataset_converted.csv")


if __name__ == "__main__":
    convert_kaggle_dataset()