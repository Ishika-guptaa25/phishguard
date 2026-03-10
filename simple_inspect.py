#!/usr/bin/env python3
import pandas as pd
import os

# Check file exists
if not os.path.exists('dataset_full.csv'):
    print("ERROR: dataset_full.csv not found!")
    exit(1)

# Read CSV
print("Reading dataset_full.csv...")
df = pd.read_csv('dataset_full.csv', nrows=2)

print("\nFile size:", os.path.getsize('dataset_full.csv') / (1024*1024), "MB")
print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst row:")
print(df.iloc[0].to_dict())
print("\nData types:")
print(df.dtypes.to_dict())