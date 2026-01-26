import pandas as pd
import os

data_dir = 'data/splits'
files = ['train.parquet', 'train_augmented.parquet', 'val.parquet', 'val_augmented.parquet']

print("Available datasets:")
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  {f:30s}: {len(df):6d} rows")
