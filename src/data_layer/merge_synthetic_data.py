import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def merge_data(project_dir, synthetic_file, train_out, val_out):
    """
    Splits synthetic neutral reviews and merges them into training and validation sets.
    """
    # 1. Load Synthetic Data
    print(f"Loading synthetic data from: {synthetic_file}")
    if not os.path.exists(synthetic_file):
        print(f"Error: File {synthetic_file} not found!")
        return

    syn_df = pd.read_csv(synthetic_file)
    print(f"Synthetic Data Size: {len(syn_df)} rows")
    
    # Preprocessing Synthetic Data
    if 'text_clean' not in syn_df.columns:
         syn_df['text_clean'] = syn_df['text']
    
    aspects = ["stayingpower","texture","smell","price","colour","shipping","packing"]
    for col in aspects:
        if col not in syn_df.columns:
            syn_df[col] = "None"
        syn_df[col] = syn_df[col].fillna("None")

    # 2. Split Synthetic Data (60% Train, 40% Val)
    # We use a fixed random state for reproducibility
    syn_train, syn_val = train_test_split(syn_df, test_size=0.4, random_state=42)
    print(f"Split Synthetic Data: {len(syn_train)} for Train, {len(syn_val)} for Val")

    # 3. Merge into Training Data
    train_path = f"{project_dir}/data/splits/train.parquet"
    print(f"\nLoading original training data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    
    train_augmented = pd.concat([train_df, syn_train], ignore_index=True)
    train_out_path = f"{project_dir}/data/splits/{train_out}"
    train_augmented.to_parquet(train_out_path, index=False)
    print(f"Saved augmented training data to: {train_out_path}")
    print(f"New Training Size: {len(train_augmented)} rows")

    # 4. Merge into Validation Data
    val_path = f"{project_dir}/data/splits/val.parquet"
    print(f"\nLoading original validation data from: {val_path}")
    val_df = pd.read_parquet(val_path)
    
    val_augmented = pd.concat([val_df, syn_val], ignore_index=True)
    val_out_path = f"{project_dir}/data/splits/{val_out}"
    val_augmented.to_parquet(val_out_path, index=False)
    print(f"Saved augmented validation data to: {val_out_path}")
    print(f"New Validation Size: {len(val_augmented)} rows")

    # 5. Verification
    print("\n--- Verification: Class Counts (Augmented) ---")
    print("TRAIN - Price:\n", train_augmented['price'].value_counts())
    print("VAL   - Price:\n", val_augmented['price'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default=".", help="Root directory of the project")
    parser.add_argument("--synthetic", default="outputs/augmentation/synthetic_neutral.csv", help="Path to synthetic CSV")
    parser.add_argument("--train_out", default="train_augmented.parquet", help="Output filename for train")
    parser.add_argument("--val_out", default="val_augmented.parquet", help="Output filename for val")
    args = parser.parse_args()
    
    merge_data(args.project_dir, args.synthetic, args.train_out, args.val_out)
