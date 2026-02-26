# src/data_layer/03_split.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from _common import load_cfg, setup_logger, ensure_dirs

def convert_parquet_to_csv(splits_dir: str, logger):
    """Converts the parquet splits to CSV format."""
    files = ["test", "train", "val"]
    for file_name in files:
        parquet_path = os.path.join(splits_dir, f"{file_name}.parquet")
        csv_path = os.path.join(splits_dir, f"{file_name}.csv")

        if os.path.exists(parquet_path):
            logger.info(f"Converting {parquet_path} to CSV...")
            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
        else:
            logger.warning(f"File not found for conversion: {parquet_path}")

def main(project_dir: str):

    project_dir, cfg = load_cfg(project_dir)
    ensure_dirs(project_dir)
    logger = setup_logger(project_dir, "03_split")

    aspects = cfg["aspects"]
    split_cfg = cfg["split"]

    df = pd.read_parquet(cfg["paths"]["processed_file"])

    # Create a label signature for stratification
    def make_signature(row):
        return "|".join(
            f"{a}:{row[a] if pd.notna(row[a]) else 'na'}"
            for a in aspects
        )

    df["signature"] = df.apply(make_signature, axis=1)

    # Split data
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["seed"],
            stratify=df["signature"]
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=split_cfg["val_size"] / (1 - split_cfg["test_size"]),
            random_state=split_cfg["seed"],
            stratify=train_df["signature"]
        )
    except ValueError as e:
        logger.warning(f"Stratification failed (likely due to rare classes): {e}")
        logger.warning("Falling back to random split.")
        train_df, test_df = train_test_split(
            df,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["seed"]
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=split_cfg["val_size"] / (1 - split_cfg["test_size"]),
            random_state=split_cfg["seed"]
        )

    # Save splits
    splits_dir = cfg["paths"]["splits_dir"]
    train_df.to_parquet(f"{splits_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{splits_dir}/val.parquet", index=False)
    test_df.to_parquet(f"{splits_dir}/test.parquet", index=False)

    logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Convert to CSV
    convert_parquet_to_csv(splits_dir, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    args = parser.parse_args()
    main(args.project_dir)
