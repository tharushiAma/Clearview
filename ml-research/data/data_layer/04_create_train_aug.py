import os
import argparse
import pandas as pd
from pathlib import Path
from _common import load_cfg, setup_logger, ensure_dirs

def main(project_dir: str):
    project_dir, cfg = load_cfg(project_dir)
    ensure_dirs(project_dir)
    logger = setup_logger(project_dir, "04_create_train_aug")
    
    splits_dir = Path(cfg["paths"]["splits_dir"])
    augmented_dir = Path(project_dir) / "data" / "augmented"
    
    # 1. Load original training data
    train_path = splits_dir / "train.parquet"
    if not train_path.exists():
        logger.error(f"Original training file not found: {train_path}")
        return
        
    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded original train: {len(train_df)} records")
    
    # 2. Load all synthetic CSVs
    aug_dfs = []
    csv_files = list(augmented_dir.glob("*.csv"))
    
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        aug_dfs.append(df)
        logger.info(f"Loaded {csv_path.name}: {len(df)} records")
        
    if not aug_dfs:
        logger.warning("No augmented CSV files found to merge.")
        return
        
    # 3. Concatenate and deduplicate
    synthetic_df = pd.concat(aug_dfs, ignore_index=True)
    logger.info(f"Total synthetic records: {len(synthetic_df)}")
    
    # Internal deduplication (LLMs often repeat exact sentences)
    initial_count = len(synthetic_df)
    synthetic_df = synthetic_df.drop_duplicates(subset=["data"])
    dupe_count = initial_count - len(synthetic_df)
    logger.info(f"Removed {dupe_count} internal synthetic duplicates. Clean synthetic: {len(synthetic_df)}")

    # Overlap check with original training data
    synthetic_df = synthetic_df[~synthetic_df["data"].isin(train_df["data"])]
    logger.info(f"Final synthetic records after training overlap check: {len(synthetic_df)}")
    
    # Combined augmented train
    combined_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    logger.info(f"Combined augmented train: {len(combined_df)} records")
    
    # 4. Save to separate files
    aug_train_parquet = splits_dir / "train_aug.parquet"
    aug_train_csv = splits_dir / "train_aug.csv"
    
    combined_df.to_parquet(aug_train_parquet, index=False)
    combined_df.to_csv(aug_train_csv, index=False)
    
    logger.info(f"Saved augmented train to {aug_train_parquet.name} and {aug_train_csv.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    args = parser.parse_args()
    main(args.project_dir)
