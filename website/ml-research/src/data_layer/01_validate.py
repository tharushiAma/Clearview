# src/data_layer/01_validate.py

import argparse
import pandas as pd

from _common import load_cfg, setup_logger, ensure_dirs

def main(project_dir: str):

    # Load config + create folders
    project_dir, cfg = load_cfg(project_dir)
    ensure_dirs(project_dir)
    logger = setup_logger(project_dir, "01_validate")

    # Read settings from config
    raw_file = cfg["paths"]["raw_file"]
    text_col = cfg["text"]["raw_col"]
    aspects = cfg["aspects"]
    valid_labels = set(cfg["labels"])

    logger.info("Loading raw CSV file")
    df = pd.read_csv(raw_file)

    logger.info(f"Total rows loaded: {len(df)}")

    # 1️⃣ Check required columns exist
    required_cols = [text_col] + aspects
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 2️⃣ Normalize aspect labels
    # Convert values like " Positive " → "positive"
    def normalize_label(x):
        if pd.isna(x):
            return None
        x = str(x).strip().lower()
        return x if x in valid_labels else None

    for aspect in aspects:
        df[aspect] = df[aspect].apply(normalize_label)

    # 3️⃣ Remove duplicate reviews
    df[text_col] = df[text_col].astype(str).str.strip()
    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    after = len(df)

    logger.info(f"Removed duplicates: {before - after}")

    # 4️⃣ Save validated file
    out_path = f"{project_dir}/data/processed/stage1_validated.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"Saved validated data -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    args = parser.parse_args()
    main(args.project_dir)
