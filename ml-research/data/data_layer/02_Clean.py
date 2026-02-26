# src/data_layer/02_clean.py

import argparse
import pandas as pd
import re
import emoji
from cleantext import clean
from tqdm import tqdm

from _common import load_cfg, setup_logger, ensure_dirs

def clean_text(text):
    """
    Basic text cleaning:
    - remove HTML
    - remove emojis
    - lowercase
    - remove URLs and emails
    """
    text = str(text)

    text = re.sub(r"<.*?>", " ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"\s+", " ", text)

    text = clean(
        text,
        lower=True,
        no_urls=True,
        no_emails=True,
        no_line_breaks=True
    )

    return text.strip()

def main(project_dir: str):

    project_dir, cfg = load_cfg(project_dir)
    ensure_dirs(project_dir)
    logger = setup_logger(project_dir, "02_clean")

    text_col = cfg["text"]["raw_col"]
    aspects = cfg["aspects"]
    valid_labels = set(cfg["labels"])

    in_path = f"{project_dir}/data/processed/stage1_validated.parquet"
    out_path = cfg["paths"]["processed_file"]

    logger.info("Loading validated data")
    df = pd.read_parquet(in_path)

    # Clean review text
    logger.info("Cleaning review text")
    tqdm.pandas(desc="Cleaning Text")
    df["text_clean"] = df[text_col].progress_apply(clean_text)


    # Ensure aspect labels are valid
    for a in aspects:
        df[a] = df[a].apply(lambda x: x if x in valid_labels else None)

    df.to_parquet(out_path, index=False)
    logger.info(f"Saved cleaned data -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    args = parser.parse_args()
    main(args.project_dir)
