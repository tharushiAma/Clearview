"""
convert_synthetic_robust.py
Convert JSON files of LLM-generated synthetic reviews into standardised CSV
format that matches the schema used by the rest of the data pipeline.

Pipeline:
  Input : data/augmented/*.json   (raw LLM output, potentially malformed JSON)
  Output: data/augmented/*.csv    (one CSV per JSON, same base filename)

Each output CSV has the columns:
  data, <aspects...>, text_clean, signature

Usage:
    python data/data_layer/convert_synthetic_robust.py --project_dir <path>

Notes:
  - JSON extraction uses a non-nested regex strategy; each review object must
    not contain nested { } braces (true for this schema).
  - Config is loaded from <project_dir>/configs/config.yaml.
"""

import os
import json
import argparse
import logging
import pandas as pd
import re
import yaml
import emoji
from cleantext import clean
from pathlib import Path


# ── Inline utilities (replaced _common dependency) ────────────────────────────

def load_cfg(project_dir: str) -> tuple:
    """Load configs/config.yaml and return (project_dir_path, cfg dict)."""
    root = Path(project_dir).resolve()
    cfg_path = root / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return root, cfg


def setup_logger(project_dir: Path, name: str) -> logging.Logger:
    """Set up a logger that writes to both console and a log file."""
    log_dir = Path(project_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File handler
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dirs(project_dir: Path):
    """Create standard output directories if they don't already exist."""
    for sub in ("data/augmented", "data/splits", "logs", "results"):
        (Path(project_dir) / sub).mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Lightweight text cleaning for synthetic reviews.
    Applies the same normalisation steps used in the main preprocessing
    pipeline so that synthetic and real reviews share the same text format.

    Steps:
      1. Strip HTML tags
      2. Replace emojis with a space
      3. Collapse whitespace
      4. Lowercase, remove URLs / emails / line breaks (via cleantext)
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
        no_line_breaks=True,
    )
    return text.strip()


def make_signature(row: pd.Series, aspects: list) -> str:
    """
    Build a compact deduplication key for a review row by concatenating all
    aspect label values.

    Example output: "stayingpower:na|texture:positive|smell:na|..."
    Missing / null aspect values are represented as 'na'.
    """
    return "|".join(
        f"{a}:{row[a] if pd.notna(row[a]) and row[a] is not None else 'na'}"
        for a in aspects
    )

def extract_json_objects(file_path, logger):
    """
    Robustly extract all JSON objects { ... } from a file, 
    even if the overall structure (commas, brackets) is broken.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple regex to find content between { and }
    # This assumes no nested objects in the synthetic data (which is true for this schema)
    # If there were nested objects, we'd need a more complex balancer.
    matches = re.findall(r'\{[^{}]+\}', content, re.DOTALL)
    
    data = []
    for match in matches:
        try:
            obj = json.loads(match)
            data.append(obj)
        except json.JSONDecodeError:
            # Try to fix common issues like trailing commas or unquoted keys if necessary
            # but usually the individual objects are fine.
            continue
            
    if not data:
        logger.error(f"No valid JSON objects found in {file_path.name}")
        return None
        
    logger.info(f"Extracted {len(data)} objects from {file_path.name}")
    return data

def main(project_dir: str):
    project_dir, cfg = load_cfg(project_dir)
    ensure_dirs(project_dir)
    logger = setup_logger(project_dir, "convert_synthetic_robust")
    
    aspects = cfg["aspects"]["names"]  # list of aspect name strings from config.yaml
    augmented_dir = Path(project_dir) / "data" / "augmented"
    
    json_files = list(augmented_dir.glob("*.json"))
    for json_path in json_files:
        logger.info(f"Processing {json_path.name}...")
        
        data = extract_json_objects(json_path, logger)
        if data is None:
            continue
            
        df = pd.DataFrame(data)
        if "review_text" in df.columns:
            df = df.rename(columns={"review_text": "data"})
            
        # Ensure all columns match target
        for a in aspects:
            if a not in df.columns:
                df[a] = None
                
        # Handle cases where some rows might have 'data' missing or null
        df = df[df["data"].notna()]
        
        df["text_clean"] = df["data"].apply(clean_text)
        df["signature"] = df.apply(lambda row: make_signature(row, aspects), axis=1)
        
        cols = ["data"] + aspects + ["text_clean", "signature"]
        df = df[cols]
        
        csv_path = json_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV -> {csv_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    args = parser.parse_args()
    main(args.project_dir)
