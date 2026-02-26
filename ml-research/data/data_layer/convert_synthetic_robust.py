import os
import json
import argparse
import pandas as pd
import re
import emoji
from cleantext import clean
from pathlib import Path
from _common import load_cfg, setup_logger, ensure_dirs

def clean_text(text):
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

def make_signature(row, aspects):
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
    
    aspects = cfg["aspects"]
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
