# -*- coding: utf-8 -*-
"""Google Translator - Modified for local execution

Original file was designed for Google Colab.
This version is adapted to run on Windows with local file paths.
"""

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm  # Using standard tqdm instead of tqdm.notebook
import os
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Load file - UPDATE THIS PATH to point to your CSV file
# Example: using the reviews.csv found in your project
input_file = r"c:\Users\TharushiAmasha\OneDrive - inivosglobal.com\FYP\Clearview\data\raw\reviews.csv"
output_file = r"c:\Users\TharushiAmasha\OneDrive - inivosglobal.com\FYP\Clearview\ml-research\notebooks\testData_translated.xlsx"

# Check if input file exists
if not os.path.exists(input_file):
    print(f"ERROR: Input file not found at: {input_file}")
    print("\nPlease update the 'input_file' variable with the correct path to your CSV file.")
    exit(1)

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows")
print("\nFirst 5 rows:")
print(df.head())

# 2. Setup translator
translator = GoogleTranslator(source="auto", target="en")
tqdm.pandas()

def safe_translate(text):
    """Safely translate text, handling edge cases and errors"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return text
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"\nTranslation error: {e}")
        return text  # fallback if API rate-limits

# 3. Translate review text
# NOTE: Adjust the column name 'data' to match your actual column name
if 'data' in df.columns:
    print("\nTranslating 'data' column...")
    df["data_en"] = df["data"].progress_apply(safe_translate)
elif 'review' in df.columns:
    print("\nTranslating 'review' column...")
    df["review_en"] = df["review"].progress_apply(safe_translate)
elif 'text' in df.columns:
    print("\nTranslating 'text' column...")
    df["text_en"] = df["text"].progress_apply(safe_translate)
else:
    print("\nERROR: Could not find a text column to translate!")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# 4. Optional: rename columns for your project
if 'data_en' in df.columns:
    df = df.rename(columns={"data_en": "review_en"})

# 5. Save translated dataset
print(f"\nSaving to: {output_file}")
df.to_excel(output_file, index=False)
print(f"✓ Successfully saved → {output_file}")
print(f"Total rows translated: {len(df)}")
