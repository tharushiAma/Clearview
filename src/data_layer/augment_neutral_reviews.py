import pandas as pd
import argparse
import os
import random

def extract_candidates(project_dir, output_file, num_samples=50):
    """
    Extracts Positive reviews for 'Price' and 'Packing' aspects to be rewritten as Neutral.
    """
    train_path = f"{project_dir}/data/splits/train.parquet"
    print(f"Loading training data from: {train_path}")
    df = pd.read_parquet(train_path)

    # Filter for Positive Price or Packing
    # Logic: It's easier to tone down a "Great price" to "Okay price" 
    # than to upgrade a "Bad price" or invent one from scratch.
    candidates = df[
        (df['price'] == 'positive') | (df['packing'] == 'positive')
    ].copy()

    print(f"Found {len(candidates)} total positive candidates.")

    # Sample if we have too many
    if len(candidates) > num_samples:
        candidates = candidates.sample(n=num_samples, random_state=42)
    
    # Prepare Output
    output_rows = []
    
    print("\n--- Generating Prompts ---")
    for idx, row in candidates.iterrows():
        text = row['text_clean']
        
        # Determine which aspect to target (prioritize Price if both are present, or random)
        target_aspect = "price" if row['price'] == 'positive' else "packing"
        
        prompt = f"Rewrite the following review to express a NEUTRAL sentiment regarding {target_aspect}. Keep other aspects unchanged.\nReview: \"{text}\"\nNeutral Rewrite:"
        
        prompt = f"Rewrite the following review to express a NEUTRAL sentiment regarding {target_aspect}. Keep other aspects unchanged.\nReview: \"{text}\"\nNeutral Rewrite:"
        
        row_data = {
            "original_text": text,
            "target_aspect": target_aspect,
            "original_sentiment": "positive",
            "prompt_for_gpt": prompt
        }
        
        # Add all aspect labels from the original row so we don't lose them
        aspects = ["stayingpower", "texture", "smell", "price", "colour", "shipping", "packing"]
        for aspect in aspects:
            row_data[aspect] = row.get(aspect, "None") # Default to None if missing
            
        output_rows.append(row_data)

    # Save to CSV
    out_df = pd.DataFrame(output_rows)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Saved {len(out_df)} prompts to {output_file}")
    print("You can now open this CSV, copy the 'prompt_for_gpt' column, and paste it into ChatGPT!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default=".", help="Root directory of the project")
    parser.add_argument("--output", default="outputs/augmentation/neutral_candidates.csv", help="Where to save the candidate list")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to extract")
    args = parser.parse_args()
    
    extract_candidates(args.project_dir, args.output, args.samples)
