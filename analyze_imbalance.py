import pandas as pd
import json
import os

def analyze_imbalance():
    aspects = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
    results = {}
    
    splits = ['train', 'val', 'test']
    base_path = 'data/splits'
    
    for split in splits:
        path = os.path.join(base_path, f'{split}.parquet')
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
            
        df = pd.read_parquet(path)
        split_results = {}
        
        for aspect in aspects:
            if aspect in df.columns:
                counts = df[aspect].value_counts().to_dict()
                # Ensure all labels are strings for JSON
                counts = {str(k): int(v) for k, v in counts.items()}
                split_results[aspect] = counts
            else:
                split_results[aspect] = "NOT_FOUND"
        
        results[split] = split_results
    
    with open('class_counts.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Analysis complete. Results saved to class_counts.json")

if __name__ == "__main__":
    analyze_imbalance()
