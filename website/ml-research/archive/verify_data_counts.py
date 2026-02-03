import pandas as pd
import numpy as np
from src.data_layer._common import ASPECTS, LABEL_MAP

def verify_counts(path):
    print(f"\n--- Verifying: {path} ---")
    df = pd.read_parquet(path)
    
    for asp in ASPECTS:
        if asp not in df.columns:
            print(f"Warning: {asp} missing from columns!")
            continue
            
        # Raw counts including NaNs
        counts = df[asp].value_counts(dropna=False).to_dict()
        
        # Simulated mapping to check NULL representation
        mapped_counts = {0: 0, 1: 0, 2: 0, 3: 0, -100: 0}
        for val, count in counts.items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                mapped_counts[3] += count
            elif val in LABEL_MAP:
                mapped_counts[LABEL_MAP[val]] += count
            else:
                s_val = str(val).lower()
                if s_val in ["nan", "none", "null", ""]:
                    mapped_counts[3] += count
                else:
                    mapped_counts[-100] += count
        
        print(f"Aspect '{asp}': {mapped_counts}")

if __name__ == "__main__":
    verify_counts("data/splits/train.parquet")
    verify_counts("data/splits/val.parquet")
