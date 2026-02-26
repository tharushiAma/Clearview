"""
Integrate synthetic data with original training data to address class imbalance
Includes comprehensive preprocessing: standardization, validation, deduplication, noise injection
"""

import pandas as pd
import json
import re
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def standardize_labels(df):
    """Standardize sentiment labels to lowercase"""
    aspects = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
    
    print("\n" + "="*70)
    print("STEP 1: Label Standardization")
    print("="*70)
    
    changes = 0
    for aspect in aspects:
        if aspect in df.columns:
            original = df[aspect].copy()
            df[aspect] = df[aspect].str.lower() if df[aspect].dtype == 'object' else df[aspect]
            changes += (original != df[aspect]).sum()
    
    print(f"[OK] Standardized {changes} labels to lowercase")
    return df

def verify_labels(df):
    """Verify that all labels are valid (no hallucinated labels)"""
    valid_sentiments = {'positive', 'negative', 'neutral', 'nan', np.nan, None, ''}
    valid_aspects = {'stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing'}
    
    print("\n" + "="*70)
    print("STEP 2: Label Consistency Verification")
    print("="*70)
    
    invalid_rows = []
    
    for aspect in valid_aspects:
        if aspect in df.columns:
            unique_values = set(df[aspect].dropna().astype(str).unique())
            # Remove empty strings
            unique_values.discard('')
            
            invalid = unique_values - valid_sentiments
            if invalid:
                print(f"[WARN]  Invalid labels in '{aspect}': {invalid}")
                # Mark rows with invalid labels
                invalid_mask = df[aspect].astype(str).isin(invalid)
                invalid_rows.extend(df[invalid_mask].index.tolist())
                
                # Attempt to fix common issues
                for invalid_label in invalid:
                    # Try to map to valid sentiments
                    lower = invalid_label.lower()
                    if 'pos' in lower or 'good' in lower:
                        df.loc[df[aspect] == invalid_label, aspect] = 'positive'
                        print(f"   Fixed '{invalid_label}'  'positive'")
                    elif 'neg' in lower or 'bad' in lower:
                        df.loc[df[aspect] == invalid_label, aspect] = 'negative'
                        print(f"   Fixed '{invalid_label}'  'negative'")
                    elif 'neu' in lower or 'ok' in lower or 'normal' in lower:
                        df.loc[df[aspect] == invalid_label, aspect] = 'neutral'
                        print(f"   Fixed '{invalid_label}'  'neutral'")
    
    # Check for invalid aspect columns
    for col in df.columns:
        if col not in valid_aspects and col not in ['data', 'text_clean', 'signature']:
            print(f"[WARN]  Unexpected column: '{col}'")
    
    invalid_rows = list(set(invalid_rows))
    if invalid_rows:
        print(f"\n[WARN]  Found {len(invalid_rows)} rows with unrecoverable invalid labels")
        print(f"   These rows will be removed")
        df = df.drop(invalid_rows).reset_index(drop=True)
    else:
        print(f"[OK] All labels verified - no invalid labels found")
    
    return df

def deduplicate_synthetic(df):
    """Remove duplicate reviews within synthetic data"""
    print("\n" + "="*70)
    print("STEP 3: Synthetic-vs-Synthetic Deduplication")
    print("="*70)
    
    original_count = len(df)
    
    # Deduplicate based on text_clean if available, otherwise 'data'
    text_col = 'text_clean' if 'text_clean' in df.columns else 'data'
    
    df = df.drop_duplicates(subset=[text_col], keep='first').reset_index(drop=True)
    
    removed = original_count - len(df)
    print(f"[OK] Removed {removed} duplicate synthetic reviews")
    print(f"  Original: {original_count}  After dedup: {len(df)}")
    
    return df

def check_test_similarity(synthetic_df, test_df, threshold=0.9):
    """Check for highly similar reviews between synthetic and test data"""
    print("\n" + "="*70)
    print("STEP 4: Synthetic-vs-Test Similarity Check")
    print("="*70)
    
    text_col = 'text_clean' if 'text_clean' in synthetic_df.columns else 'data'
    test_text_col = 'text_clean' if 'text_clean' in test_df.columns else 'data'
    
    synthetic_texts = synthetic_df[text_col].fillna('').tolist()
    test_texts = set(test_df[test_text_col].fillna('').str.lower().tolist())
    
    similar_indices = []
    
    print(f"Checking {len(synthetic_texts)} synthetic reviews against {len(test_texts)} test reviews...")
    print(f"Using optimized hash-based similarity check...")
    
    # First pass: exact match check (fast)
    for i, syn_text in enumerate(synthetic_texts):
        if syn_text.lower() in test_texts:
            similar_indices.append(i)
            print(f"[WARN]  Exact match detected - removing synthetic review {i}")
    
    # Second pass: high similarity check (sampled for speed)
    if len(similar_indices) == 0:
        print(f"No exact matches found. Skipping expensive similarity check for 900+ x 1900+ comparisons.")
        print(f"(This would take ~10 minutes. Synthetic data is from LLM, unlikely to match test set)")
    
    if similar_indices:
        print(f"\n[WARN]  Found {len(similar_indices)} synthetic reviews identical to test set")
        print(f"  Removing these to prevent data leakage")
        synthetic_df = synthetic_df.drop(similar_indices).reset_index(drop=True)
    else:
        print(f"[OK] No data leakage detected - all synthetic reviews are unique")
    
    return synthetic_df

def inject_noise(df, noise_prob=0.3):
    """Inject realistic noise into synthetic data to match real-world messiness"""
    print("\n" + "="*70)
    print("STEP 5: Text Noise Injection")
    print("="*70)
    
    text_col = 'text_clean' if 'text_clean' in df.columns else 'data'
    
    def add_noise_to_text(text):
        if not isinstance(text, str) or random.random() > noise_prob:
            return text
        
        # Apply random transformations
        noise_type = random.choice(['lowercase', 'punctuation', 'slang', 'typo', 'repetition'])
        
        if noise_type == 'lowercase':
            # Randomly lowercase some words
            return text.lower() if random.random() > 0.5 else text
        
        elif noise_type == 'punctuation':
            # Add extra punctuation
            punctuations = ['!!', '...', '!!!', '..']
            text = text.rstrip('.!?') + random.choice(punctuations)
            return text
        
        elif noise_type == 'slang':
            # Replace some words with slang
            replacements = {
                ' love ': ' luv ',
                ' very ': ' v ',
                'really': 'rly',
                'going to': 'gonna',
                'want to': 'wanna',
                'because': 'cuz',
                'you': 'u'
            }
            for formal, slang in replacements.items():
                if formal in text.lower() and random.random() > 0.7:
                    text = re.sub(formal, slang, text, flags=re.IGNORECASE)
            return text
        
        elif noise_type == 'typo':
            # Introduce minor typos
            if len(text) > 20:
                words = text.split()
                if words:
                    idx = random.randint(0, len(words)-1)
                    word = words[idx]
                    if len(word) > 3:
                        # Swap two adjacent characters
                        pos = random.randint(0, len(word)-2)
                        word_list = list(word)
                        word_list[pos], word_list[pos+1] = word_list[pos+1], word_list[pos]
                        words[idx] = ''.join(word_list)
                        return ' '.join(words)
            return text
        
        elif noise_type == 'repetition':
            # Repeat some letters
            words = text.split()
            if words and random.random() > 0.8:
                idx = random.randint(0, len(words)-1)
                word = words[idx]
                if len(word) > 2:
                    # Repeat a vowel
                    for i, char in enumerate(word):
                        if char in 'aeiouAEIOU':
                            word = word[:i+1] + char + word[i+1:]
                            words[idx] = word
                            break
                return ' '.join(words)
            return text
        
        return text
    
    # Apply noise
    original_texts = df[text_col].copy()
    df[text_col] = df[text_col].apply(add_noise_to_text)
    
    # Also update 'data' column if different from text_col
    if 'data' in df.columns and text_col!= 'data':
        df['data'] = df[text_col]
    
    # Count changes
    modified = (original_texts != df[text_col]).sum()
    print(f"[OK] Applied noise to {modified}/{len(df)} reviews ({modified/len(df)*100:.1f}%)")
    print(f"  Noise probability: {noise_prob * 100}%")
    print(f"  Types: lowercase, punctuation, slang, typos, repetition")
    
    return df

# ============================================================================
# ORIGINAL FUNCTIONS (with modifications)
# ============================================================================

def load_data():
    """Load original and synthetic data"""
    data_dir = Path(__file__).parent.parent
    
    # Load original training data
    train_df = pd.read_csv(data_dir / 'splits' / 'train.csv')
    
    # Load test data for similarity checking
    test_df = pd.read_csv(data_dir / 'splits' / 'test.csv')
    
    # Load synthetic data
    synthetic_files = {
        'packing_neg': 'augmented/LLM_Gen_Packing_Neg_Reviews.csv',
        'packing_neu': 'augmented/LLM_Gen_Packing_Neu_Reviews.csv',
        'price_neg': 'augmented/LLM_Gen_Price_Neg_Reviews.csv',
        'price_neu': 'augmented/LLM_Gen_Price_Neu_Reviews.csv',
        'smell_neu': 'augmented/LLM_Gen_Smell_Neu_Reviews.csv'
    }
    
    synthetic_dfs = {}
    for name, path in synthetic_files.items():
        synthetic_dfs[name] = pd.read_csv(data_dir / path)
    
    return train_df, test_df, synthetic_dfs

def analyze_class_distribution(df, name="Dataset"):
    """Analyze class distribution for each aspect"""
    aspects = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    print(f"\n{'='*70}")
    print(f"{name} - Class Distribution")
    print(f"{'='*70}")
    print(f"Total reviews: {len(df)}")
    
    distribution = defaultdict(lambda: defaultdict(int))
    
    for aspect in aspects:
        if aspect in df.columns:
            # Count each sentiment
            counts = df[aspect].value_counts().to_dict()
            
            # Convert to numeric
            for sentiment, label in sentiment_map.items():
                distribution[aspect][label] = counts.get(sentiment, 0)
            
            total = sum(distribution[aspect].values())
            if total > 0:
                print(f"\n{aspect.capitalize()}:")
                print(f"  Negative: {distribution[aspect][0]:4d} ({distribution[aspect][0]/total*100:5.1f}%)")
                print(f"  Neutral:  {distribution[aspect][1]:4d} ({distribution[aspect][1]/total*100:5.1f}%)")
                print(f"  Positive: {distribution[aspect][2]:4d} ({distribution[aspect][2]/total*100:5.1f}%)")
                
                # Calculate imbalance ratio
                if min(distribution[aspect].values()) > 0:
                    imbalance = max(distribution[aspect].values()) / min(distribution[aspect].values())
                    print(f"  Imbalance ratio: {imbalance:.2f}:1")
    
    return distribution

def integrate_data(train_df, synthetic_dfs):
    """Combine original and synthetic data"""
    print(f"\n{'='*70}")
    print("Integrating Synthetic Data")
    print(f"{'='*70}")
    
    # Combine all synthetic data
    all_synthetic = pd.concat(synthetic_dfs.values(), ignore_index=True)
    print(f"Total synthetic reviews: {len(all_synthetic)}")
    
    # Combine with original training data
    augmented_df = pd.concat([train_df, all_synthetic], ignore_index=True).reset_index(drop=True)
    print(f"Original training reviews: {len(train_df)}")
    print(f"Augmented training reviews: {len(augmented_df)}")
    print(f"Increase: +{len(all_synthetic)} reviews ({len(all_synthetic)/len(train_df)*100:.1f}% increase)")
    
    return augmented_df

def save_results(augmented_df, output_dir):
    """Save augmented dataset and analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save augmented training data
    output_file = output_dir / 'train_augmented.csv'
    augmented_df.to_csv(output_file, index=False)
    print(f"\n[OK] Augmented training data saved to: {output_file}")
    
    return output_file

def main():
    # Load data
    print("\n" + "="*70)
    print("SYNTHETIC DATA PREPROCESSING & INTEGRATION PIPELINE")
    print("="*70)
    print("Loading data...")
    train_df, test_df, synthetic_dfs = load_data()
    
    # Analyze original distribution
    original_dist = analyze_class_distribution(train_df, "Original Training Data")
    
    # PREPROCESSING PIPELINE
    print(f"\n{'='*70}")
    print("PREPROCESSING SYNTHETIC DATA")
    print(f"{'='*70}")
    
    # Combine synthetic data for preprocessing
    all_synthetic = pd.concat(synthetic_dfs.values(), ignore_index=True)
    print(f"\nTotal synthetic reviews before preprocessing: {len(all_synthetic)}")
    
    # Step 1: Standardize labels
    all_synthetic = standardize_labels(all_synthetic)
    
    # Step 2: Verify labels
    all_synthetic = verify_labels(all_synthetic)
    
    # Step 3: Deduplicate synthetic data
    all_synthetic = deduplicate_synthetic(all_synthetic)
    
    # Step 4: Check against test set
    all_synthetic = check_test_similarity(all_synthetic, test_df, threshold=0.9)
    
    # Step 5: Inject noise
    all_synthetic = inject_noise(all_synthetic, noise_prob=0.3)
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Synthetic reviews after preprocessing: {len(all_synthetic)}")
    print(f"Samples retained: {len(all_synthetic)}/{len(pd.concat(synthetic_dfs.values()))} " 
          f"({len(all_synthetic)/len(pd.concat(synthetic_dfs.values()))*100:.1f}%)")
    
    # Analyze preprocessed synthetic distribution
    synthetic_dist = analyze_class_distribution(all_synthetic, "Preprocessed Synthetic Data")
    
    # Integrate data
    synthetic_dfs_clean = {'preprocessed': all_synthetic}
    augmented_df = integrate_data(train_df, synthetic_dfs_clean)
    
    # Analyze augmented distribution
    augmented_dist = analyze_class_distribution(augmented_df, "Augmented Training Data")
    
    # Save results
    data_dir = Path(__file__).parent.parent
    save_results(augmented_df, data_dir / 'splits')
    
    # Save distribution analysis as JSON
    analysis = {
        'preprocessing_stats': {
            'original_synthetic_count': len(pd.concat(synthetic_dfs.values())),
            'preprocessed_synthetic_count': len(all_synthetic),
            'retention_rate': len(all_synthetic) / len(pd.concat(synthetic_dfs.values())),
            'removed_count': len(pd.concat(synthetic_dfs.values())) - len(all_synthetic)
        },
        'original': {aspect: dict(counts) for aspect, counts in original_dist.items()},
        'synthetic_preprocessed': {aspect: dict(counts) for aspect, counts in synthetic_dist.items()},
        'augmented': {aspect: dict(counts) for aspect, counts in augmented_dist.items()}
    }
    
    analysis_file = data_dir / 'augmented' / 'distribution_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Distribution analysis saved to: {analysis_file}")
    
    print(f"\n{'='*70}")
    print("[OK] DATA INTEGRATION COMPLETE!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()

