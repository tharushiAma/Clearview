"""
Data Preprocessing and Stratified Splitting Script
For: Class balanced aspect base mixed sentiment resolution with XAI

This script:
1. Loads translated data
2. Handles missing values
3. Processes aspect sentiment labels
4. Creates stratified train/val/test splits (70%/15%/15%)
5. Analyzes and documents class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import json

# Configuration
DATA_PATH = 'data/full_data_en.csv'
OUTPUT_DIR = 'data'
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Aspect columns (based on the CSV structure)
ASPECT_COLUMNS = ['stayingpower', 'texture', 'smell', 'price', 'others', 'colour', 'shipping', 'packing']

def load_and_explore_data():
    """Load translated data and perform initial exploration"""
    print("Loading translated data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def process_aspect_labels(df):
    """Process aspect sentiment labels"""
    print("\nProcessing aspect sentiment labels...")
    
    # Replace empty strings with NaN for consistent handling
    for col in ASPECT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace('', np.nan)
    
    # Map sentiment labels to standardized format
    sentiment_mapping = {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        # Handle any variations if present
    }
    
    for col in ASPECT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(lambda x: sentiment_mapping.get(str(x).lower(), x) if pd.notna(x) else x)
    
    return df

def create_multi_label_target(df):
    """Create a composite target for stratification that considers all aspects"""
    print("\nCreating multi-label target for stratified sampling...")
    
    # Create a string representation of all aspect sentiments
    def combine_labels(row):
        labels = []
        for col in ASPECT_COLUMNS:
            if col in df.columns:
                val = row[col]
                if pd.notna(val) and val != '':
                    labels.append(f"{col}_{val}")
        return '|'.join(sorted(labels)) if labels else 'no_labels'
    
    df['_stratify_key'] = df.apply(combine_labels, axis=1)
    
    # Handle rare stratification keys (count < 2) which cause train_test_split to fail
    key_counts = df['_stratify_key'].value_counts()
    rare_keys = key_counts[key_counts < 2].index
    
    print(f"Found {len(rare_keys)} rare stratification keys with only 1 sample. Grouping them to 'other'...")
    df['_stratify_key'] = df['_stratify_key'].apply(lambda x: 'other_rare_combination' if x in rare_keys else x)
    
    return df

def perform_stratified_split(df):
    """Perform stratified train/val/test split"""
    print("\nPerforming stratified split...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT,
        stratify=df['_stratify_key'],
        random_state=RANDOM_SEED
    )
    
    # Second split: train vs val
    # Calculate validation size as percentage of train+val
    val_size_adjusted = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['_stratify_key'],
        random_state=RANDOM_SEED
    )
    
    # Remove the temporary stratify key
    train_df = train_df.drop('_stratify_key', axis=1)
    val_df = val_df.drop('_stratify_key', axis=1)
    test_df = test_df.drop('_stratify_key', axis=1)
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def analyze_class_distribution(df, dataset_name="Full"):
    """Analyze class distribution for each aspect"""
    print(f"\n{'='*60}")
    print(f"Class Distribution Analysis - {dataset_name} Dataset")
    print(f"{'='*60}")
    
    distribution_data = {}
    
    for aspect in ASPECT_COLUMNS:
        if aspect not in df.columns:
            continue
            
        # Count values including NaN
        value_counts = df[aspect].value_counts(dropna=False)
        total = len(df)
        
        print(f"\n{aspect.upper()}:")
        print("-" * 40)
        
        aspect_dist = {}
        for label, count in value_counts.items():
            percentage = (count / total) * 100
            print(f"  {str(label):15s}: {count:6d} ({percentage:5.2f}%)")
            aspect_dist[str(label)] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        distribution_data[aspect] = aspect_dist
    
    return distribution_data

def identify_imbalanced_classes(train_df, val_df, test_df, threshold=10.0):
    """Identify imbalanced classes across all datasets"""
    print(f"\n{'='*60}")
    print("IDENTIFYING IMBALANCED CLASSES")
    print(f"{'='*60}")
    print(f"Threshold: Classes with < {threshold}% representation are considered rare")
    
    imbalanced_info = {
        'threshold_percentage': threshold,
        'rare_classes': {},
        'recommendations': []
    }
    
    for aspect in ASPECT_COLUMNS:
        if aspect not in train_df.columns:
            continue
        
        # Analyze in training set (most important)
        value_counts = train_df[aspect].value_counts(dropna=True)
        total = len(train_df)
        
        rare_classes = []
        for label, count in value_counts.items():
            percentage = (count / total) * 100
            if percentage < threshold and label not in ['', np.nan]:
                rare_classes.append({
                    'label': str(label),
                    'count_train': int(count),
                    'percentage_train': float(percentage)
                })
        
        if rare_classes:
            imbalanced_info['rare_classes'][aspect] = rare_classes
            print(f"\n⚠️  {aspect.upper()} - Rare classes detected:")
            for rc in rare_classes:
                print(f"    - {rc['label']}: {rc['count_train']} ({rc['percentage_train']:.2f}%)")
    
    # Generate recommendations
    if imbalanced_info['rare_classes']:
        imbalanced_info['recommendations'] = [
            "Use class-balanced loss functions (e.g., Focal Loss, Weighted Cross-Entropy)",
            "Consider oversampling rare classes or undersampling majority classes",
            "Apply data augmentation techniques for minority classes",
            "Use ensemble methods to boost minority class performance",
            "Monitor per-class metrics in addition to overall metrics"
        ]
    
    return imbalanced_info

def save_splits(train_df, val_df, test_df):
    """Save the splits to CSV files"""
    print("\nSaving splits to CSV files...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, 'train.csv')
    val_path = os.path.join(OUTPUT_DIR, 'val.csv')
    test_path = os.path.join(OUTPUT_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Saved train set to: {train_path}")
    print(f"✓ Saved validation set to: {val_path}")
    print(f"✓ Saved test set to: {test_path}")

def create_class_imbalance_report(train_dist, val_dist, test_dist, imbalance_info):
    """Create markdown report for class imbalance"""
    
    report = f"""# Class Imbalance Analysis

**Project**: Class balanced aspect base mixed sentiment resolution with XAI  
**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This document analyzes the class distribution across all aspect-based sentiment labels in the cosmetic review dataset after stratified splitting.

## Dataset Split Distribution

- **Train**: {train_dist.get('_total', 'N/A')} samples (70%)
- **Validation**: {val_dist.get('_total', 'N/A')} samples (15%)
- **Test**: {test_dist.get('_total', 'N/A')} samples (15%)

## Aspect-wise Class Distribution

"""
    
    # Add distribution tables for each aspect
    for aspect in ASPECT_COLUMNS:
        if aspect in train_dist and aspect != '_total':
            report += f"\n### {aspect.upper()}\n\n"
            report += "| Class | Train Count (%) | Val Count (%) | Test Count (%) |\n"
            report += "|-------|----------------|---------------|----------------|\n"
            
            # Get all unique labels across all sets
            all_labels = set()
            for dist in [train_dist, val_dist, test_dist]:
                if aspect in dist:
                    all_labels.update(dist[aspect].keys())
            
            for label in sorted(all_labels):
                train_info = train_dist.get(aspect, {}).get(label, {'count': 0, 'percentage': 0.0})
                val_info = val_dist.get(aspect, {}).get(label, {'count': 0, 'percentage': 0.0})
                test_info = test_dist.get(aspect, {}).get(label, {'count': 0, 'percentage': 0.0})
                
                report += f"| {label} | {train_info['count']} ({train_info['percentage']:.2f}%) | "
                report += f"{val_info['count']} ({val_info['percentage']:.2f}%) | "
                report += f"{test_info['count']} ({test_info['percentage']:.2f}%) |\n"
    
    # Add imbalanced classes section
    report += "\n## Imbalanced Classes Identified\n\n"
    report += f"**Threshold**: Classes with < {imbalance_info['threshold_percentage']}% representation\n\n"
    
    if imbalance_info['rare_classes']:
        for aspect, rare_list in imbalance_info['rare_classes'].items():
            report += f"### {aspect.upper()}\n\n"
            for rare_class in rare_list:
                report += f"- **{rare_class['label']}**: {rare_class['count_train']} samples ({rare_class['percentage_train']:.2f}%)\n"
            report += "\n"
        
        report += "## Recommendations for Handling Class Imbalance\n\n"
        for i, rec in enumerate(imbalance_info['recommendations'], 1):
            report += f"{i}. {rec}\n"
    else:
        report += "*No severely imbalanced classes detected based on the threshold.*\n"
    
    report += "\n## Stratification Strategy\n\n"
    report += "The dataset was split using **stratified sampling** based on multi-label combinations to ensure:\n"
    report += "- Proportional representation of all aspect-sentiment combinations\n"
    report += "- Rare classes maintain the same percentage across train/val/test sets\n"
    report += "- Model evaluation is performed on representative test data\n"
    
    return report

def main():
    """Main execution function"""
    print("="*60)
    print("Data Preprocessing and Stratified Splitting")
    print("="*60)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    
    # Step 2: Process aspect labels
    df = process_aspect_labels(df)
    
    # Step 3: Create multi-label target for stratification
    df = create_multi_label_target(df)
    
    # Step 4: Perform stratified split
    train_df, val_df, test_df = perform_stratified_split(df)
    
    # Step 5: Analyze class distribution
    train_dist = analyze_class_distribution(train_df, "Training")
    val_dist = analyze_class_distribution(val_df, "Validation")
    test_dist = analyze_class_distribution(test_df, "Test")
    
    # Add total counts
    train_dist['_total'] = len(train_df)
    val_dist['_total'] = len(val_df)
    test_dist['_total'] = len(test_df)
    
    # Step 6: Identify imbalanced classes
    imbalance_info = identify_imbalanced_classes(train_df, val_df, test_df)
    
    # Step 7: Save splits
    save_splits(train_df, val_df, test_df)
    
    # Step 8: Create class imbalance report
    report = create_class_imbalance_report(train_dist, val_dist, test_dist, imbalance_info)
    
    report_path = 'class_imbalance.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ Saved class imbalance report to: {report_path}")
    
    # Save distribution data as JSON for future reference
    distribution_json = {
        'train': train_dist,
        'val': val_dist,
        'test': test_dist,
        'imbalance_info': imbalance_info
    }
    
    json_path = 'data/class_distribution.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(distribution_json, f, indent=2)
    print(f"✓ Saved distribution data to: {json_path}")
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
