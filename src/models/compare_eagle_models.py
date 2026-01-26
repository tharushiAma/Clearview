#!/usr/bin/env python3
"""
EAGLE V2 vs EAGLE_FINAL Comparison Script

This script compares the performance of EAGLE V2 against the original EAGLE_FINAL
model, highlighting improvements in critical areas:
- Price aspect detection (negative/neutral)
- Packing negative detection
- Neutral class performance across all aspects
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def parse_metrics_file(filepath: str) -> Dict[str, Dict]:
    """
    Parse a metrics.txt file and extract performance metrics.
    
    Args:
        filepath: Path to metrics file
    
    Returns:
        Dictionary of aspect -> metrics
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    aspects = {}
    current_aspect = None
    
    # Split by aspect sections
    sections = content.split('--------------------------------------------------')
    
    for section in sections:
        # Find aspect name
        aspect_match = re.search(r'Aspect:\s+(\w+)', section)
        if aspect_match:
            current_aspect = aspect_match.group(1)
            
            # Extract metrics for each class
            metrics = {}
            
            for sentiment in ['negative', 'neutral', 'positive']:
                pattern = rf'{sentiment}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
                match = re.search(pattern, section)
                
                if match:
                    metrics[sentiment] = {
                        'precision': float(match.group(1)),
                        'recall': float(match.group(2)),
                        'f1-score': float(match.group(3)),
                        'support': int(match.group(4))
                    }
            
            # Extract macro avg
            macro_pattern = r'macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
            macro_match = re.search(macro_pattern, section)
            
            if macro_match:
                metrics['macro avg'] = {
                    'precision': float(macro_match.group(1)),
                    'recall': float(macro_match.group(2)),
                    'f1-score': float(macro_match.group(3)),
                    'support': int(macro_match.group(4))
                }
            
            aspects[current_aspect] = metrics
    
    return aspects


def calculate_improvements(original: Dict, enhanced: Dict) -> Dict:
    """
    Calculate improvement metrics.
    
    Args:
        original: Original model metrics
        enhanced: Enhanced model metrics
    
    Returns:
        Dictionary of improvements
    """
    improvements = {}
    
    for aspect in original.keys():
        if aspect not in enhanced:
            continue
        
        improvements[aspect] = {}
        
        for sentiment in ['negative', 'neutral', 'positive', 'macro avg']:
            if sentiment not in original[aspect] or sentiment not in enhanced[aspect]:
                continue
            
            orig_f1 = original[aspect][sentiment]['f1-score']
            enh_f1 = enhanced[aspect][sentiment]['f1-score']
            
            delta = enh_f1 - orig_f1
            pct_change = (delta / orig_f1 * 100) if orig_f1 > 0 else float('inf')
            
            improvements[aspect][sentiment] = {
                'original_f1': orig_f1,
                'enhanced_f1': enh_f1,
                'delta': delta,
                'pct_change': pct_change
            }
    
    return improvements


def print_comparison_table(improvements: Dict, aspect: str, title: str):
    """Print a formatted comparison table for one aspect."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    if aspect not in improvements:
        print(f"No data for {aspect}")
        return
    
    print(f"\n{'Sentiment':<12} {'Original F1':>12} {'Enhanced F1':>12} {'Δ':>8} {'% Change':>10}")
    print("-" * 80)
    
    for sentiment in ['negative', 'neutral', 'positive', 'macro avg']:
        if sentiment not in improvements[aspect]:
            continue
        
        data = improvements[aspect][sentiment]
        
        # Color coding
        if data['delta'] > 0.05:
            symbol = "✓✓"
        elif data['delta'] > 0:
            symbol = "✓"
        elif data['delta'] < -0.05:
            symbol = "✗✗"
        elif data['delta'] < 0:
            symbol = "✗"
        else:
            symbol = "="
        
        print(f"{sentiment:<12} {data['original_f1']:>12.4f} {data['enhanced_f1']:>12.4f} "
              f"{data['delta']:>8.4f} {data['pct_change']:>9.1f}% {symbol}")


def create_comparison_visualization(improvements: Dict, output_dir: str):
    """
    Create visualizations comparing models.
    
    Args:
        improvements: Improvement metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price aspect detailed comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if 'price' in improvements:
        sentiments = ['negative', 'neutral', 'positive']
        orig_scores = [improvements['price'][s]['original_f1'] for s in sentiments]
        enh_scores = [improvements['price'][s]['enhanced_f1'] for s in sentiments]
        
        x = np.arange(len(sentiments))
        width = 0.35
        
        ax.bar(x - width/2, orig_scores, width, label='EAGLE_FINAL', color='#ff7f0e')
        ax.bar(x + width/2, enh_scores, width, label='EAGLE V2', color='#2ca02c')
        
        ax.set_ylabel('F1 Score')
        ax.set_title('Price Aspect: EAGLE_FINAL vs EAGLE V2', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sentiments)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'price_aspect_comparison.png'), dpi=300)
        print(f"Saved: {output_dir}/price_aspect_comparison.png")
    
    # 2. Neutral class comparison across all aspects
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    aspects = list(improvements.keys())
    orig_neutral = [improvements[a].get('neutral', {}).get('original_f1', 0) for a in aspects]
    enh_neutral = [improvements[a].get('neutral', {}).get('enhanced_f1', 0) for a in aspects]
    
    x = np.arange(len(aspects))
    width = 0.35
    
    ax.bar(x - width/2, orig_neutral, width, label='EAGLE_FINAL', color='#ff7f0e')
    ax.bar(x + width/2, enh_neutral, width, label='EAGLE V2', color='#2ca02c')
    
    ax.set_ylabel('F1 Score (Neutral Class)')
    ax.set_title('Neutral Class Performance: All Aspects', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neutral_class_comparison.png'), dpi=300)
    print(f"Saved: {output_dir}/neutral_class_comparison.png")
    
    # 3. Heatmap of improvements
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    sentiments = ['negative', 'neutral', 'positive', 'macro avg']
    
    # Create matrix of deltas
    delta_matrix = []
    for aspect in aspects:
        row = [improvements[aspect].get(s, {}).get('delta', 0) for s in sentiments]
        delta_matrix.append(row)
    
    delta_matrix = np.array(delta_matrix)
    
    sns.heatmap(delta_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=sentiments, yticklabels=aspects,
                cbar_kws={'label': 'F1 Score Improvement'}, ax=ax)
    
    ax.set_title('F1 Score Improvements (EAGLE V2 - EAGLE_FINAL)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_heatmap.png'), dpi=300)
    print(f"Saved: {output_dir}/improvement_heatmap.png")
    
    plt.close('all')


def generate_summary_report(improvements: Dict, output_path: str):
    """Generate a summary report as markdown."""
    
    report = []
    report.append("# EAGLE V2 vs EAGLE_FINAL - Performance Comparison Report\n")
    report.append(f"**Generated**: {pd.Timestamp.now()}\n\n")
    
    report.append("---\n\n")
    
    # Overall summary
    report.append("## 📊 Overall Summary\n\n")
    
    total_improvements = 0
    total_degradations = 0
    significant_improvements = []
    
    for aspect, metrics in improvements.items():
        macro_delta = metrics.get('macro avg', {}).get('delta', 0)
        
        if macro_delta > 0.05:
            significant_improvements.append((aspect, macro_delta))
            total_improvements += 1
        elif macro_delta < -0.05:
            total_degradations += 1
    
    report.append(f"- **Aspects Improved**: {total_improvements}\n")
    report.append(f"- **Aspects Degraded**: {total_degradations}\n")
    report.append(f"- **Aspects Stable**: {len(improvements) - total_improvements - total_degradations}\n\n")
    
    if significant_improvements:
        report.append("### 🎯 Significant Improvements (Δ > 0.05)\n\n")
        for aspect, delta in sorted(significant_improvements, key=lambda x: x[1], reverse=True):
            report.append(f"- **{aspect.upper()}**: +{delta:.4f}\n")
        report.append("\n")
    
    # Critical aspects
    report.append("---\n\n")
    report.append("## 🔥 Critical Aspects Performance\n\n")
    
    critical_aspects = ['price', 'packing']
    
    for aspect in critical_aspects:
        if aspect not in improvements:
            continue
        
        report.append(f"### {aspect.upper()}\n\n")
        report.append("| Sentiment | EAGLE_FINAL | EAGLE V2 | Δ F1 | % Change |\n")
        report.append("|-----------|-------------|----------|------|----------|\n")
        
        for sentiment in ['negative', 'neutral', 'positive', 'macro avg']:
            if sentiment not in improvements[aspect]:
                continue
            
            data = improvements[aspect][sentiment]
            report.append(f"| {sentiment.capitalize():<9} | "
                         f"{data['original_f1']:.4f} | "
                         f"{data['enhanced_f1']:.4f} | "
                         f"{data['delta']:+.4f} | "
                         f"{data['pct_change']:+.1f}% |\n")
        
        report.append("\n")
    
    # Neutral class analysis
    report.append("---\n\n")
    report.append("## 🎭 Neutral Class Analysis\n\n")
    report.append("| Aspect | EAGLE_FINAL | EAGLE V2 | Δ F1 | % Change |\n")
    report.append("|--------|-------------|----------|------|----------|\n")
    
    neutral_deltas = []
    
    for aspect in improvements.keys():
        if 'neutral' not in improvements[aspect]:
            continue
        
        data = improvements[aspect]['neutral']
        neutral_deltas.append(data['delta'])
        
        report.append(f"| {aspect.capitalize():<6} | "
                     f"{data['original_f1']:.4f} | "
                     f"{data['enhanced_f1']:.4f} | "
                     f"{data['delta']:+.4f} | "
                     f"{data['pct_change']:+.1f}% |\n")
    
    avg_neutral_improvement = np.mean(neutral_deltas) if neutral_deltas else 0
    report.append(f"\n**Average Neutral Improvement**: {avg_neutral_improvement:+.4f}\n\n")
    
    # Write report
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    print(f"\nSummary report saved to: {output_path}")


def main():
    """Main comparison function."""
    
    # Paths
    project_dir = Path("c:/Users/lucif/Desktop/Clearview")
    reports_dir = project_dir / "outputs" / "reports"
    
    eagle_final_metrics = reports_dir / "eagle_final_metrics.txt"
    
    # Find the latest EAGLE V2 metrics file
    eagle_v2_files = list(reports_dir.glob("eagle_v2_*_metrics.txt"))
    
    if not eagle_v2_files:
        print("❌ No EAGLE V2 metrics files found!")
        print("Please train EAGLE V2 first: python src/models/train_eagle_v2.py")
        return
    
    # Use the latest file
    eagle_v2_metrics = sorted(eagle_v2_files)[-1]
    
    print(f"\n{'='*80}")
    print("EAGLE V2 vs EAGLE_FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"\nOriginal Model: {eagle_final_metrics.name}")
    print(f"Enhanced Model: {eagle_v2_metrics.name}")
    
    # Parse metrics
    print("\nParsing metrics files...")
    original_metrics = parse_metrics_file(str(eagle_final_metrics))
    enhanced_metrics = parse_metrics_file(str(eagle_v2_metrics))
    
    # Calculate improvements
    print("Calculating improvements...")
    improvements = calculate_improvements(original_metrics, enhanced_metrics)
    
    # Print comparisons
    print_comparison_table(improvements, 'price', '💰 PRICE ASPECT COMPARISON')
    print_comparison_table(improvements, 'packing', '📦 PACKING ASPECT COMPARISON')
    
    # Print neutral class summary
    print(f"\n{'='*80}")
    print("🎭 NEUTRAL CLASS AVERAGE IMPROVEMENT")
    print(f"{'='*80}\n")
    
    neutral_improvements = []
    for aspect, metrics in improvements.items():
        if 'neutral' in metrics:
            neutral_improvements.append((aspect, metrics['neutral']['delta']))
    
    for aspect, delta in sorted(neutral_improvements, key=lambda x: x[1], reverse=True):
        symbol = "✓✓" if delta > 0.05 else "✓" if delta > 0 else "✗"
        print(f"{aspect:<15} {delta:>8.4f} {symbol}")
    
    avg_neutral_delta = np.mean([d for _, d in neutral_improvements])
    print(f"\n{'Average':<15} {avg_neutral_delta:>8.4f}")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    viz_dir = reports_dir / "comparisons"
    create_comparison_visualization(improvements, str(viz_dir))
    
    # Generate summary report
    report_path = viz_dir / "eagle_v2_comparison_report.md"
    generate_summary_report(improvements, str(report_path))
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {viz_dir}")
    print(f"- Visualizations: *.png")
    print(f"- Summary report: {report_path.name}")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
