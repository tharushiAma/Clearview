import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shap
import numpy as np
import io

def plot_msr_resolution(row, aspects):
    """
    Creates a 'Tug-of-War' or 'Balance Scale' visualization to explain 
    why the model decided 'Mixed', 'Positive', or 'Negative'.
    
    This is designed for NON-TECHNICAL users.
    """
    # 1. Extract Scores
    pos_score = 0.0
    neg_score = 0.0
    
    pos_contributors = []
    neg_contributors = []
    
    for a in aspects:
        pred = row.get(f"{a}_pred") 
        conf = row.get(f"{a}_conf", 0.0)
        
        if pred == "positive":
            pos_score += conf
            pos_contributors.append(f"{a} ({conf:.2f})")
        elif pred == "negative":
            neg_score += conf
            neg_contributors.append(f"{a} ({conf:.2f})")
            
    resolution = row.get("msr_resolution", "Unknown")
    conflict_score = row.get("conflict_score", 0.0)
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Draw the "Rope"
    ax.plot([0, 10], [0, 0], color='gray', lw=2, zorder=1)
    
    # Center Point
    center = 5.0
    
    # Calculate Pull (Simple logic: if pos > neg, marker moves right)
    total_pull = pos_score + neg_score + 1e-9
    # Map balance to range [2, 8] (center is 5)
    balance_point = center + ((pos_score - neg_score) / total_pull) * 3
    
    # Draw Positive Team (Right)
    ax.scatter([8], [0], s=pos_score*500, c='blue', label='Positive Mass', zorder=2, alpha=0.7)
    for i, contrib in enumerate(pos_contributors):
        ax.text(8.2, 0.2 + (i*0.2), contrib, color='blue', fontsize=10)
        
    # Draw Negative Team (Left)
    ax.scatter([2], [0], s=neg_score*500, c='red', label='Negative Mass', zorder=2, alpha=0.7)
    for i, contrib in enumerate(neg_contributors):
        ax.text(1.8, 0.2 + (i*0.2), contrib, color='red', ha='right', fontsize=10)
        
    # Draw The Result Marker
    marker_color = 'purple' if resolution == "MIXED" else ('blue' if "Positive" in resolution else 'red')
    ax.scatter([balance_point], [0], s=100, c=marker_color, marker='^', zorder=3, edgecolors='black')
    ax.text(balance_point, -0.3, f"RESULT:\n{resolution}", ha='center', fontweight='bold', color=marker_color)
    
    # Annotations
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 2)
    ax.set_title(f"Resolution Logic: Conflict Score = {conflict_score:.2f} (Threshold > 0.25 is MIXED)", fontsize=12)
    ax.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_shap_text(shap_values, filename):
    """
    Wraps standard SHAP text plot but saves it as HTML.
    Red highlight = Pushes towards Negative
    Blue highlight = Pushes towards Positive
    """
    # Create the HTML visualization
    html = shap.plots.text(shap_values, display=False)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
        
def plot_lime_html(lime_exp, filename):
    """
    Saves LIME explanation as HTML.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(lime_exp.as_html())
