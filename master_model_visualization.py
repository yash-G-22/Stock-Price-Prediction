"""
Master Model Comparison Visualization
=====================================
Creates comprehensive comparison visualizations for all viable models (R² >= -2).

This script:
1. Loads cleaned model results from master_results_comparison.csv
2. Generates publication-ready visualization dashboard
3. Creates detailed ranking chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def load_results():
    """Load cleaned model results."""
    df = pd.read_csv('master_results_comparison.csv')
    df = df.sort_values('R2', ascending=False).reset_index(drop=True)
    print(f"✓ Loaded {len(df)} models from master_results_comparison.csv")
    return df


def get_performance_tier(r2):
    """Assign performance tier based on R² score."""
    if r2 >= 0.9:
        return 'Excellent', '#2ecc71'  # Green
    elif r2 >= 0.5:
        return 'Good', '#3498db'  # Blue
    elif r2 >= 0.0:
        return 'Moderate', '#f39c12'  # Orange
    else:
        return 'Poor', '#e74c3c'  # Red


def create_master_visualization(df):
    """Create comprehensive multi-panel visualization dashboard."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Title
    fig.suptitle('Master Model Comparison Dashboard\nStock Price Prediction - Viable Models Only (R² ≥ -2)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Assign colors based on performance tier
    colors = [get_performance_tier(r2)[1] for r2 in df['R2']]
    tiers = [get_performance_tier(r2)[0] for r2 in df['R2']]
    df = df.copy()
    df['Tier'] = tiers
    df['Color'] = colors
    
    # ==================== Panel 1: R² Score Comparison ====================
    ax1 = fig.add_subplot(2, 2, 1)
    
    bars = ax1.barh(range(len(df)), df['R2'], 
                    color=df['Color'], edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['Model'], fontsize=10)
    ax1.set_xlabel('R² Score', fontweight='bold')
    ax1.set_title('R² Score by Model (Higher is Better)', fontweight='bold', fontsize=12)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Excellent (0.9)')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['R2'])):
        ax1.text(val + 0.02 if val >= 0 else val - 0.15, 
                 bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9, fontweight='bold')
    
    # ==================== Panel 2: RMSE Comparison ====================
    ax2 = fig.add_subplot(2, 2, 2)
    
    bars2 = ax2.barh(range(len(df)), df['RMSE'], 
                     color=df['Color'], edgecolor='white', linewidth=0.5)
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels(df['Model'], fontsize=10)
    ax2.set_xlabel('RMSE (Lower is Better)', fontweight='bold')
    ax2.set_title('RMSE by Model', fontweight='bold', fontsize=12)
    ax2.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars2, df['RMSE']):
        ax2.text(val + 10, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', ha='left', fontsize=9)
    
    # ==================== Panel 3: Performance Tier Distribution ====================
    ax3 = fig.add_subplot(2, 2, 3)
    
    tier_counts = df['Tier'].value_counts()
    tier_colors = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Moderate': '#f39c12', 'Poor': '#e74c3c'}
    colors_pie = [tier_colors.get(t, '#95a5a6') for t in tier_counts.index]
    
    wedges, texts, autotexts = ax3.pie(tier_counts, labels=tier_counts.index, 
                                        autopct='%1.0f%%', colors=colors_pie,
                                        explode=[0.05]*len(tier_counts),
                                        shadow=True, startangle=90)
    ax3.set_title('Model Performance Distribution', fontweight='bold', fontsize=12)
    
    # ==================== Panel 4: R² vs RMSE Scatter ====================
    ax4 = fig.add_subplot(2, 2, 4)
    
    scatter = ax4.scatter(df['RMSE'], df['R2'], 
                          c=df['Color'], s=150, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Annotate points
    for _, row in df.iterrows():
        ax4.annotate(row['Model'], (row['RMSE'], row['R2']), 
                     fontsize=8, alpha=0.9, ha='left', va='bottom',
                     xytext=(5, 5), textcoords='offset points')
    
    ax4.set_xlabel('RMSE (Lower is Better)', fontweight='bold')
    ax4.set_ylabel('R² Score (Higher is Better)', fontweight='bold')
    ax4.set_title('Model Performance: R² vs RMSE Trade-off', fontweight='bold', fontsize=12)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='R²=0.9 (Excellent)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def create_ranking_chart(df):
    """Create a detailed ranking chart for all models."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Assign colors based on performance
    colors = [get_performance_tier(r2)[1] for r2 in df['R2']]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['R2'], color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Model'], fontsize=11)
    ax.invert_yaxis()
    
    ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Ranking by R² Score\n(Only Viable Models - R² ≥ -2)', 
                 fontsize=14, fontweight='bold')
    
    # Add vertical lines for thresholds
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (R²≥0.9)')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1, label='Baseline (R²=0)')
    
    # Add value labels on bars
    for bar, val in zip(bars, df['R2']):
        width = bar.get_width()
        label_x = width + 0.02 if width >= 0 else width - 0.15
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', ha=ha, fontsize=10, fontweight='bold')
    
    # Legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Set x-axis limits
    ax.set_xlim(-2.5, 1.1)
    
    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("=" * 60)
    print("MASTER MODEL COMPARISON VISUALIZATION")
    print("Showing only viable models (R² ≥ -2)")
    print("=" * 60)
    
    # Load results
    df = load_results()
    
    print(f"\n📊 Total viable models: {len(df)}")
    
    # Display summary
    print("\n🏆 MODEL RANKINGS:")
    print("-" * 50)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        tier, _ = get_performance_tier(row['R2'])
        print(f"  {i}. {row['Model']}: R²={row['R2']:.4f} ({tier})")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    fig1 = create_master_visualization(df)
    fig1.savefig('master_model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  ✓ Saved: master_model_comparison.png")
    
    fig2 = create_ranking_chart(df)
    fig2.savefig('model_ranking_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  ✓ Saved: model_ranking_chart.png")
    
    plt.close('all')
    
    print("\n" + "=" * 60)
    print("✅ Visualization complete!")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    results_df = main()
