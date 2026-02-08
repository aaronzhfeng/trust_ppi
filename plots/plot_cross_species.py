"""
Plot: Cross-Species Trust Signal Performance

Creates a clean visualization of trust signal performance across species.
For the focused proposal v2.

Usage:
    python plots/plot_cross_species.py
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Manually entered from multi-seed aggregated results
RESULTS = {
    'yeast': {
        'n': 1000,
        'baseline': 0.740,
        'deform_auroc': 0.799,
        'deform_improvement': 0.098,
        'confidence_improvement': -0.001,
    },
    'human': {
        'n': 500,
        'baseline': 0.860,
        'deform_auroc': 0.763,
        'deform_improvement': 0.050,
        'confidence_improvement': 0.000,
    },
    'ecoli': {
        'n': 500,
        'baseline': 0.668,
        'deform_auroc': 0.704,
        'deform_improvement': 0.084,
        'confidence_improvement': -0.016,
    },
}


def plot_auroc_comparison(output_path: Path):
    """Bar chart: AUROC across species."""
    species = ['Yeast\n(n=1000)', 'Human\n(n=500)', 'E. coli\n(n=500)']
    aurocs = [RESULTS['yeast']['deform_auroc'], 
              RESULTS['human']['deform_auroc'],
              RESULTS['ecoli']['deform_auroc']]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(species, aurocs, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Random (0.5)')
    ax.set_ylabel('AUROC (Error Detection)', fontsize=12)
    ax.set_title('Deformation Stability: Cross-Species Performance', fontsize=13, fontweight='bold')
    ax.set_ylim(0.4, 0.9)
    ax.legend(loc='lower right')
    
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', 
                ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_selective_improvement(output_path: Path):
    """Bar chart: Selective prediction improvement at 80% coverage."""
    species = ['Yeast', 'Human', 'E. coli']
    
    deform_imp = [RESULTS['yeast']['deform_improvement'] * 100,
                  RESULTS['human']['deform_improvement'] * 100,
                  RESULTS['ecoli']['deform_improvement'] * 100]
    conf_imp = [RESULTS['yeast']['confidence_improvement'] * 100,
                RESULTS['human']['confidence_improvement'] * 100,
                RESULTS['ecoli']['confidence_improvement'] * 100]
    
    x = np.arange(len(species))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    bars1 = ax.bar(x - width/2, deform_imp, width, label='Deformation Stability',
                   color='#2E86AB', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, conf_imp, width, label='Generic Confidence',
                   color='#E74C3C', edgecolor='black', linewidth=1)
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Accuracy Improvement (%)', fontsize=12)
    ax.set_title('Selective Prediction @ 80% Coverage', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(species, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(-3, 12)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, 
                f'+{height:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#2E86AB')
    for bar in bars2:
        height = bar.get_height()
        label = f'{height:.1f}%' if height >= 0 else f'{height:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2, height - 0.8 if height < 0 else height + 0.3,
                label, ha='center', fontsize=10, fontweight='bold', color='#E74C3C')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined(output_path: Path):
    """Combined figure with both plots."""
    species = ['Yeast\n(n=1000)', 'Human\n(n=500)', 'E. coli\n(n=500)']
    species_short = ['Yeast', 'Human', 'E. coli']
    
    aurocs = [RESULTS['yeast']['deform_auroc'], 
              RESULTS['human']['deform_auroc'],
              RESULTS['ecoli']['deform_auroc']]
    
    deform_imp = [RESULTS['yeast']['deform_improvement'] * 100,
                  RESULTS['human']['deform_improvement'] * 100,
                  RESULTS['ecoli']['deform_improvement'] * 100]
    conf_imp = [RESULTS['yeast']['confidence_improvement'] * 100,
                RESULTS['human']['confidence_improvement'] * 100,
                RESULTS['ecoli']['confidence_improvement'] * 100]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    # Left: AUROC
    ax1 = axes[0]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars1 = ax1.bar(species, aurocs, color=colors, edgecolor='black', linewidth=1.2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Random')
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title('(a) Error Detection', fontsize=13, fontweight='bold')
    ax1.set_ylim(0.4, 0.9)
    ax1.legend(loc='lower right', fontsize=9)
    
    for bar, val in zip(bars1, aurocs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', 
                ha='center', fontsize=11, fontweight='bold')
    
    # Right: Selective improvement
    ax2 = axes[1]
    x = np.arange(len(species_short))
    width = 0.35
    
    bars2a = ax2.bar(x - width/2, deform_imp, width, label='Deformation Stability',
                     color='#2E86AB', edgecolor='black', linewidth=1)
    bars2b = ax2.bar(x + width/2, conf_imp, width, label='Generic Confidence',
                     color='#E74C3C', edgecolor='black', linewidth=1)
    
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('Accuracy Improvement (%)', fontsize=12)
    ax2.set_title('(b) Selective Prediction @ 80% Coverage', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(species_short, fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(-3, 12)
    
    for bar in bars2a:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.3, 
                f'+{height:.1f}%', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
    for bar in bars2b:
        height = bar.get_height()
        label = f'{height:.1f}%'
        y_pos = height - 0.8 if height < 0 else height + 0.3
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                label, ha='center', fontsize=9, fontweight='bold', color='#E74C3C')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot cross-species results')
    parser.add_argument('--output-dir', type=str, default='plots/figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_auroc_comparison(output_dir / 'cross_species_auroc.png')
    plot_selective_improvement(output_dir / 'cross_species_selective.png')
    plot_combined(output_dir / 'cross_species_combined.png')
    
    print("\n=== Cross-Species Summary ===")
    for name, data in RESULTS.items():
        print(f"{name}: baseline={data['baseline']:.1%}, AUROC={data['deform_auroc']:.3f}, "
              f"deform_imp=+{data['deform_improvement']*100:.1f}%, "
              f"conf_imp={data['confidence_improvement']*100:.1f}%")


if __name__ == '__main__':
    main()
