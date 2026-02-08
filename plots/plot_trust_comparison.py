"""
Plot: Trust Signals Comparison (AUROC + Selective Prediction)

Creates a side-by-side comparison of:
1. AUROC for error detection across different trust signals
2. Selective prediction improvement at 80% coverage

Usage:
    python plots/plot_trust_comparison.py
    python plots/plot_trust_comparison.py --output plots/figures/trust_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path):
    """Load yeast and human results."""
    yeast_files = list(results_dir.glob('protein_trust_eval_yeast_seed42_*.json'))
    human_files = list(results_dir.glob('protein_trust_eval_human_seed42_*.json'))
    
    if not yeast_files or not human_files:
        raise FileNotFoundError(f"Results not found in {results_dir}")
    
    with open(yeast_files[0]) as f:
        yeast = json.load(f)
    with open(human_files[0]) as f:
        human = json.load(f)
    
    return yeast, human


def plot_trust_comparison(yeast: dict, human: dict, output_path: Path):
    """Create trust signals comparison plot."""
    
    # Signal configuration
    signals = ['confidence', 't_deform_combined', 't_deform_flip', 't_interface_combined']
    labels = ['Generic\nConfidence', 'Deformation\nCombined', 'Deformation\nFlip', 'Interface\nCombined']
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: AUROC comparison (Yeast)
    aurocs_yeast = [yeast['error_detection'][s]['auroc'] for s in signals]
    
    ax1 = axes[0]
    bars1 = ax1.bar(labels, aurocs_yeast, color=colors, edgecolor='black', linewidth=1.2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random (0.5)')
    ax1.set_ylabel('AUROC (Error Detection)', fontsize=12)
    ax1.set_title('Yeast Dataset: Error Detection Performance', fontsize=13, fontweight='bold')
    ax1.set_ylim(0.4, 0.9)
    
    for bar, val in zip(bars1, aurocs_yeast):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=10)
    
    # Plot 2: Selective Prediction Improvement
    improvements_yeast = [yeast['selective_prediction'][s]['improvement'] * 100 for s in signals]
    improvements_human = [human['selective_prediction'][s]['improvement'] * 100 for s in signals]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax2 = axes[1]
    bars2a = ax2.bar(x - width/2, improvements_yeast, width, label='Yeast', color='#1f77b4', edgecolor='black')
    bars2b = ax2.bar(x + width/2, improvements_human, width, label='Human', color='#ff7f0e', edgecolor='black')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Accuracy Improvement (%)', fontsize=12)
    ax2.set_title('Selective Prediction @ 80% Coverage', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_ylim(-2, 12)
    
    # Add value labels
    for bar in bars2a:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{height:.1f}%', ha='center', fontsize=9)
    for bar in bars2b:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{height:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot trust signals comparison')
    parser.add_argument('--results-dir', type=str, 
                        default='experiments/tier3_full/results',
                        help='Directory containing result JSON files')
    parser.add_argument('--output', type=str,
                        default='plots/figures/trust_comparison.png',
                        help='Output path for the figure')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and plot
    yeast, human = load_results(results_dir)
    plot_trust_comparison(yeast, human, output_path)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Yeast baseline: {yeast['baseline_accuracy']:.1%}")
    print(f"Human baseline: {human['baseline_accuracy']:.1%}")
    print(f"Best protein signal: deform_flip")
    print(f"  Yeast: AUROC={yeast['error_detection']['t_deform_flip']['auroc']:.3f}, +{yeast['selective_prediction']['t_deform_flip']['improvement']*100:.1f}%")
    print(f"  Human: AUROC={human['error_detection']['t_deform_flip']['auroc']:.3f}, +{human['selective_prediction']['t_deform_flip']['improvement']*100:.1f}%")


if __name__ == '__main__':
    main()
