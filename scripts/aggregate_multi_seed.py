"""
Aggregate TrustPPI results across multiple seeds.

Computes mean ± std for key metrics and runs paired significance tests.

Usage:
    # By model and dataset (auto-discovers result files)
    python scripts/aggregate_multi_seed.py --model dscript --data ecoli
    python scripts/aggregate_multi_seed.py --model tuna --data yeast
    python scripts/aggregate_multi_seed.py --model plm-interact --data human

    # Or by glob pattern
    python scripts/aggregate_multi_seed.py --pattern "experiments/tier3_full/results/protein_trust_eval_ecoli_seed*"

    # Save output
    python scripts/aggregate_multi_seed.py --model dscript --data ecoli --output results/agg_dscript_ecoli.json
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

# Model-to-pattern mapping
MODEL_PATTERNS = {
    'dscript': {
        'dir': 'experiments/tier3_full/results',
        'prefix': 'protein_trust_eval',
    },
    'tuna': {
        'dir': 'experiments/tier4_cross_model/results',
        'prefix': 'tuna_trust',
    },
    'plm-interact': {
        'dir': 'experiments/tier4_cross_model/results',
        'prefix': 'plm_interact_trust',
    },
    'equipis': {
        'dir': 'experiments/tier4_cross_model/results',
        'prefix': 'equipis_trust',
    },
}


def discover_result_files(
    model: str,
    dataset: str,
    project_root: Path = PROJECT_ROOT,
) -> List[Path]:
    """Find result files for a given model and dataset across all seeds."""
    if model not in MODEL_PATTERNS:
        raise ValueError(f"Unknown model: {model}. Choose from {list(MODEL_PATTERNS.keys())}")

    info = MODEL_PATTERNS[model]
    results_dir = project_root / info['dir']
    pattern = f"{info['prefix']}_{dataset}_seed*_*.json"

    files = sorted(results_dir.glob(pattern))
    return files


def discover_by_pattern(pattern: str, project_root: Path = PROJECT_ROOT) -> List[Path]:
    """Find files matching a glob pattern."""
    return sorted(project_root.glob(pattern))


def deduplicate_by_seed(files: List[Path]) -> List[Path]:
    """Keep only the latest file per seed."""
    seed_files: Dict[str, Path] = {}
    for f in files:
        name = f.stem
        # Extract seed from filename: ..._seed42_20260129_...
        for part in name.split('_'):
            if part.startswith('seed'):
                seed = part
                # Keep latest (files are sorted, last = latest)
                seed_files[seed] = f
                break
    return sorted(seed_files.values())


# ---------------------------------------------------------------------------
# Metric Extraction
# ---------------------------------------------------------------------------

def extract_metrics(result: Dict) -> Dict[str, float]:
    """Extract key metrics from a result JSON."""
    metrics = {}

    # Baseline accuracy
    metrics['baseline_accuracy'] = result.get('baseline_accuracy', 0.0)
    metrics['n_samples'] = result.get('n_samples', 0)

    # Error detection AUROC
    ed = result.get('error_detection', {})
    for signal in ['confidence', 't_deform_stable', 't_deform_flip',
                    't_deform_combined', 'gp_variance']:
        if signal in ed:
            metrics[f'{signal}_auroc'] = ed[signal].get('auroc', 0.0)
            metrics[f'{signal}_correlation'] = ed[signal].get('correlation', 0.0)

    # Selective prediction
    sp = result.get('selective_prediction', {})
    for signal in ['confidence', 't_deform_stable', 't_deform_flip',
                    't_deform_combined', 'gp_variance']:
        if signal in sp:
            metrics[f'{signal}_improvement'] = sp[signal].get('improvement', 0.0)
            metrics[f'{signal}_accuracy'] = sp[signal].get('accuracy', 0.0)

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict]:
    """Compute mean ± std across seeds."""
    if not all_metrics:
        return {}

    # Collect all keys
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    aggregated = {}
    for key in sorted(all_keys):
        values = [m[key] for m in all_metrics if key in m]
        if not values:
            continue
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'n': len(values),
            'values': [float(v) for v in values],
        }

    return aggregated


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

def significance_tests(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict]:
    """Run paired significance tests between deformation and confidence."""
    tests = {}

    # Need at least 3 seeds for meaningful tests
    if len(all_metrics) < 3:
        return {'note': f'Only {len(all_metrics)} seeds — need ≥3 for significance tests'}

    try:
        from scipy import stats
    except ImportError:
        return {'note': 'scipy not available for significance tests'}

    # Deformation vs Confidence — AUROC
    deform_aurocs = [m.get('t_deform_combined_auroc', 0) for m in all_metrics]
    conf_aurocs = [m.get('confidence_auroc', 0) for m in all_metrics]

    if all(d > 0 for d in deform_aurocs) and all(c > 0 for c in conf_aurocs):
        t_stat, p_value = stats.ttest_rel(deform_aurocs, conf_aurocs)
        tests['deform_vs_confidence_auroc'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_005': p_value < 0.05,
            'deform_wins': float(np.mean(deform_aurocs)) > float(np.mean(conf_aurocs)),
        }

    # Deformation vs Confidence — Improvement
    deform_imps = [m.get('t_deform_combined_improvement', 0) for m in all_metrics]
    conf_imps = [m.get('confidence_improvement', 0) for m in all_metrics]

    if all(d != 0 for d in deform_imps) or all(c != 0 for c in conf_imps):
        t_stat, p_value = stats.ttest_rel(deform_imps, conf_imps)
        tests['deform_vs_confidence_improvement'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_005': p_value < 0.05,
            'deform_wins': float(np.mean(deform_imps)) > float(np.mean(conf_imps)),
        }

    # GP variance vs Deformation (TUnA only)
    gp_aurocs = [m.get('gp_variance_auroc', 0) for m in all_metrics]
    if all(g > 0 for g in gp_aurocs):
        t_stat, p_value = stats.ttest_rel(deform_aurocs, gp_aurocs)
        tests['deform_vs_gp_auroc'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_005': p_value < 0.05,
            'deform_wins': float(np.mean(deform_aurocs)) > float(np.mean(gp_aurocs)),
        }

    return tests


# ---------------------------------------------------------------------------
# Per-Species Analysis
# ---------------------------------------------------------------------------

def cross_species_analysis(
    model: str,
    species_list: List[str],
    project_root: Path = PROJECT_ROOT,
) -> Dict:
    """Analyze patterns across species for a given model."""
    species_metrics = {}

    for species in species_list:
        files = discover_result_files(model, species, project_root)
        files = deduplicate_by_seed(files)
        if not files:
            continue

        all_metrics = []
        for f in files:
            with open(f) as fp:
                result = json.load(fp)
            all_metrics.append(extract_metrics(result))

        agg = aggregate_metrics(all_metrics)
        species_metrics[species] = agg

    # Compute cross-species correlations
    analysis = {'species_metrics': species_metrics}

    baselines = []
    improvements = []
    for sp, agg in species_metrics.items():
        if 'baseline_accuracy' in agg and 't_deform_combined_improvement' in agg:
            baselines.append(agg['baseline_accuracy']['mean'])
            improvements.append(agg['t_deform_combined_improvement']['mean'])

    if len(baselines) >= 3:
        try:
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(baselines, improvements)
            analysis['baseline_vs_improvement_correlation'] = {
                'pearson_r': float(corr),
                'p_value': float(p_value),
                'interpretation': (
                    'Worse models benefit more from deformation stability'
                    if corr < 0 else
                    'Better models also benefit from deformation stability'
                ),
            }
        except ImportError:
            pass

    return analysis


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary(aggregated: Dict, tests: Dict, files: List[Path]):
    """Print formatted summary."""
    print(f"\n{'=' * 70}")
    print(f"Multi-Seed Aggregation — {len(files)} seeds")
    print(f"{'=' * 70}")

    for f in files:
        print(f"  {f.name}")

    print(f"\n{'Metric':<35} {'Mean':>8} {'± Std':>8} {'  Range':>16}")
    print("-" * 70)

    for key, val in aggregated.items():
        if key == 'n_samples':
            print(f"  {key:<33} {val['mean']:>8.0f} {'±':>2}{val['std']:>6.0f}")
        elif 'auroc' in key or 'accuracy' in key:
            print(f"  {key:<33} {val['mean']:>8.3f} {'±':>2}{val['std']:>6.3f}"
                  f"  [{val['min']:.3f}, {val['max']:.3f}]")
        elif 'improvement' in key:
            print(f"  {key:<33} {val['mean']:>+7.1%} {'±':>2}{val['std']:>5.1%}"
                  f"  [{val['min']:+.1%}, {val['max']:+.1%}]")
        else:
            print(f"  {key:<33} {val['mean']:>8.3f} {'±':>2}{val['std']:>6.3f}")

    if tests and 'note' not in tests:
        print(f"\n{'Significance Tests':}")
        print("-" * 70)
        for name, result in tests.items():
            sig = "**" if result.get('significant_at_005') else "ns"
            winner = "deform" if result.get('deform_wins') else "baseline"
            print(f"  {name}: p={result['p_value']:.4f} {sig} (winner: {winner})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate TrustPPI results across multiple seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', type=str,
                        choices=list(MODEL_PATTERNS.keys()),
                        help='Model name')
    parser.add_argument('--data', type=str,
                        help='Dataset name (e.g., yeast, ecoli, human)')
    parser.add_argument('--pattern', type=str,
                        help='Glob pattern for result files (alternative to --model/--data)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--cross-species', action='store_true',
                        help='Run cross-species analysis for given model')
    parser.add_argument('--species', nargs='+',
                        default=['yeast', 'human', 'ecoli'],
                        help='Species list for cross-species analysis')

    args = parser.parse_args()

    if not args.pattern and not (args.model and args.data) and not args.cross_species:
        parser.error("Provide either --model/--data, --pattern, or --cross-species")

    # Cross-species analysis mode
    if args.cross_species:
        if not args.model:
            parser.error("--cross-species requires --model")
        analysis = cross_species_analysis(args.model, args.species)

        print(f"\n{'=' * 70}")
        print(f"Cross-Species Analysis — {args.model}")
        print(f"{'=' * 70}")

        for species, metrics in analysis.get('species_metrics', {}).items():
            baseline = metrics.get('baseline_accuracy', {}).get('mean', 0)
            deform = metrics.get('t_deform_combined_auroc', {}).get('mean', 0)
            imp = metrics.get('t_deform_combined_improvement', {}).get('mean', 0)
            n_seeds = metrics.get('baseline_accuracy', {}).get('n', 0)
            print(f"  {species:>8}: baseline={baseline:.1%}, "
                  f"deform_AUROC={deform:.3f}, improvement={imp:+.1%} "
                  f"({n_seeds} seeds)")

        corr = analysis.get('baseline_vs_improvement_correlation')
        if corr:
            print(f"\n  Baseline vs Improvement: r={corr['pearson_r']:.3f}, "
                  f"p={corr['p_value']:.4f}")
            print(f"  → {corr['interpretation']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nSaved to: {args.output}")

        return

    # Single model+dataset mode
    if args.pattern:
        files = discover_by_pattern(args.pattern)
    else:
        files = discover_result_files(args.model, args.data)

    files = deduplicate_by_seed(files)

    if not files:
        print("No result files found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} result files (deduplicated by seed)")

    # Load and extract metrics
    all_metrics = []
    all_seeds = []
    for f in files:
        with open(f) as fp:
            result = json.load(fp)
        all_metrics.append(extract_metrics(result))
        all_seeds.append(result.get('seed', 'unknown'))

    # Aggregate
    aggregated = aggregate_metrics(all_metrics)

    # Significance tests
    tests = significance_tests(all_metrics)

    # Print
    print_summary(aggregated, tests, files)

    # Save
    if args.output:
        output = {
            'model': args.model or 'unknown',
            'dataset': args.data or 'unknown',
            'n_seeds': len(files),
            'seeds': all_seeds,
            'files': [str(f) for f in files],
            'metrics': aggregated,
            'significance_tests': tests,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
