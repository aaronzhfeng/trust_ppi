"""Utility functions for TrustPPI experiments."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def save_experiment_results(
    experiment_name: str,
    metrics: Dict[str, Any],
    figures: Optional[Dict] = None,
    tier: str = "tier1_validation",
    project_root: Optional[Path] = None
) -> Path:
    """
    Save experiment results in standard format.

    Args:
        experiment_name: Name of the experiment (e.g., 'swap_test')
        metrics: Dictionary of metrics to save
        figures: Dictionary of matplotlib figures to save (optional)
        tier: Experiment tier directory name
        project_root: Root directory of the project (auto-detected if None)

    Returns:
        Path to the saved metrics file
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent

    # Create results directory
    results_dir = project_root / "experiments" / tier / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics as JSON
    metrics_file = results_dir / f"{experiment_name}_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "timestamp": timestamp,
            "metrics": metrics
        }, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Save figures
    if figures:
        for name, fig in figures.items():
            fig_file = results_dir / f"{experiment_name}_{name}_{timestamp}.png"
            fig.savefig(fig_file, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {fig_file}")

    return metrics_file


def log_experiment_to_markdown(
    experiment_name: str,
    goal: str,
    setup: Dict[str, str],
    results: Dict[str, Any],
    figures: list,
    conclusion: str,
    next_step: str,
    tier: int,
    exp_num: int,
    project_root: Optional[Path] = None
) -> None:
    """
    Append experiment entry to the experiment log.

    Args:
        experiment_name: Name of the experiment
        goal: Goal of the experiment
        setup: Dictionary of setup parameters
        results: Dictionary of metric results
        figures: List of figure paths
        conclusion: Conclusion from the experiment
        next_step: Next step to take
        tier: Tier number (1, 2, or 3)
        exp_num: Experiment number within tier
        project_root: Root directory of the project
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent

    log_file = project_root / "docs" / "03_EXPERIMENT_LOG.md"
    date = datetime.now().strftime("%Y-%m-%d")

    entry = f"\n## {date}: {experiment_name} (Tier {tier}, Exp {exp_num})\n\n"
    entry += f"**Goal**: {goal}\n\n"
    entry += "**Setup**:\n"
    for key, value in setup.items():
        entry += f"- {key}: {value}\n"
    entry += "\n"
    entry += "**Results**:\n"
    entry += "| Metric | Value |\n"
    entry += "|--------|-------|\n"
    for key, value in results.items():
        if isinstance(value, float):
            entry += f"| {key} | {value:.4f} |\n"
        else:
            entry += f"| {key} | {value} |\n"
    entry += "\n"
    if figures:
        entry += "**Figures**:\n"
        for fig_path in figures:
            entry += f"- `{fig_path}`\n"
        entry += "\n"
    entry += f"**Conclusion**: {conclusion}\n\n"
    entry += f"**Next**: {next_step}\n\n"
    entry += "---\n"

    # Read existing content
    with open(log_file, 'r') as f:
        content = f.read()

    # Find the Tier 1 section and append
    tier_section = f"## Tier {tier}:"
    if tier_section in content:
        # Find the next tier section or end
        tier_start = content.find(tier_section)
        next_tier = content.find(f"## Tier {tier + 1}:", tier_start)
        if next_tier == -1:
            # Append at the end
            content = content.rstrip() + entry
        else:
            # Insert before next tier
            content = content[:next_tier] + entry + content[next_tier:]
    else:
        content = content.rstrip() + entry

    with open(log_file, 'w') as f:
        f.write(content)

    print(f"Logged experiment to {log_file}")
