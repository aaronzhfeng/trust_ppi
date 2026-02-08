#!/bin/bash
# =============================================================================
# TrustPPI: Reproduce All Paper Results
# =============================================================================
#
# This script reproduces all experiments from the paper:
#   "TrustPPI: Deformation Stability as a Model-Agnostic Trust Signal
#    for Protein-Protein Interaction Prediction"
#
# Prerequisites:
#   - Python environment with requirements.txt installed
#   - External models set up (D-SCRIPT, TUnA, PLM-interact)
#   - Data downloaded (see README.md)
#
# Usage:
#   bash scripts/reproduce_all.sh          # Full reproduction
#   bash scripts/reproduce_all.sh --quick  # Quick smoke test
#
# =============================================================================

set -e

QUICK=""
if [ "$1" = "--quick" ]; then
    QUICK="--quick"
    echo "=== QUICK MODE: Reduced samples for smoke testing ==="
fi

SEEDS=(42 123 456)
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "Table 1: D-SCRIPT Trust Evaluation (3 species × 3 seeds)"
echo "============================================================"
for seed in "${SEEDS[@]}"; do
    for data in yeast human ecoli; do
        echo "  D-SCRIPT | ${data} | seed=${seed}"
        python -m experiments.eval_dscript_trust \
            --data "$data" --limit 500 --seed "$seed" $QUICK
    done
done

echo ""
echo "============================================================"
echo "Table 1: TUnA Trust Evaluation (5 species × 3 seeds)"
echo "============================================================"
for seed in "${SEEDS[@]}"; do
    for data in yeast ecoli fly mouse worm; do
        echo "  TUnA | ${data} | seed=${seed}"
        python -m experiments.eval_tuna_trust \
            --data "$data" --limit 500 --seed "$seed" $QUICK
    done
done

echo ""
echo "============================================================"
echo "Table 1: PLM-interact Trust Evaluation (3 species × 3 seeds)"
echo "============================================================"
for seed in "${SEEDS[@]}"; do
    for data in yeast human ecoli; do
        echo "  PLM-interact | ${data} | seed=${seed}"
        python -m experiments.eval_plm_interact_trust \
            --data "$data" --limit 500 --seed "$seed" $QUICK
    done
done

echo ""
echo "============================================================"
echo "Table 2: Ablation Study (noise_std × n_perturbations)"
echo "============================================================"
python -m experiments.ablate_deformation \
    --data yeast --limit 200 --seed 42 $QUICK

echo ""
echo "============================================================"
echo "Table 3: Trust-Guided Experiment Selection (3 species)"
echo "============================================================"
for data in yeast human ecoli; do
    echo "  Selection | ${data}"
    python -m experiments.eval_trust_selection \
        --data "$data" --limit 500 --seed 42 $QUICK
done

echo ""
echo "============================================================"
echo "Section 4.2: OOD Baselines"
echo "============================================================"
python -m experiments.eval_ood_baselines \
    --data yeast --limit 200 $QUICK

echo ""
echo "============================================================"
echo "Table 4: Cross-Species Reliability (aggregate analysis)"
echo "============================================================"
python scripts/analyze_cross_species.py

echo ""
echo "============================================================"
echo "Appendix B: EquiPPIS Pilot"
echo "============================================================"
python -m experiments.eval_equipis --limit 100 --seed 42 $QUICK

echo ""
echo "============================================================"
echo "Appendix C: Conformal Prediction Pilot"
echo "============================================================"
python -m experiments.eval_conformal \
    --data yeast --limit 500 --seed 42 $QUICK

echo ""
echo "============================================================"
echo "Multi-Seed Aggregation"
echo "============================================================"
python scripts/aggregate_multi_seed.py --model dscript --data yeast
python scripts/aggregate_multi_seed.py --model tuna --data yeast

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Results saved to: ${RESULTS_DIR}/"
echo "============================================================"
