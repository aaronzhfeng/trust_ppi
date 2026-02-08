# TrustPPI

**Deformation Stability as a Model-Agnostic Trust Signal for Protein-Protein Interaction Prediction**

Computational PPI models achieve high accuracy, but practitioners have no reliable way to know *which* predictions to trust. Generic confidence scores and even purpose-built uncertainty mechanisms (e.g., Gaussian Process variance) fail to detect errors under cross-species distribution shift.

TrustPPI introduces **deformation stability** — a model-agnostic trust signal grounded in the geometric deep learning principle that robust representations should be stable under small perturbations. By injecting Gaussian noise into a model's intermediate embeddings and measuring prediction consistency, TrustPPI identifies unreliable predictions without retraining or architectural modification.

## Key Results

- **Error detection**: AUROC 0.66–0.83 across three architectures (CNN, Transformer+GP, PLM) and eight species-level settings
- **Selective prediction**: +2–10% accuracy improvement at 80% coverage
- **Trust-guided selection**: Discovers true interactions 2–3× faster than confidence-based selection
- **Label-free assessment**: Aggregate deformation scores rank model reliability across species without ground truth

## Repository Structure

```
trust_ppi/
├── src/trustppi/                  # Core library
│   ├── trust/
│   │   ├── deformation_stability.py   # Core contribution
│   │   ├── metrics.py                 # Selective prediction metrics
│   │   ├── wrapper.py                 # Trust wrapper framework
│   │   ├── dscript_wrapper.py         # D-SCRIPT adapter
│   │   ├── interface_confidence.py    # Contact map trust signals
│   │   └── threshold_search.py        # Trust weight optimization
│   ├── models/
│   │   └── egnn.py                    # E(n)-equivariant GNN
│   ├── conformal/                     # Conformal prediction
│   │   ├── fcs.py                     # Full Conformal Sets
│   │   ├── aps.py                     # Adaptive Prediction Sets
│   │   └── nonconformity.py           # Score functions
│   ├── data.py                        # PPI dataset loading
│   ├── sabdab_data.py                 # SAbDab structural data
│   └── utils.py                       # Experiment utilities
├── experiments/                   # Paper experiments
│   ├── eval_dscript_trust.py          # Table 1: D-SCRIPT
│   ├── eval_tuna_trust.py             # Table 1: TUnA
│   ├── eval_plm_interact_trust.py     # Table 1: PLM-interact
│   ├── ablate_deformation.py          # Table 2: Ablation
│   ├── eval_trust_selection.py        # Table 3: Experiment selection
│   ├── eval_ood_baselines.py          # §4.2: OOD baselines
│   ├── eval_equipis.py                # Appendix B: EquiPPIS pilot
│   └── eval_conformal.py             # Appendix C: Conformal pilot
├── scripts/
│   ├── reproduce_all.sh               # Master reproduction script
│   ├── aggregate_multi_seed.py        # Multi-seed aggregation
│   └── analyze_cross_species.py       # Table 4: Cross-species
└── plots/                         # Figure generation scripts
```

## Installation

### 1. Environment Setup

```bash
conda create -n trustppi python=3.10
conda activate trustppi

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. External Model Dependencies

**D-SCRIPT** (required for Tables 1–3):
```bash
mkdir -p external && cd external
git clone https://github.com/samsledje/D-SCRIPT.git
cd D-SCRIPT && pip install -e . && cd ../..
```

**TUnA** (required for Table 1, TUnA rows):
```bash
cd external
git clone https://github.com/Wangbiolab/TUnA.git
# Follow TUnA setup instructions for pretrained weights
cd ..
```

**PLM-interact** (required for Table 1, PLM-interact rows):
```bash
# Download weights from HuggingFace: danliu1226/PLM-interact-650M-humanV11
mkdir -p checkpoints/plm_interact
# Place pytorch_model.bin and config.json in checkpoints/plm_interact/
```

### 3. Data Setup

PPI test datasets (TSV pair files + FASTA sequences) should be placed in `data/ppi/`. These use the standard benchmark splits from D-SCRIPT and TUnA.

## Reproducing Paper Results

### All Results at Once

```bash
bash scripts/reproduce_all.sh
```

### Individual Experiments

| Paper Element | Command |
|---|---|
| **Table 1** (D-SCRIPT) | `python -m experiments.eval_dscript_trust --data yeast --limit 500 --seed 42` |
| **Table 1** (TUnA) | `python -m experiments.eval_tuna_trust --data yeast --limit 500 --seed 42` |
| **Table 1** (PLM-interact) | `python -m experiments.eval_plm_interact_trust --data yeast --limit 500 --seed 42` |
| **Table 2** (Ablation) | `python -m experiments.ablate_deformation --data yeast --limit 200 --seed 42` |
| **Table 3** (Selection) | `python -m experiments.eval_trust_selection --data yeast --limit 500 --seed 42` |
| **Table 4** (Cross-species) | `python scripts/analyze_cross_species.py` |
| **§4.2** (OOD baselines) | `python -m experiments.eval_ood_baselines --data yeast --limit 200` |
| **Appendix B** (EquiPPIS) | `python -m experiments.eval_equipis --limit 100 --seed 42` |
| **Appendix C** (Conformal) | `python -m experiments.eval_conformal --data yeast --limit 500 --seed 42` |

All experiments support `--quick` for fast smoke testing with reduced samples.

Multi-seed results (mean ± std) are computed with:
```bash
python scripts/aggregate_multi_seed.py --model dscript --data yeast
```

## Citation

```bibtex
@inproceedings{trustppi2026,
  title={TrustPPI: Deformation Stability as a Model-Agnostic Trust Signal for Protein-Protein Interaction Prediction},
  author={Anonymous},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```
