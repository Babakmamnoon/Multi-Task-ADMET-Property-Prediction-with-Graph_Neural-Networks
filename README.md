# Multi-Task ADMET Property Prediction with Graph Neural Networks
End-to-end multi-task molecular property prediction with Graph Neural Networks (PyTorch Geometric + RDKit). Predict multiple ADMET endpoints from molecular graphs, featuring masked multi-task learning, interpretability via atom saliency, comprehensive evaluation, and rich visualizations

A production-grade, end-to-end deep learning pipeline for predicting **Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET)** properties directly from molecular structure, using a multi-task **Graph Isomorphism Network with Edge Features (GINEConv)** built on **PyTorch Geometric** and **RDKit**.

The notebook is fully self-contained and executable in Google Colab with zero manual configuration.

## Launch Notebook

Run this notebook instantly in Google Colab (No installation required).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Babakmamnoon/Multi-Task-ADMET-Property-Prediction-with-Graph_Neural-Networks/blob/main/admet_multitask_gnn.ipynb)

---

## Project Goals

Early and accurate ADMET profiling is one of the leading bottlenecks in drug discovery — a large fraction of clinical-stage attrition is still attributable to poor pharmacokinetics and unanticipated toxicity. This project demonstrates how a single shared molecular graph encoder can be trained jointly across multiple ADMET endpoints to:

- Exploit shared chemical structure-property relationships across tasks (multi-task positive transfer)
- Handle realistically sparse/incomplete labels per molecule via masked multi-task loss
- Provide calibrated, benchmarked predictions with literature-grounded evaluation
- Offer atom-level interpretability for medicinal chemistry decision-making

---

## Datasets

Compound data and labels are sourced from the **[Therapeutics Data Commons (TDC)](https://tdcommons.ai/)** ADMET benchmark suite (Huang et al., *NeurIPS Datasets and Benchmarks*, 2021). If TDC is unreachable (e.g., no internet access in the runtime), the notebook automatically falls back to a chemically-grounded **synthetic dataset** generated from RDKit scaffolds with property labels derived from physicochemical descriptors (LogP, TPSA, molecular weight, aromaticity, H-bond donors/acceptors), so the full pipeline remains executable end-to-end regardless of connectivity.

| Task | Type | ADMET Domain | Source Dataset | Endpoint |
|---|---|---|---|---|
| `CYP2D6_Veith` | Classification | Metabolism | CYP2D6 inhibition (Veith et al.) | CYP2D6 enzyme inhibition |
| `CYP3A4_Veith` | Classification | Metabolism | CYP3A4 inhibition (Veith et al.) | CYP3A4 enzyme inhibition |
| `hERG` | Classification | Toxicity | hERG Central | hERG cardiac channel blockade |
| `AMES` | Classification | Toxicity | AMES mutagenicity | Bacterial reverse mutation (genotoxicity) |
| `Solubility_AqSolDB` | Regression | Physicochemical | AqSolDB | Aqueous solubility (log S) |
| `Lipophilicity_AstraZeneca` | Regression | Physicochemical | AstraZeneca | Lipophilicity (log D7.4) |

Molecules are standardized and validated with RDKit (sanitization, canonicalization, size filtering 2–80 heavy atoms, de-duplication), and split using a **Bemis–Murcko scaffold split** (80/10/10) to provide a realistic, leakage-resistant estimate of generalization to novel chemical series — the field-standard practice over random splitting (Wu et al., *Chem. Sci.*, 2018).

---

## Methodology

### Molecular Representation
- Atoms and bonds are converted into graphs using RDKit, encoding: atom type, degree, formal charge, hybridization, aromaticity, total Hs, ring membership, and atomic mass; bonds encode type, conjugation, ring membership, and stereochemistry.

### Model Architecture
- **Shared encoder**: 4-layer `GINEConv` stack with edge-conditioned message passing, batch normalization, ReLU, and dropout, followed by jumping-knowledge-style concatenation across layers.
- **Readout**: Combined mean + sum graph pooling, projected into a shared molecular embedding.
- **Task heads**: Independent 2-layer MLP heads per ADMET endpoint sharing the upstream encoder, enabling cross-task representation learning.

### Training
- **Masked multi-task loss**: BCE-with-logits for classification tasks and Huber loss for regression tasks, both masked per-sample to handle missing labels without label imputation bias.
- Regression targets standardized using train-split-only statistics to avoid data leakage.

### Evaluation
- Classification: ROC-AUC, PR-AUC (average precision), confusion matrices.
- Regression: RMSE, MAE, R².
- All metrics benchmarked qualitatively against published MoleculeNet/TDC leaderboard baselines for context.

### Interpretability
- Atom-level saliency via input-gradient attribution, highlighting substructures driving individual predictions — useful for medicinal chemist review and SAR rationalization.

---

## Expected Results

Performance will vary depending on whether real TDC data or the synthetic fallback dataset is used (the notebook reports which source was active). As a general reference point, when trained on authentic TDC data, GNN-based multi-task models on these endpoints typically fall in the following ranges, consistent with published MoleculeNet/TDC benchmark results:

| Task | Metric | Typical Literature Range |
|---|---|---|
| CYP2D6_Veith | ROC-AUC | 0.75 – 0.86 |
| CYP3A4_Veith | ROC-AUC | 0.80 – 0.90 |
| hERG | ROC-AUC | 0.78 – 0.88 |
| AMES | ROC-AUC | 0.80 – 0.89 |
| Solubility_AqSolDB | RMSE (log S) | 0.7 – 1.1 |
| Lipophilicity_AstraZeneca | RMSE (log D) | 0.5 – 0.8 |

On the synthetic fallback dataset, metrics tend to appear stronger than real-world TDC performance because labels are deterministically derived from a small set of RDKit descriptors, making the structure-property relationship easier to learn — this is expected and clearly distinguishable in the notebook's reported data source.

---

## Repository Contents

```
admet_multitask_gnn.ipynb   # Full Colab-ready notebook (data, model, training, evaluation, visualization)
README.md                   # This file
```

## Quick Start

1. Open `admet_multitask_gnn.ipynb` in Google Colab.
2. Run all cells top to bottom — dependencies (PyTorch, PyTorch Geometric, RDKit, PyTDC) auto-install in Cell 2.
3. No API keys or manual dataset downloads required.

## Tech Stack

`PyTorch` · `PyTorch Geometric` · `RDKit` · `PyTDC` · `scikit-learn` · `pandas` / `numpy` · `matplotlib` / `seaborn` · `UMAP`

## References

1. Huang, K. *et al.* "Therapeutics Data Commons: A Foundation for Therapeutic Science." *NeurIPS Datasets and Benchmarks*, 2021.
2. Hu, W. *et al.* "Strategies for Pre-training Graph Neural Networks." *ICLR*, 2020.
3. Xiong, Z. *et al.* "Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism." *J. Med. Chem.*, 2020.
4. Wu, Z. *et al.* "MoleculeNet: A Benchmark for Molecular Machine Learning." *Chem. Sci.*, 2018.
5. Veith, H. *et al.* "Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries." *Nat. Biotechnol.*, 2009.

## License

MIT License — free to use, modify, and distribute with attribution.

## Author

Babak Mamnoon / cheminformatics / AI-driven drug discovery portfolio focused on production-grade, reproducible GNN pipelines for molecular property prediction.
