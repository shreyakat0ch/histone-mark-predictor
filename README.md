# Histone Mark Predictor

A deep learning framework for cell-conditioned, multi-task prediction of chromatin signals from DNA sequence and global cellular state.

## Overview

This repository implements a hybrid CNN-Transformer architecture designed to predict continuous histone mark intensities across diverse cellular contexts. By integrating local genomic sequences with global RNA-seq expression fingerprints via **FiLM (Feature-wise Linear Modulation)**, the model generalizes to unseen cell lines and captures cell-type specific regulatory logic.

### Key Features
- **Cell-Conditioned Latent Space**: Leverages RNA-seq fingerprints to modulate DNA sequence representations.
- **Multi-Task Regression**: Predicts intensities for multiple marks (H3K27ac, H3K4me3, H3K27me3, H3K9me3, H3K4me1) simultaneously.
- **Mark-Specific Context Awareness**: Optimized receptive fields for sharp (promoter) vs. broad (repressor) marks.
- **High-Performance Pipeline**: Memory-mapped binary genome caching for O(1) random access during training.

## Project Status: Completed Research Prototype
This project is a finalized research implementation demonstrating robust Pearson correlation across 440+ held-out cell lines.

> [!NOTE]
> This codebase represents the final architecture for the Histone Mark Predictor project. Architectural details and hyperparameters are fixed as per the validated results.

## Architecture

The model follows a tiered design:
1. **DNA Encoder**: Multi-layer CNN stem for local feature extraction, followed by a Transformer Encoder for long-range dependency modeling.
2. **RNA Fingerprinting**: MLP-based compression of top variable genes.
3. **Fusion Layer**: FiLM conditioning parameters ($\gamma, \beta$) generated from RNA embeddings to modulate DNA features.
4. **Task Heads**: Attention-pooled shared representations routed to mark-specific regression heads.

## How to Use

### 1. Data Preparation
Organize your data in the `data/` directory (or set `HISTONE_DATA_DIR`):
- `data/<cell_line>/chip_histone/*.bed`: narrowPeak files for target marks.
- `data/<cell_line>/RNA/*.tsv`: Expression files.
- `data/hg38.fa`: Reference genome.

### 2. Preprocessing
Run the offline data pipeline to index peaks and compute global RNA fingerprints:
```bash
python src/data_pipeline.py
```

### 3. Training
Start the multi-task training loop:
```bash
python src/train.py
```
Checkpoints and diagnostic plots will be saved to `runs/`.

### 4. Evaluation
Evaluate the model on the held-out test set (by cell line):
```bash
python src/evaluate.py
```
This generates Pearson, Spearman, and classification metrics (AUROC/AUPRC) across all marks.

## Citations & Related Work

This work builds upon foundational research in genomic deep learning:

- **Basenji**: Kelley, D. R., et al. (2018). "Sequential regulatory activity prediction across chromosomes with convolutional neural networks." *Nature Communications*.
- **Enformer**: Avsec, Ž., et al. (2021). "Effective gene expression prediction from sequence by integrating long-range interactions." *Methods in Molecular Biology*.
- **DeepSEA**: Zhou, J., & Troyanskaya, O. G. (2015). "Predicting effects of noncoding variants with deep learning-based sequence model." *Nature Methods*.
- **FiLM**: Perez, E., et al. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." *AAAI*.

## License

Copyright (c) 2026 shreya katoch. All rights reserved. 

This source code is provided solely for portfolio review by recruiters and hiring managers. No permission is granted for redistribution or use pending academic publication. See `LICENSE` for full details.
