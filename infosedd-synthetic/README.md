# Information Estimation with Discrete Diffusion

This repository contains the code and experiments for the anonymous submission "Information Estimation with Discrete Diffusion".

## Overview

InfoSEDD is a novel mutual information (MI) estimator for discrete random variables using score-based diffusion models. It leverages the connection between mutual information and score functions for accurate high-dimensional discrete MI estimation.

## Installation

```bash
# Create conda environment
conda create -n infosedd python=3.10
conda activate infosedd

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Interactive Tutorial

For a hands-on introduction, see the **quickstart notebook**:

```bash
jupyter notebook quickstart.ipynb
```

The notebook demonstrates:
- Basic usage of InfoSEDD and F-DIME estimators
- Configuration examples with different parameters
- Simple training loops for quick experimentation

### Using Shell Scripts (Recommended)

A convenient shell scripts is provided for running experiments:

```bash
cd scripts
bash run_exp.sh
```

### Manual Execution

```bash
# InfoSEDD
python train.py --config-name=config_infosedd

# F-DIME variants
python train.py --config-name=config_fdime_gan
python train.py --config-name=config_smile

# MINDE
python train.py --config-name=config_minde
```

### Key Parameters

To reproduce paper experiments, configure these parameters:

- `alphabet_size`: Support size of each random variable in the sequence
- `seq_length`: Length of each random vector (total sample length = 2 × seq_length)  
- `mutual_information`: Target MI value

```bash
# Example: alphabet_size=4, seq_length=20, MI=10
python train.py alphabet_size=4 seq_length=20 mutual_information=10

# Or edit configs/config_*.yaml files directly
```

## Repository Structure

```
├── configs/               # Hydra configurations
├── mi_estimator.py       # Main estimator class
├── datamodule.py         # Data generation
├── train.py              # Training script
├── quickstart.ipynb      # Interactive tutorial
├── scripts/              # Convenience scripts
└── requirements.txt      # Dependencies
```