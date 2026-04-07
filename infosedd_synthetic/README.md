# Information Estimation with Discrete Diffusion

This repository contains the code and experiments for the anonymous submission "Information Estimation with Discrete Diffusion".

## Overview

InfoSEDD is a novel mutual information (MI) estimator for discrete random variables using score-based diffusion models. It leverages the connection between mutual information and score functions for accurate high-dimensional discrete MI estimation.

## Installation

Use the **global repository environment** from the root:

```bash
cd ..
uv venv .venv
source .venv/bin/activate
uv sync
```

This installs dependencies from the root `pyproject.toml` and works for both subprojects.

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

Convenience scripts are provided under `scripts/`:

```bash
# run these from the repository root
bash infosedd-synthetic/scripts/run_exp.sh
bash infosedd-synthetic/scripts/single_run.sh
```

> Note: these scripts call `cd infosedd-synthetic` internally and include hardcoded `CUDA_VISIBLE_DEVICES` values. Adjust them for your machine.

### Manual Execution

```bash
cd infosedd-synthetic

# InfoSEDD
python train.py --config-name=config estimator=infosedd_j
python train.py --config-name=config estimator=infosedd_c

# F-DIME variants
python train.py --config-name=config estimator=fdime_gan
python train.py --config-name=config estimator=fdime_hd
python train.py --config-name=config estimator=fdime_mine
python train.py --config-name=config estimator=fdime_nwj
python train.py --config-name=config estimator=fdime_smile

# MINDE
python train.py --config-name=config estimator=minde
```

### Key Parameters

To reproduce paper experiments, configure these parameters:

- `alphabet_size`: support size of each random variable in the sequence
- `seq_length`: length of each random vector
- `mutual_information`: target MI value

```bash
cd infosedd-synthetic

# Example: alphabet_size=4, seq_length=20, MI=10
python train.py --config-name=config alphabet_size=4 seq_length=20 mutual_information=10

# Or edit files under configs/ directly
```

## Repository Structure

```
├── configs/              # Hydra configuration tree
├── mi_estimator.py       # Estimator implementation
├── datamodule.py         # Synthetic data module
├── train.py              # Main training entrypoint
├── quickstart.ipynb      # Interactive tutorial
├── scripts/              # run_exp.sh / single_run.sh
└── ...                   # Dependencies managed at repository root
```
