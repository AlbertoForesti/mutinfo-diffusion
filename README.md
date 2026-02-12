# InfoSEDD Repository

This repository contains the implementation and experiments for Information Estimation with Discrete Diffusion, organized into two main directories:

## Repository Structure

### `infosedd-real-data/`
Contains experiments and implementations for real-world data applications:
- **Text experiments**: Natural language processing tasks
- **Promoter experiments**: Biological promoter sequence analysis
- **DNA experiments**: Genomic sequence modeling and analysis

This directory includes:
- Model implementations for various architectures (transformers, diffusion models, etc.)
- Training scripts for different data types
- Configuration files for experiments
- Data loading utilities

### `infosedd-synthetic/`
Contains experiments with synthetic data for method validation and analysis:
- Synthetic data generation and modeling
- Mutual information estimation experiments
- Sample complexity analysis
- Baseline comparisons with other methods

This directory includes:
- Synthetic data generators
- MI estimation utilities
- Training and evaluation scripts
- Configuration files for synthetic experiments

## Getting Started

Each directory contains its own README with specific instructions for running experiments and setting up the environment. Please refer to the individual README files in each subdirectory for detailed usage instructions.

## Acknowledgements
This repository is heavily based on [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) and [MDLM](https://github.com/kuleshov-group/mdlm)

## TODOs
- [x] Training code
- [x] Evaluation code
- [x] Basic coneniency scripts
- [ ] Add more conveniency scipts
