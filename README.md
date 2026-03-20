# Information Estimation with Discrete Diffusion

This repository contains code and experiment assets for **Information Estimation with Discrete Diffusion**, split into synthetic and real-data settings.

## Paper

- OpenReview (PDF): https://openreview.net/pdf?id=m18MXVdrV9

## Repository Structure

### `infosedd-synthetic/`
Synthetic-data experiments for mutual information estimation:
- synthetic data generation
- InfoSEDD and baseline estimators
- Hydra configs and sweep scripts

See `infosedd-synthetic/README.md` for setup and usage.

### `infosedd-real-data/`
Real-data experiments:
- SummEval text experiments
- genomic/DNA experiments
- promoter analysis workflows

See `infosedd-real-data/README.md` for setup and usage.

## Acknowledgements

This repository is heavily based on [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) and [MDLM](https://github.com/kuleshov-group/mdlm).
