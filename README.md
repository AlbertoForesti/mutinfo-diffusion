# Information Estimation with Discrete Diffusion

This repository contains code and experiment assets for **Information Estimation with Discrete Diffusion**, split into synthetic and real-data settings.

## Global environment setup (uv + pyproject.toml)

This repository uses a **single global dependency setup** at the repository root via `pyproject.toml`.

```bash
# from repository root
uv venv .venv
source .venv/bin/activate
uv sync
```

## Paper

- OpenReview (PDF): https://openreview.net/pdf?id=m18MXVdrV9

## Repository Structure

### `infosedd-synthetic/`
Synthetic-data experiments for mutual information estimation:
- synthetic data generation
- InfoSEDD and baseline estimators
- Hydra configs and sweep scripts

See `infosedd-synthetic/README.md` for usage.

### `infosedd-real-data/`
Real-data experiments:
- SummEval text experiments
- genomic/DNA experiments
- promoter analysis workflows

See `infosedd-real-data/README.md` for usage.

## Acknowledgements

This repository is heavily based on [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) and [MDLM](https://github.com/kuleshov-group/mdlm).
