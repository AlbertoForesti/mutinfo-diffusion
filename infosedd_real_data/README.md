# Information Estimation with Discrete Diffusion

This repository contains the code for **"Information Estimation with Discrete Diffusion"**.

## 📝 Text Experiments

### 1. Data Preparation
Follow the instructions from [SummEval](https://github.com/Yale-LILY/SummEval) to download and pair the summarization data.

### 2. Training Models

Run from the repository root unless noted otherwise.

**Train discriminative baselines on SummEval:**
```bash
bash infosedd-real-data/scripts/summeval/train_fdimeMINEsummeval.sh
bash infosedd-real-data/scripts/summeval/train_fdimeNWJsummeval.sh
bash infosedd-real-data/scripts/summeval/train_fdimeSMILEsummeval.sh
bash infosedd-real-data/scripts/summeval/train_fdimeHDsummeval.sh
bash infosedd-real-data/scripts/summeval/train_fdimeGANsummeval.sh
```

**Train INFO-SEDD:**
```bash
cd infosedd-real-data
bash scripts/summeval/train_infosedd_summeval.sh
```

> Note: several scripts use hardcoded paths (for example `data_path`) and GPU IDs. Update them before running.

### 3. Consistency Tests
Run the training scripts with the following configuration:
- Set `p_random` values linearly spaced from 0.0, 0.1, ..., 1.0
- Set `data_path` to the `.jsonl` file of M22
- Note: `p_random` is equivalent to `1-ρ` in the paper

### 4. Mutual Information Computation
To compute MI for each summarizer:
- Set `p_random = 0.0`
- Update the path to paired summaries for your chosen model

---

## 🧬 DNA Experiments

**Train discriminative baselines:**
```bash
bash infosedd-real-data/scripts/dna/train_fdimeMINE.sh
bash infosedd-real-data/scripts/dna/train_fdimeNWJ.sh
bash infosedd-real-data/scripts/dna/train_fdimesmile.sh
bash infosedd-real-data/scripts/dna/train_fdimeHD.sh
bash infosedd-real-data/scripts/dna/train_fdimeGAN.sh
```

**Train INFO-SEDD:**
```bash
bash infosedd-real-data/scripts/dna/train_infosedd.sh
```

---

## 🔍 Motif Selection Experiments

### 1. Data Acquisition
Download the required genomic data from [CNNPromoterData](https://github.com/solovictor/CNNPromoterData):
- `Arabidopsis_non_prom_big.fa`
- `Arabidopsis_tata.fa`

### 2. Dataset Creation
Generate the HuggingFace dataset:
```bash
jupyter notebook create_promoter_dataset.ipynb
```

### 3. Model Training
Train the INFO-SEDD model on the promoter dataset:
```bash
cd infosedd-real-data
bash scripts/promoters/train_infosedd_promoters.sh
```

### 4. Motif Discovery
Run motif selection analysis:
```bash
cd infosedd-real-data
bash scripts/promoters/motif_selection.sh
```

**Required inputs:**
- Path to the generated dataset
- Path to the trained model checkpoint

In `motif_selection.sh`, replace placeholder values:
- `data_path=???`
- `eval.checkpoint_path=???`

Also note that motif selection writes output to a hardcoded path in `main.py`:
- `/home/foresti/mdlm/motif_selection.txt`

**Output:** A `.txt` file where each row contains:
```
start_of_mask mutual_information_mean mutual_information_std
```

---

## 📋 Key Dependencies

- **PyTorch**: 2.1.1 with CUDA 11.8
- **Lightning**: Framework for training
- **Transformers**: 4.47.0
- **Flash Attention**: 2.3.6+
- **Mamba SSM**: State space models
- **Causal Conv1D**: Convolution operations

Core dependencies are managed in the repository root `pyproject.toml`.
Use `uv sync` to install them, or `uv sync --extra caduceus` for motif-related extras.
