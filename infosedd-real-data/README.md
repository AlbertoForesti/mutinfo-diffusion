# Information Estimation with Discrete Diffusion

This repository contains the code for the anonymous submission **"Information Estimation with Discrete Diffusion"**.

## 🚀 Installation

### Prerequisites
- Python 3.10
- NVIDIA GPU with Flash Attention support
- Conda or Miniconda

### Setup Environment

1. **Create and activate conda environment:**
   ```bash
   conda create -n infosedd python=3.10
   conda activate infosedd
   ```

2. **Install CUDA toolkit:**
   ```bash
   conda install cuda-toolkit=11.8 -c nvidia
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/VanessB/pytorch-kld
   ```

---

## 📝 Text Experiments

### 1. Data Preparation
Follow the instructions from [SummEval](https://github.com/Yale-LILY/SummEval) to download and pair the summarization data.

### 2. Training Models

**Train MINE-like estimator:**
```bash
bash train_mine_summeval.sh
```
> 💡 Change the `loss` field to train different model variants.

**Train INFO-SEDD:**
```bash
bash train_infosedd_summeval.sh
```

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

**Train MINE-like estimator:**
```bash
bash train_mine_dna.sh
```
> 💡 Change the `loss` field to train different model variants.

**Train INFO-SEDD:**
```bash
bash train_infosedd_dna.sh
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
bash train_infosedd_promoters.sh
```

### 4. Motif Discovery
Run motif selection analysis:
```bash
bash motif_selection.sh
```

**Required inputs:**
- Path to the generated dataset
- Path to the trained model checkpoint

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

For the complete dependency list, see the requirements file