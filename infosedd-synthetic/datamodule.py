import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from distribution_generator.distributions import get_rv
from infosedd_utils import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
from transformers import AutoTokenizer
import torch

class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.alphabet_size = config.alphabet_size
        self.mutual_information = config.mutual_information
        self.n_samples = config.n_samples
        self.normalize = config.normalize
        self.noise_rv = config.noise_rv
        self.load_path = config.load_path
        self.config = config

    def setup(self, stage=None):
        if self.load_path is not None:
            print("Loading dataset from:", self.load_path)
            self.data = torch.load(self.load_path)
            return
        self.rv = get_rv(self.config.mutual_information,\
                        dim=self.alphabet_size,\
                        seq_length=self.seq_length,\
                        min_val=1e-3,\
                        n_generations=self.config.n_generations, \
                        noise_rv=self.config.noise_rv, \
                        )
        assert np.isclose(self.rv.mutual_information, self.mutual_information, rtol=1e-2, atol=1e-3), \
            f"Expected mutual information {self.mutual_information}, but got {self.rv.mutual_information}"
        x, y = self.rv.rvs(self.n_samples)
        if self.normalize:
            x = StandardScaler(copy=True).fit_transform(x)
            y = StandardScaler(copy=True).fit_transform(y)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)

        self.data = TensorDataset(x, y)
        if self.config.save_dataset:
            torch.save(self.data, f'{self.config.save_root}/synthetic_dataset_mi_{self.mutual_information}_samples_{self.n_samples}.pt')
    
    def save_dataset_with_n_samples(self, n_samples):
        x, y = self.rv.rvs(n_samples)
        if self.normalize:
            x = StandardScaler(copy=True).fit_transform(x)
            y = StandardScaler(copy=True).fit_transform(y)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)

        data = TensorDataset(x, y)
        torch.save(data, f'{self.config.save_root}/synthetic_dataset_mi_{self.mutual_information}_samples_{n_samples}.pt')

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

class CSVDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.file_path = config.file_path
        self.normalize = config.normalize
        self.x_col = config.x_col
        self.y_col = config.y_col
        self.config = config

    def setup(self, stage=None):
        df = pd.read_csv(self.file_path, header=None)
        print(f"First 5 rows of the dataset:\n{df.head()}")
        print(f"Dataset shape: {df.shape}")
        print(f"Dataframe types:\n{df.dtypes}")
        x_text = df.iloc[:,self.x_col].values
        y_text = df.iloc[:,self.y_col].values

        all_texts = np.concatenate([x_text, y_text])
        unique_texts = np.unique(all_texts)

        # Character-level tokenization

        self.char_to_id = {char: i for i, char in enumerate(sorted(unique_texts))}
        self.alphabet_size = len(self.char_to_id)
        
        def tokenize(texts):
            tokens = []
            for text in texts:
                token_ids = [self.char_to_id[char] for char in text]
                tokens.append(token_ids)
            return np.array(tokens)
        
        x = tokenize(x_text)
        print(f"First 5 tokenized x samples:\n{x[:5]}")
        y = tokenize(y_text)
        
        if self.normalize:
            x = StandardScaler(copy=True).fit_transform(x)
            y = StandardScaler(copy=True).fit_transform(y)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
        
        self.data = TensorDataset(x, y)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)