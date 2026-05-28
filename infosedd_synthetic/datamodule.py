import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
try:
    from .distribution_generator.distributions import get_rv
except ImportError:
    from distribution_generator.distributions import get_rv
try:
    from .infosedd_utils import *
except ImportError:
    from infosedd_utils import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
from transformers import AutoTokenizer
import torch
import numpy as np
from collections.abc import Sequence

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
        self.csv_header = getattr(config, "csv_header", None)
        self.float_handling = getattr(config, "float_handling", "bin")
        self.float_bins = int(getattr(config, "float_bins", 10))
        self.add_channel_dim = bool(getattr(config, "add_channel_dim", False))
        self.config = config

    def _resolve_columns(self, df, col_spec, spec_name):
        if isinstance(col_spec, (str, bytes)):
            selectors = [col_spec]
        elif isinstance(col_spec, Sequence) or isinstance(col_spec, (np.ndarray, pd.Index)):
            selectors = list(col_spec)
        else:
            selectors = [col_spec]

        resolved_cols = []
        for selector in selectors:
            if isinstance(selector, (int, np.integer)):
                if selector < 0 or selector >= len(df.columns):
                    raise ValueError(
                        f"{spec_name} contains out-of-range index {selector}. "
                        f"Valid range is [0, {len(df.columns) - 1}]."
                    )
                resolved_cols.append(df.columns[int(selector)])
            elif isinstance(selector, str):
                if selector not in df.columns:
                    raise ValueError(
                        f"{spec_name} contains column name '{selector}' that does not exist. "
                        f"Available columns: {list(df.columns)}"
                    )
                resolved_cols.append(selector)
            else:
                raise ValueError(
                    f"{spec_name} must be a column index/name or a list of indexes/names. "
                    f"Got selector {selector!r} of type {type(selector).__name__}."
                )

        return resolved_cols

    def _encode_dataframe(self, df):
        encoded_cols = []
        has_unbinned_floats = False

        for col_name in df.columns:
            series = df[col_name]

            if pd.api.types.is_float_dtype(series):
                if self.float_handling == "bin":
                    binned = pd.cut(
                        series,
                        bins=self.float_bins,
                        labels=False,
                        include_lowest=True,
                        duplicates="drop",
                    ).fillna(0)
                    encoded_cols.append(binned.to_numpy(dtype=np.int64))
                elif self.float_handling == "keep":
                    encoded_cols.append(series.to_numpy(dtype=np.float32))
                    has_unbinned_floats = True
                else:
                    raise ValueError(
                        f"Unsupported float_handling='{self.float_handling}'. "
                        f"Use 'bin' or 'keep'."
                    )
            elif pd.api.types.is_numeric_dtype(series):
                encoded_cols.append(series.to_numpy(dtype=np.int64))
            else:
                codes, _ = pd.factorize(series.astype(str), sort=True)
                encoded_cols.append(codes.astype(np.int64))

        return np.column_stack(encoded_cols), has_unbinned_floats

    def _to_tensor(self, array, force_float=False):
        if force_float or np.issubdtype(array.dtype, np.floating):
            return torch.tensor(array, dtype=torch.float32)
        return torch.tensor(array, dtype=torch.long)

    def setup(self, stage=None):
        if self.float_handling == "bin" and self.float_bins < 2:
            raise ValueError("float_bins must be >= 2 when float_handling='bin'.")

        df = pd.read_csv(self.file_path, header=self.csv_header)
        print(f"First 5 rows of the dataset:\n{df.head()}")
        print(f"Dataset shape: {df.shape}")
        print(f"Dataframe types:\n{df.dtypes}")

        x_columns = self._resolve_columns(df, self.x_col, "x_col")
        y_columns = self._resolve_columns(df, self.y_col, "y_col")
        x_df = df.loc[:, x_columns]
        y_df = df.loc[:, y_columns]

        x, x_has_unbinned_floats = self._encode_dataframe(x_df)
        y, y_has_unbinned_floats = self._encode_dataframe(y_df)
        if x_has_unbinned_floats or y_has_unbinned_floats:
            print("WARNING: Float columns were kept as continuous values. Only MINDE can work on this kind of data.")

        x_discrete = np.issubdtype(x.dtype, np.integer)
        y_discrete = np.issubdtype(y.dtype, np.integer)
        if x_discrete and y_discrete:
            self.alphabet_size = int(max(x.max(), y.max()) + 1)
        else:
            self.alphabet_size = None

        if self.normalize:
            x = StandardScaler(copy=True).fit_transform(x)
            y = StandardScaler(copy=True).fit_transform(y)
            if self.add_channel_dim:
                x = np.expand_dims(x, axis=-1)
                y = np.expand_dims(y, axis=-1)
            x = self._to_tensor(x, force_float=True)
            y = self._to_tensor(y, force_float=True)
        else:
            if self.add_channel_dim:
                x = np.expand_dims(x, axis=-1)
                y = np.expand_dims(y, axis=-1)
            x = self._to_tensor(x)
            y = self._to_tensor(y)
        
        self.data = TensorDataset(x, y)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)
