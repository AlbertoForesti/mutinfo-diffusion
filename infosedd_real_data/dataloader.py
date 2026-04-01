import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers
import numpy as np

try:
    from . import utils
except ImportError:
    import utils
import hydra

from itertools import cycle
from datasets import load_dataset

LOGGER = utils.get_logger(__name__)


def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x

class IdentityTokenizer(transformers.PreTrainedTokenizer):
  def __init__(
      self,
      vocab_size):
    self._vocab_size = vocab_size
    super().__init__()
  
  def _tokenize(self, data, **kwargs):
    return data
  
  def _convert_token_to_id(self, token):
    return token
  
  def _convert_id_to_token(self, index):
    return index
  
  def convert_tokens_to_string(self, tokens):
    return ''.join(str(tokens))
  
  def get_vocab(self):
    return {str(i): i for i in range(self.vocab_size)}
  
  @property
  def vocab_size(self):
    return self._vocab_size


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result

def get_summeval_dataset(dataset_name, tokenizer, wrap, mode,
    field_size_dict, block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False, p_random=0.0):
  assert sum(list(field_size_dict.values())) == block_size, f"Total field size must be {block_size}, instead got {list(field_size_dict.values())} with sum {sum(list(field_size_dict.values()))}"
  field_length_str = '_'.join(
      [f'{k}{v}' for k, v in field_size_dict.items()])
  if wrap:
    filename = f'{dataset_name.replace("/","")}_{mode}_bs{block_size}_{field_length_str}_wrapped.pt'
  else:
    filename = f'{dataset_name.replace("/","")}_{mode}_bs{block_size}_{field_length_str}_unwrapped.pt'

  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]
  print(f"EOS is {EOS}, BOS is {BOS}\n************************************")
  def preprocess_and_tokenize(example, field, max_field_length):
    text = example[field]
      
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                        max_length=max_field_length,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        return_token_type_ids=True,)
        # print(len(tokens['input_ids']))
    return tokens
  data = load_dataset('json', data_files=dataset_name)['train']
  tokenized_datasets_by_field = {}
  for field, max_field_length in field_size_dict.items():
    preprocess_and_tokenize_field = functools.partial(
      preprocess_and_tokenize, field=field, max_field_length=max_field_length)
    if streaming:
      tokenized_dataset = data.map(
        preprocess_and_tokenize_field,
        batched=True,
        desc='Tokenizing')
    else:
      tokenized_dataset = data.map(
        preprocess_and_tokenize_field,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc='Tokenizing')
    tokenized_datasets_by_field[field] = tokenized_dataset
  
  tokenized_dataset = ConcatenatedDataset(
    tokenized_datasets_by_field,
    p_random=p_random)
  return tokenized_dataset


def get_dataset(
    dataset_name, tokenizer, wrap, mode,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False, field_size_dict=None, p_random=0.0, data_config=None):
  
  if 'aligned' in dataset_name:
    assert field_size_dict is not None, f"field_size_dict must be provided for {dataset_name} dataset"
    return get_summeval_dataset(
      dataset_name, tokenizer, wrap, mode,
      field_size_dict=field_size_dict, block_size=block_size, num_proc=num_proc, streaming=streaming, p_random=p_random)
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2

  if 'Arabidopsis' in dataset_name:
    dataset = datasets.load_from_disk(dataset_name)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      streaming=streaming)

  if 'Genomic' in dataset_name:
    if mode == 'validation':
      return dataset['test']
    else:
      data = dataset[mode]
  else:
    data = dataset[mode]
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  if "Genomic" in dataset_name:
    num_classes = np.unique(data['label']).shape[0]
    
    encoding_length = int(np.ceil(np.log(num_classes) / np.log(4)))

  def preprocess_and_tokenize(example):
    if 'Genomic' in dataset_name:
      text = example['seq']
      label = example['label']
      if "Arabidopsis" in dataset_name:
        assert label == example['is_promoter'], f"label: {label}, is_promoter: {example['is_promoter']}"

      label_to_id = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
      def get_label_or_seq(lbl):
        if np.random.rand() < p_random:
          lbl = np.random.randint(0, num_classes)
        lbl_enc = ''
        while lbl > 0:
          lbl_enc = label_to_id[lbl % 4] + lbl_enc
          lbl = lbl // 4
        if len(lbl_enc) < encoding_length:
          lbl_enc = 'A' * (encoding_length - len(lbl_enc)) + lbl_enc
        return lbl_enc
      label = list(map(get_label_or_seq, label))
      text = list(map(lambda x: f'{x[0]}{x[1]}', zip(label, text)))


    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                        max_length=block_size,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        return_token_type_ids=True,)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  if 'Genomic' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['seq', 'label'])
    if "Arabidopsis" in dataset_name:
      tokenized_dataset = tokenized_dataset.remove_columns(
        ['description', 'gene_id', 'is_promoter', 'fp_id', 'range_info'])
    setattr(tokenized_dataset, 'var_indices', [list(range(encoding_length)), list(range(encoding_length, block_size))])
    print(f"var_indices: {tokenized_dataset.var_indices}")
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    return tokenized_dataset.with_format('torch')
  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  if "synthetic" in config.data.train:
    try:
      tokenizer = IdentityTokenizer(
      vocab_size=config.data.random_variable.dim)
    except:
      tokenizer = IdentityTokenizer(
      vocab_size=2)
    return tokenizer
  
  if config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path, trust_remote_code=True)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer

def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  if "synthetic" in config.data.train:
    return get_synthetic_dataloaders(config, tokenizer)
  if skip_train:
    train_set = None
  else:
    if hasattr(config.data, 'field_size_dict'):
      field_size_dict = config.data.field_size_dict
    else:
      field_size_dict = None
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      block_size=config.model.length,
      field_size_dict=field_size_dict,
      p_random=config.data.p_random,
      data_config=config.data)
  
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  elif 'Genomic' in config.data.valid:
    if 'Arabidopsis' in config.data.valid and config.mode == "motif_selection":
      validation_split = 'test'
    else:
      validation_split = 'train'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    if hasattr(config.data, 'field_size_dict'):
      field_size_dict = config.data.field_size_dict
    else:
      field_size_dict = None
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      block_size=config.model.length,
      field_size_dict=field_size_dict,
      streaming=False,
      p_random=config.data.p_random,
      data_config=config.data)

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    if config.eval.compute_mutinfo:
      valid_loader = InfiniteDataLoader(valid_loader)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer
  return train_loader, valid_loader

class MaskedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies masking to specific token positions"""
    
    def __init__(self, dataset, mask_indices, mask_token_id):
        """
        Args:
            dataset: The original dataset
            mask_indices: List of indices to mask in the input_ids
            mask_token_id: Token ID to use for masking
        """
        self.dataset = dataset
        self.mask_indices = mask_indices
        self.mask_token_id = mask_token_id
        # Preserve any additional attributes from the original dataset
        self._var_indices = getattr(dataset, 'var_indices', None)
        self._var_indices[0] = list(
            set(self._var_indices[0]) - set(self.mask_indices) - set(range(252,1030)))
        self._var_indices[1] = list(
            set(self._var_indices[1]) - set(self.mask_indices) - set(range(252,1030)))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get the original item
        item = self.dataset[idx]
        
        # Create a copy of input_ids to avoid modifying the original
        input_ids = item['input_ids'].clone()

        # Apply masking
        for idx in self.mask_indices:
            if idx < len(input_ids):
                input_ids[idx] = self.mask_token_id
        
        ret = {**item, 'input_ids': input_ids}
        # Return the modified item
        # raise UserWarning(f"label: {input_ids[0]}, seq: {input_ids[1:]}, mask_indices: {self.mask_indices}, var_indices: {self._var_indices}")
        return ret
    
    @property
    def var_indices(self):
        return self._var_indices


# Modify InfiniteDataLoader to accept mask_indices and apply masking
class InfiniteDataLoader:
    def __init__(self, dataloader, mask_indices=None, mask_token_id=None):
        """
        Args:
            dataloader: The original dataloader
            mask_indices: Optional list of indices to mask in input_ids
            mask_token_id: Token ID to use for masking
        """
        if isinstance(dataloader, InfiniteDataLoader):
            dataloader = dataloader.dataloader

        if mask_indices is not None and mask_token_id is not None:
            # Wrap the dataset with masking
            masked_dataset = MaskedDataset(
                dataloader.dataset, 
                mask_indices,
                mask_token_id
            )
            # Create a new dataloader with the masked dataset
            self.dataloader = torch.utils.data.DataLoader(
                masked_dataset,
                batch_size=dataloader.batch_size,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=getattr(dataloader, 'pin_memory', False)
            )
        else:
            # Use the original dataloader
            self.dataloader = dataloader
            
        self.iterator = cycle(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)
    
    def __len__(self):
        return np.inf
    
    @property
    def dataset(self):
        return self.dataloader.dataset

class ConcatenatedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, p_random=0.0):
        self.datasets = datasets
        self._var_indices = None
        self.p_random = p_random

    def __len__(self):
        return len(next(iter(self.datasets.values())))

    def __getitem__(self, idx):
        input_ids_list = []
        attention_mask_list = []

        for i, dataset in enumerate(self.datasets.values()):
            if i > 0 and np.random.rand() < self.p_random:
              random_idx = torch.randint(
              low=0, high=len(dataset), size=(1,)).item()
              idx = random_idx
            item = dataset[idx]
            input_ids_list.extend(item['input_ids'])
            if 'attention_mask' in item:
                attention_mask_list.append(torch.tensor(item['attention_mask']))

        input_ids = torch.tensor(input_ids_list)
        if attention_mask_list:
            attention_mask = torch.cat(attention_mask_list, dim=0)
        else:
            attention_mask = torch.ones_like(input_ids)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    @property
    def var_indices(self):
      if self._var_indices is not None:
        return self._var_indices
      self._var_indices = []
      for dataset in self.datasets.values():
        example = dataset[0]['input_ids']
        if len(self._var_indices) == 0:
          self._var_indices.append(list(range(len(example))))
        else:
          self._var_indices.append(list(
            range(self._var_indices[-1][-1] + 1, self._var_indices[-1][-1] + 1 + len(example))))
      return self._var_indices

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data):
      # check data is iterable
      assert hasattr(data, '__iter__'), 'Data must be iterable.'
      self.data = torch.cat(data, dim=1)
      self._var_indices = []
      for var in data:
        if len(self._var_indices) == 0:
          self._var_indices.append(list(range(var.shape[1])))
        else:
          self._var_indices.append(list(
            range(self._var_indices[-1][-1] + 1, self._var_indices[-1][-1] + 1 + var.shape[1])))
    
    def __len__(self):
      return len(self.data)
    
    def __getitem__(self, idx):
      return {'input_ids': self.data[idx], 'attention_mask': torch.ones_like(self.data[idx])}
    
    @property
    def var_indices(self):
      return self._var_indices

def get_synthetic_dataloaders(config, tokenizer):

  random_variable = hydra.utils.instantiate(
    config.data.random_variable)
  
  outputs = random_variable.rvs(
    config.data.train_size)
  
  if isinstance(outputs, tuple):
    xy_train = (
      torch.tensor(outputs[0], dtype=torch.long),
      torch.tensor(outputs[1], dtype=torch.long
      ))
  else:
    xy_train = (torch.tensor(outputs, dtype=torch.long),)
  
  xy_valid = (xy_train[0][:config.data.train_size // 5], 
              xy_train[1][:config.data.train_size // 5])
  xy_train = (xy_train[0][config.data.train_size // 5:],
              xy_train[1][config.data.train_size // 5:])

  train_loader = torch.utils.data.DataLoader(
    SyntheticDataset(xy_train),
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=True)
  train_loader.tokenizer = tokenizer

  valid_loader = torch.utils.data.DataLoader(
    SyntheticDataset(xy_valid),
    batch_size=config.loader.eval_batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False)
  if config.eval.compute_mutinfo:
    valid_loader = InfiniteDataLoader(valid_loader)
  valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader

# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0