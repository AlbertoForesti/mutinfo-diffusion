import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import re
import yaml

try:
    from . import dataloader
    from . import diffusion
    from . import utils
except ImportError:
    import dataloader
    import diffusion
    import utils

from hydra.utils import instantiate
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

import numpy as np

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver("model_length", lambda seq_length: 2 * int(seq_length))

def get_data_key(dataconfig):
  if hasattr(dataconfig, "random_variable"):
    return f"{dataconfig.train}_seqlen={dataconfig.random_variable.seq_length}_dim={dataconfig.random_variable.dim}_mutinfo={dataconfig.random_variable.mutual_information}"
  if "summ" in dataconfig.train:
    model_id = extract_model_id(dataconfig.train)
    return f"summeval_summarizer={model_id}_{os.path.basename(dataconfig.train)}"
  if "Genomic" in dataconfig.train:
    return dataconfig.train.split('/')[-1]
  raise NotImplementedError(f"Resolver not implemented for train={dataconfig.train} and valid={dataconfig.valid}")

def get_model_key(modelconfig):
  return modelconfig.name

omegaconf.OmegaConf.register_new_resolver("get_data_key", get_data_key)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def mutinfo_eval(config, logger, tokenizer):
  logger.info('Starting Information Metrics Eval.')

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  if hasattr(valid_ds.dataset, 'var_indices'):
    # If dataset has var_indices, pass them to the model config
    if omegaconf.OmegaConf.select(config, "mutinfo") is not None:
        omegaconf.OmegaConf.update(config.mutinfo, "var_indices", valid_ds.dataset.var_indices, force_add=True)
    else:
        omegaconf.OmegaConf.update(config, "mutinfo.var_indices", valid_ds.dataset.var_indices, force_add=True)
  else:
    omegaconf.OmegaConf.update(config, "mutinfo.var_indices", valid_ds.dataset.var_indices, force_add=True)
    omegaconf.OmegaConf.update(config, "mutinfo._var_indices", valid_ds.dataset.var_indices, force_add=True)
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger,
    limit_val_batches=config.eval.mc_estimates)
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  trainer.validate(model, valid_ds)
  mutinfo_estimate = float(model.valid_mutinfo)
  mutinfo_std = float(model.valid_mutinfo_std)
  print(f"Mutual information estimate: {mutinfo_estimate} +/- {mutinfo_std}")

def motif_selection(config, logger, tokenizer):
  logger.info('Starting Motif Selection.')
  
  lower_bound = config.lower_bound
  upper_bound = config.upper_bound
  start = max(0, lower_bound)
  end = max(min(252-config.box_length, upper_bound),start+1)
  mutinfos = []
  mutinfos_std = []
  for i in tqdm(range(start,end), desc="Motif Selection"):

    model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
    if config.eval.disable_ema:
      logger.info('Disabling EMA.')
      model.ema = None

    if config.mask_type == 'remove':
      mask_indices = list(range(1+i, 1+i+config.box_length))
    elif config.mask_type == 'keep':
      mask_indices = list(range(1,1+i)) + list(range(1+i+config.box_length, 252))
    else:
      raise ValueError(f"Unknown mask type {config.mask_type}.")

    omegaconf.OmegaConf.update(config.data, "mask_indeces", 
                          mask_indices, 
                          force_add=True)
    
    callbacks = []
    if 'callbacks' in config:
      for _, callback in config.callbacks.items():
        callbacks.append(hydra.utils.instantiate(callback))
    omegaconf.OmegaConf.update(config, "trainer.limit_val_batches", config.eval.mc_estimates, force_add=True)
    _, valid_ds = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, valid_seed=config.seed)
    if hasattr(valid_ds.dataset, 'var_indices'):
      # If dataset has var_indices, pass them to the model config
      if omegaconf.OmegaConf.select(config, "mutinfo") is not None:
          omegaconf.OmegaConf.update(config.mutinfo, "var_indices", valid_ds.dataset.var_indices, force_add=True)
      else:
          omegaconf.OmegaConf.update(config, "mutinfo.var_indices", valid_ds.dataset.var_indices, force_add=True)
    else:
      omegaconf.OmegaConf.update(config, "mutinfo.var_indices", mask_indices, force_add=True)
      omegaconf.OmegaConf.update(config, "mutinfo._var_indices", mask_indices, force_add=True)
    trainer = hydra.utils.instantiate(
      config.trainer,
      default_root_dir=os.getcwd(),
      callbacks=callbacks,
      strategy=hydra.utils.instantiate(config.strategy),
      logger=None,
      enable_progress_bar=False,
      enable_model_summary=False,)

    # Create the masked dataloader
    masked_dataloader = dataloader.InfiniteDataLoader(
        valid_ds,
        mask_indices=mask_indices,
        mask_token_id=tokenizer.mask_token_id
    )

    # Use the masked dataloader for validation
    model.config.mutinfo.var_indices = masked_dataloader.dataset.var_indices
    trainer.validate(model, masked_dataloader)
    mutinfo_estimate = float(model.valid_mutinfo)
    mutinfos.append(deepcopy(mutinfo_estimate))
    mutinfos_std.append(deepcopy(model.valid_mutinfo_std))
    # After trainer.validate
    torch.cuda.synchronize(device=config.strategy.device)  # Wait for all CUDA operations to complete

    # Free memory manually if needed
    del masked_dataloader
    import gc
    gc.collect()
    torch.cuda.empty_cache()
  output_path = getattr(config, "motif_output_path", "/home/foresti/mdlm/motif_selection.txt")
  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(output_path, 'w') as f:
    for i in range(end-start):
      f.write(f"{start+i} {mutinfos[i]} {mutinfos_std[i]}\n")
  logger.info(f"Motif selection results saved to {output_path}")
  

def extract_model_id(path):
    """Extract model ID (like M7) from a path"""
    # Use regex to find MX pattern in the path
    match = re.search(r'/M(\d+)/', path)
    if match:
        return f"M{match.group(1)}"
    return None

def _get_unique_wandb_config(config):
  wandb_config = dict(config.wandb)
    
  # Generate a unique name that includes run parameters (for multirun)
  base_name = wandb_config.pop("name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
  
  # Add a unique identifier - combine timestamp with random string
  import uuid
  unique_id = str(uuid.uuid4())[:8]
  job_id = os.environ.get("HYDRA_JOB_NUM", "0")
  
  # Add key parameters that distinguish this run in the multirun
  param_str = ""
  assert not hasattr(config.wandb, 'tags'), "wandb tags are set automatically, do not set them in the config"
  if hasattr(config, 'train_marginal'):
    param_str += f"_marg{config.train_marginal}"
  param_str = f"{get_data_key(config.data)}_{get_model_key(config.model)}"
  tags = [get_data_key(config.data), get_model_key(config.model)]
  omegaconf.OmegaConf.update(config, "wandb.tags", tags, force_add=True)
  # Create the unique run name
  wandb_config["name"] = f"{base_name}_job{job_id}_{unique_id}{param_str}_{config.parameterization}"
  
  # Force creation of a new run
  wandb_config["id"] = None
  wandb_config["resume"] = "never"
  return wandb_config

def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    # Create a unique name for the W&B run in multirun mode

    wandb_config = _get_unique_wandb_config(config)
    
    wandb_logger = L.pytorch.loggers.WandbLogger(
        config=omegaconf.OmegaConf.to_object(config),
        **wandb_config)
    
  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None
  
  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, valid_seed=config.seed)
  _print_batch(train_ds, valid_ds, tokenizer)
  if config.eval.compute_mutinfo or config.training.compute_mutinfo:
    omegaconf.OmegaConf.update(config, "trainer.limit_val_batches", config.eval.mc_estimates, force_add=True)
    if hasattr(train_ds.dataset, 'var_indices'):
      # If dataset has var_indices, pass them to the model config
      if omegaconf.OmegaConf.select(config, "mutinfo") is not None:
          raise ValueError("config.mutinfo already exists.")
      else:
          omegaconf.OmegaConf.update(config, "mutinfo.var_indices", train_ds.dataset.var_indices, force_add=True)
    else:
      raise ValueError("Dataset does not have var_indices.")
  
  # Setup DORA fine-tuning if enabled
  if hasattr(config, "dora") and config.dora.enabled:
    logger.info('DORA parameter-efficient fine-tuning enabled')
    # Make sure we're using a pre-trained model
    if not hasattr(config.model, "pretrained") or not config.model.pretrained:
        logger.warning('DORA is meant for fine-tuning pretrained models. Setting pretrained=True')
        omegaconf.OmegaConf.update(config, "model.pretrained", True, force_add=True)
    
    # Add PEFT configuration to the model config
    omegaconf.OmegaConf.update(config, "model.peft_config", config.dora, force_add=True)
  
  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)
  
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

  last_mi_estimate = model.valid_mutinfo
  result_dir = config.mi_estimates_save_dir
  if config.parameterization == "discriminative":
    result_dir = os.path.join(result_dir, "discriminative", config.divergence)
  else:
    result_dir = os.path.join(result_dir, "infosedd", config.variant)
  if config.data.p_random is not None:
    result_dir = os.path.join(result_dir, f"p_random={config.data.p_random}")
  if result_dir is not None:
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)
    with open(os.path.join(result_dir, "final_mutinfo.txt"), 'w') as f:
      f.write(f"{last_mi_estimate}\n")
    logger.info(f"Final mutual information estimate {last_mi_estimate} saved to {os.path.join(result_dir, 'final_mutinfo.txt')}")
  else:
    logger.info(f"Final mutual information estimate: {last_mi_estimate}, but result_dir is not set so not saving it.")


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.mode == 'mutinfo_eval':
    mutinfo_eval(config, logger, tokenizer)
  elif config.mode == 'motif_selection':
    motif_selection(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()
