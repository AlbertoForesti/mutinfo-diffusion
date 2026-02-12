import pytorch_lightning as pl
import hydra
import os
import yaml
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("min", lambda x, y: min(x, y))

def _mi_estimate_run(config):
  if hasattr(config.data.config, 'noise_rv') and hasattr(config.estimator.config, 'alphabet_size') and config.data.config.noise_rv is not None:
    OmegaConf.update(config.estimator, "config.alphabet_size", config.estimator.config.alphabet_size * (config.data.config.noise_rv.n+1), force_add=True)
  data = hydra.utils.instantiate(config.data)
  estimator = hydra.utils.instantiate(config.estimator)
  logger = hydra.utils.instantiate(config.logger) if 'logger' in config else None
  trainer = hydra.utils.instantiate(config.trainer, default_root_dir=os.getcwd(), logger=logger)
  trainer.fit(estimator, datamodule=data)
  results = estimator.results
  if not os.path.exists(os.path.dirname(config.output_path)):
    os.makedirs(os.path.dirname(config.output_path))
  yaml.dump(results, open(config.output_path, 'w'))
  print(f"Results saved to {config.output_path}")


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  pl.seed_everything(config.seed)
  _mi_estimate_run(config)


if __name__ == '__main__':
  main()