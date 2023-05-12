"""Configuration file for experiment with Waterbirds baseline experiment."""

import ml_collections
from skai.model.configs import base_config


def get_config() -> ml_collections.ConfigDict:
  """Get mlp config."""
  config = base_config.get_config()

  config.train_bias = False
  config.num_rounds = 1
  config.round_idx = 0
  config.train_stage_2_as_ensemble = False
  config.save_train_ids = False

  data = config.data
  data.name = 'waterbirds'
  data.num_classes = 2
  data.subgroup_ids = ()  # ('0_1', '1_0')
  data.subgroup_proportions = ()  # (0.04, 0.012)
  data.initial_sample_proportion = 1.

  model = config.model
  model.name = 'resnet50v2'

  config.upsampling.do_upsampling = True

  # Set to 0 to compute introspection signal based on the best epoch.
  config.eval.num_signal_ckpts = 0
  return config
