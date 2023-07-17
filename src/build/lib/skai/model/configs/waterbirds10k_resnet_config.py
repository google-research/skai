"""Configuration file for experiment with Waterbirds data and ResNet model."""

import ml_collections
from skai.model.configs import base_config


def get_config() -> ml_collections.ConfigDict:
  """Get mlp config."""
  config = base_config.get_config()

  # Consider landbirds on water and waterbirds on land as subgroups.
  config.data.subgroup_ids = ()  # ('0_1', '1_0')
  config.data.subgroup_proportions = ()  # (0.04, 0.012)
  config.data.initial_sample_proportion = .25

  config.active_sampling.num_samples_per_round = 500
  config.num_rounds = 4

  data = config.data
  data.name = 'waterbirds10k'
  data.num_classes = 2
  data.corr_strength = 0.95

  model = config.model
  model.name = 'resnet'
  model.dropout_rate = 0.2

  return config
