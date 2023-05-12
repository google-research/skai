"""Configuration file for experiment with Waterbirds data and ResNet model."""

import ml_collections
from skai.model.configs import base_config


def get_config() -> ml_collections.ConfigDict:
  """Get mlp config."""
  config = base_config.get_config()

  # Consider landbirds on water and waterbirds on land as subgroups.
  config.data.subgroup_ids = ('0_1', '1_0')

  data = config.data
  data.name = 'waterbirds'
  data.num_classes = 2

  model = config.model
  model.name = 'resnet50v2'
  model.dropout_rate = 0.2

  return config
