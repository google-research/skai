r"""Configuration file for experiment with SKAI data and ResNet model.

"""


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
  data.name = 'skai'
  data.num_classes = 2
  # TODO(jlee24): Determine what are considered subgroups in SKAI domain
  # and add support for identifying by ID.
  data.subgroup_ids = ()
  data.subgroup_proportions = ()
  data.initial_sample_proportion = 1.
  data.tfds_dataset_name = 'skai_dataset'
  data.tfds_data_dir = '/tmp/skai_dataset'
  data.labeled_train_pattern = ''
  data.unlabeled_train_pattern = ''
  data.validation_pattern = ''
  data.use_post_disaster_only = False

  model = config.model
  model.name = 'resnet50v2'
  model.num_channels = 6

  return config
