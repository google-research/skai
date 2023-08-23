r"""Configuration file for experiment with SKAI data and TwoTower model.
"""

import ml_collections
from skai.model.configs import base_config


def get_config() -> ml_collections.ConfigDict:
  """Get two tower config."""
  config = base_config.get_config()

  config.train_bias = False
  config.num_rounds = 1
  config.round_idx = 0
  config.train_stage_2_as_ensemble = False
  config.save_train_ids = False

  data = config.data
  data.name = 'skai'
  data.num_classes = 2
  data.subgroup_ids = ()
  data.subgroup_proportions = ()
  data.initial_sample_proportion = 1.
  data.tfds_dataset_name = 'skai_dataset'
  data.tfds_data_dir = '/tmp/skai_dataset'
  data.labeled_train_pattern = ''
  data.unlabeled_train_pattern = ''
  data.validation_pattern = ''
  data.use_post_disaster_only = False
  data.batch_size = 32

  model = config.model
  model.load_pretrained_weights = True
  model.name = 'two_tower'
  model.num_channels = 6

  config.optimizer.learning_rate = 1e-4
  config.optimizer.type = 'adam'

  config.training.num_epochs = 100

  return config
