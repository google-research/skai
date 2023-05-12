r"""Binary executable for generating ids to sample in next round.

This file serves as a binary to compute the ids of samples to be included in
next round of training in an active learning loop.


Note: In output_dir, models trained on different splits of data must already
exist and be present in directory.
"""

import os

from absl import app
from absl import flags
from ml_collections import config_flags
import pandas as pd
from skai.model import sampling_policies
from skai.model.configs import base_config
import tensorflow as tf


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')


def main(_) -> None:

  config = FLAGS.config
  base_config.check_flags(config)
  bias_table = pd.read_csv(os.path.join(config.output_dir, 'bias_table.csv'))
  predictions_table = pd.read_csv(os.path.join(config.output_dir,
                                               'predictions_table.csv'))
  tf.io.gfile.makedirs(config.ids_dir)
  _ = sampling_policies.sample_and_split_ids(
      bias_table['example_id'].to_numpy(),
      predictions_table,
      config.active_sampling.sampling_score,
      config.active_sampling.num_samples_per_round,
      config.data.num_splits,
      config.ids_dir,
      True)

if __name__ == '__main__':
  app.run(main)
