# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Binary to run training on a single model once.


# pylint: enable=line-too-long
"""

import logging as native_logging
import os

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import pandas as pd
from skai.model import data
from skai.model import generate_bias_table_lib
from skai.model import models
from skai.model import sampling_policies
from skai.model import train_lib
from skai.model.configs import base_config
from skai.model.train_strategy import get_strategy
import tensorflow as tf


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')
flags.DEFINE_bool('keep_logs', True, 'If True, creates a log file in output '
                  'directory. If False, only logs to console.')
flags.DEFINE_bool(
    'is_vertex', False, 'True if the training job will be executed on VertexAI.'
)
flags.DEFINE_enum(
  'accelerator_type',
  default='cpu',
  help='Accelerator to use for computations',
  enum_values=['cpu', 'gpu', 'tpu']
  )
flags.DEFINE_string('ensemble_dir', '', 'If specified, loads the models at '
                    'this directory to consider the ensemble.')
flags.DEFINE_string(
    'trial_name',
    None,
    'Name of the job trial that measurements should be submitted to. Format:'
    ' projects/{project}/locations/{location}/studies/{study}/trials/{trial}',
)


def main(_) -> None:
  config = FLAGS.config
  base_config.check_flags(config)

  if FLAGS.keep_logs and not config.training.log_to_xm:
    if not tf.io.gfile.exists(config.output_dir):
      tf.io.gfile.makedirs(config.output_dir)
    stream = tf.io.gfile.GFile(
        os.path.join(config.output_dir, 'log'), mode='w'
    )
    stream_handler = native_logging.StreamHandler(stream)
    logging.get_absl_logger().addHandler(stream_handler)

  dataset_builder = data.get_dataset(config.data.name)
  ds_kwargs = {}
  if config.data.name == 'waterbirds10k':
    ds_kwargs = {'corr_strength': config.data.corr_strength}
  elif config.data.name == 'skai':
    ds_kwargs.update({
        'tfds_dataset_name': config.data.tfds_dataset_name,
        'data_dir': config.data.tfds_data_dir,
        'labeled_train_pattern': config.data.labeled_train_pattern,
        'unlabeled_train_pattern': config.data.unlabeled_train_pattern,
        'validation_pattern': config.data.validation_pattern,
        'use_post_disaster_only': config.data.use_post_disaster_only,
        'load_small_images': config.data.load_small_images,
    })
    if config.data.use_post_disaster_only:
      config.model.num_channels = 3
  if config.upsampling.do_upsampling:
    ds_kwargs.update({
        'upsampling_lambda': config.upsampling.lambda_value,
        'upsampling_signal': config.upsampling.signal,
    })

  logging.info('Running Round %d of Training.', config.round_idx)
  get_split_config = lambda x: x if config.data.use_splits else 1
  if config.round_idx == 0:
    dataloader = dataset_builder(
        num_splits=get_split_config(config.data.num_splits),
        initial_sample_proportion=get_split_config(
            config.data.initial_sample_proportion),
        subgroup_ids=config.data.subgroup_ids,
        subgroup_proportions=config.data.subgroup_proportions, **ds_kwargs)
  else:
    # If latter round, keep track of split generated in last round of active
    # sampling
    dataloader = dataset_builder(config.data.num_splits,
                                 initial_sample_proportion=1,
                                 subgroup_ids=(),
                                 subgroup_proportions=(),
                                 **ds_kwargs)

    # Filter each split to only have examples from example_ids_table
    dataloader.train_splits = [
        dataloader.train_ds.filter(
            generate_bias_table_lib.filter_ids_fn(ids_tab)) for
        ids_tab in sampling_policies.convert_ids_to_table(config.ids_dir)]
  print("Ids dir: ", config.ids_dir)
  model_params = models.ModelTrainingParameters(
      model_name=config.model.name,
      train_bias=config.train_bias,
      num_classes=config.data.num_classes,
      num_subgroups=dataloader.num_subgroups,
      subgroup_sizes=dataloader.subgroup_sizes,
      worst_group_label=dataloader.worst_group_label,
      num_epochs=config.training.num_epochs,
      num_channels=config.model.num_channels,
      l2_regularization_factor=config.model.l2_regularization_factor,
      optimizer=config.optimizer.type,
      learning_rate=config.optimizer.learning_rate,
      batch_size=config.data.batch_size,
      load_pretrained_weights=config.model.load_pretrained_weights,
      use_pytorch_style_resnet=config.model.use_pytorch_style_resnet,
      do_reweighting=config.reweighting.do_reweighting,
      reweighting_lambda=config.reweighting.lambda_value,
      reweighting_signal=config.reweighting.signal
  )
  model_params.train_bias = config.train_bias

  if FLAGS.is_vertex:
    job_id = os.path.basename(FLAGS.trial_name)
    output_dir = os.path.join(config.output_dir, job_id)
    tf.io.gfile.makedirs(output_dir)
  else:
    #TODO - Choose a diretory name in case vertex ai is not used in running experiments
    output_dir = config.output_dir
  example_id_to_bias_table = None

  if config.train_bias or (config.reweighting.do_reweighting and
                           config.reweighting.signal == 'bias'):
    # Bias head will be trained as well, so gets bias labels.
    if config.path_to_existing_bias_table:
      example_id_to_bias_table = (
          generate_bias_table_lib.load_existing_bias_table(
              config.path_to_existing_bias_table,
              config.bias_head_prediction_signal,
          )
      )
    else:
      logging.info(
          'Error: Bias table not found')
      return
  if config.data.use_splits:
    # Training a single model on a combination of data splits.
    included_splits_idx = [int(i) for i in config.data.included_splits_idx]
    new_train_ds = data.gather_data_splits(included_splits_idx,
                                           dataloader.train_splits)
    val_ds = data.gather_data_splits(included_splits_idx, dataloader.val_splits)
  elif config.data.use_filtering:
    # Use filter tables to generate subsets.
    # This allows a better control over the number of trained models that.
    # The number of models is independent of the odd ratio. E.g., 10 splits with
    # an odd ratio 0f 0.5 trains 252 models and with an ood ratio of 0.1 only
    # 10. Using filitering we can train 50 models for both of these ood ratios.
    new_train_ds = data.filter_set(
        dataloader=dataloader,
        initial_sample_proportion=config.data.initial_sample_proportion,
        initial_sample_seed=config.data.initial_sample_seed,
        split_proportion=config.data.split_proportion,
        split_id=config.data.split_id,
        split_seed=config.data.split_seed,
        training=True
    )
    val_ds = data.filter_set(
                dataloader=dataloader,
        initial_sample_proportion=config.data.initial_sample_proportion,
        initial_sample_seed=config.data.initial_sample_seed,
        split_proportion=config.data.split_proportion,
        split_id=config.data.split_id,
        split_seed=config.data.split_seed,
        training=False
    )
  else:
    raise ValueError(
        'In `config.data`, one of `(use_splits, use_filtering)` must be True.')

  dataloader.train_ds = new_train_ds
  dataloader.eval_ds['val'] = val_ds
  experiment_name = 'stage_2' if config.train_bias else 'stage_1'

  if config.save_train_ids:
    table_name = 'training_ids_table'
    ids = data.get_ids_from_dataset(dataloader.train_ds)
    dict_values = {'example_id': ids}
    df = pd.DataFrame(dict_values)
    df.to_csv(os.path.join(output_dir, table_name + '.csv'), index=False)
  # Apply batching (must apply batching only after filtering)
  dataloader = data.apply_batch(dataloader, config.data.batch_size)

  if FLAGS.accelerator_type == 'tpu':
    # Encode string data components as numerical
    # This is useful when using TPU which does not accept string datatype
    dataloader = data.DataEncoder().encode_string_labels(dataloader)
    dataloader = data.DataEncoder().encode_example_ids(dataloader)

  strategy = get_strategy(accelerator_type=FLAGS.accelerator_type)

  _ = train_lib.train_and_evaluate(
      train_as_ensemble=config.train_stage_2_as_ensemble,
      dataloader=dataloader,
      model_params=model_params,
      num_splits=config.data.num_splits,
      ood_ratio=config.data.ood_ratio,
      output_dir=output_dir,
      experiment_name=experiment_name,
      save_model_checkpoints=config.training.save_model_checkpoints,
      save_best_model=config.training.save_best_model,
      early_stopping=config.training.early_stopping,
      ensemble_dir=FLAGS.ensemble_dir,
      example_id_to_bias_table=example_id_to_bias_table,
      vizier_trial_name=FLAGS.trial_name,
      is_vertex=FLAGS.is_vertex,
      strategy=strategy
  )


if __name__ == '__main__':
  app.run(main)