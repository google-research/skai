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

r"""XM Launcher.


# pylint: enable=line-too-long
"""

import asyncio
import itertools
import os

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections import config_flags
import tensorflow as tf
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib.internal import parameter_controller


NUM_SLICES_DEFAULT = 5
# Subdirectory for models trained on splits in FLAGS.output_dir.
COMBOS_SUBDIR = 'combos'
# Subdirectory for checkpoints in FLAGS.output_dir.
CHECKPOINTS_SUBDIR = 'checkpoints'

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')


def main(_) -> None:
  config = FLAGS.config
  config_path = config_flags.get_config_filename(FLAGS['config'])
  config_filename = config_path.split('/')[-1]
  config_resource = xm_abc.Fileset(
      # Dict from a path in google3 to a path in the package.
      files={os.path.join('//', config_path): config_filename}
  )

  with xm_abc.create_experiment(
      experiment_title=(
          'Introspective Active Sampling'
          f' {config.data.name}_{config.model.name}'
      ),
      attribution_urls=['rh/efforts/1910'],
  ) as experiment:
    executor = xm_abc.Borg(
        requirements=xm.JobRequirements(
            location='li',
            service_tier=xm.ServiceTier.PROD,
            cpu=8,
            ram=16 * xm.GiB,
            tmp_ram_fs=32 * xm.GiB,
            v100=2))

    [train_executable, generate_bias_table_executable, sampling_executable
    ] = experiment.package([
        xm.bazel_binary(
            label='//third_party/py/skai/model:train',
            dependencies=[config_resource],
            bazel_args=xm_abc.bazel_args.gpu(),
            executor_spec=xm_abc.Borg.Spec(),
            args={
                'config':
                    config_resource.get_path(config_filename,
                                             xm_abc.Borg.Spec()),
            }),
        xm.bazel_binary(
            label='//third_party/py/skai/model:generate_bias_table',
            dependencies=[config_resource],
            bazel_args=xm_abc.bazel_args.gpu(),
            executor_spec=xm_abc.Borg.Spec(),
            args={
                'config':
                    config_resource.get_path(config_filename,
                                             xm_abc.Borg.Spec()),
            }),
        xm.bazel_binary(
            label='//third_party/py/skai/model:sample_ids',
            dependencies=[config_resource],
            bazel_args=xm_abc.bazel_args.gpu(),
            executor_spec=xm_abc.Borg.Spec(),
            args={
                'config':
                    config_resource.get_path(config_filename,
                                             xm_abc.Borg.Spec()),
            }),
    ])

    @parameter_controller.controller(interpreter=xm_abc.ml_python())
    async def run_train_rounds(
        experiment: xm.Experiment,
        config: ml_collections.ConfigDict
    ) -> None:

      num_rounds = config.num_rounds
      ids_dir = ''

      num_splits = config.data.num_splits
      num_ood_splits = int(num_splits * config.data.ood_ratio)
      num_id_splits = num_splits - num_ood_splits
      train_combos = [
          list(c) for c in list(
              itertools.combinations(range(num_splits), num_id_splits))
      ]
      output_dir = os.path.join(config.output_dir, 'no_introspection')
      output_dir = os.path.join(
          output_dir, config.active_sampling.sampling_score)

      for round_idx in range(num_rounds):
        logging.info('Running Round %d of Active Learning.', round_idx)
        round_dir = os.path.join(output_dir, f'round_{round_idx}')
        ids_dir = os.path.join(round_dir, 'ids')
        tf.io.gfile.makedirs(round_dir)

        logging.info('Training on different splits ...')
        # Launch training on multiple datasets in parallel.
        train_operations = []
        for combo in train_combos:
          combo_name = '_'.join(map(str, combo))
          combo_dir = os.path.join(round_dir, COMBOS_SUBDIR, combo_name)
          logging.info('Training combo %s', combo_name)
          combo_tuple = '(' + ','.join(map(str, combo)) + ')'
          train_operations.append(await experiment.add(
              xm.Job(
                  train_executable,
                  args={
                      'config.output_dir':
                          combo_dir,
                      'config.data.included_splits_idx':
                          combo_tuple,
                      'config.train_bias':
                          False,
                      'config.round_idx':
                          round_idx,
                      'config.ids_dir':
                          ids_dir,
                  },
                  executor=executor),
              identity=f'combo_{combo_name}_round_{round_idx}'))
        await asyncio.gather(
            *(op.wait_until_complete() for op in train_operations))

        # Calculate the bias table using model predictions.
        generate_bias_table_step = await experiment.add(
            xm.Job(
                generate_bias_table_executable,
                args={
                    'config.output_dir': round_dir,
                    'config.generate_bias_table': True,
                    'config.save_dir': round_dir,
                    'config.ids_dir': ids_dir,
                    'config.round_idx': round_idx,
                },
                executor=executor),
            identity=f'generate_bias_table_round_{round_idx}')
        await generate_bias_table_step.wait_until_complete()
        # Calculate the predictions table using model predictions.
        generate_predictions_table_step = await experiment.add(
            xm.Job(
                generate_bias_table_executable,
                args={
                    'config.output_dir': round_dir,
                    'config.generate_bias_table': False,
                    'config.save_dir': round_dir,
                    'config.ids_dir': ids_dir,
                },
                executor=executor),
            identity=f'generate_predictions_table_round_{round_idx}')
        await generate_predictions_table_step.wait_until_complete()

        # Get ids of examples to add in next round
        next_round_dir = os.path.join(output_dir, f'round_{round_idx+1}')
        next_round_ids_dir = os.path.join(next_round_dir, 'ids')
        sample_ids_step = await experiment.add(
            xm.Job(
                sampling_executable,
                args={
                    'config.output_dir': round_dir,
                    'config.ids_dir': next_round_ids_dir,
                },
                executor=executor),
            identity=f'sample_ids_round_{round_idx}')
        await sample_ids_step.wait_until_complete()
    experiment.add(
        run_train_rounds(config))


if __name__ == '__main__':
  app.run(main)
