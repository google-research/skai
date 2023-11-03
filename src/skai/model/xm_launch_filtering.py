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

This is a script to use the filtering functionality for greating subbsets.
Currently we only support training for 1 round.
# TODO(jlee24): We should merge this into xm_launch. Need to adopt
# generate_bias_table_lib.get_example_id_to_bias_label_table to work with
# filtered subsets.
Usage:
# pylint: disable=line-too-long

Examples of training only the main classification task (no bias):
  ml_python3 third_party/py/skai/model/xm_launch_filtering.py \
    --adhoc_import_modules=skai \
      -- \
      --xm_deployment_env=alphabet \
      --xm_resource_pool="mnl" \
      --xm_resource_alloc="group:mnl/mnl-shared-ml-user" \
      --config=third_party/py/skai/model/configs/waterbirds_resnet_config.py \
      --config.output_dir=/cns/path/test \
      --config.train_bias=False

# pylint: enable=line-too-long
"""

import asyncio
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
      files={
          os.path.join('//', config_path): config_filename
      })

  with xm_abc.create_experiment(
      experiment_title=(
          'Introspective  Training'
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

    [train_executable, prediction_executable
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
    ])

    @parameter_controller.controller(interpreter=xm_abc.ml_python())
    async def run_train_rounds(
        experiment: xm.Experiment,
        config: ml_collections.ConfigDict
    ) -> None:

      # Currently the script only supports running 1 round.
      num_rounds = 1
      ids_dir = ''

      output_dir = os.path.join(
          config.output_dir, config.active_sampling.sampling_score)

      for round_idx in range(num_rounds):
        logging.info('Running Round %d of Active Learning.', round_idx)
        round_dir = os.path.join(output_dir, f'round_{round_idx}')
        ids_dir = os.path.join(round_dir, 'ids')
        tf.io.gfile.makedirs(round_dir)

        if config.train_bias and not config.path_to_existing_bias_table:
          logging.info('Training on different splits to calculate bias...')
          # Launch training on multiple datasets in parallel.
          # TODO(jlee24): Decompose to its own function.
          train_operations = []
          for split_id in range(config.data.num_splits):
            split_name = str(split_id)
            config.data.split_id = split_id
            combo_dir = os.path.join(round_dir, COMBOS_SUBDIR, split_name)
            logging.info('Training combo %s', split_name)
            train_operations.append(await experiment.add(
                xm.Job(
                    train_executable,
                    args={
                        'config.output_dir':
                            combo_dir,
                        'config.train_stage_2_as_ensemble':
                            config.train_stage_2_as_ensemble,
                        'config.data.split_id':
                            split_id,
                        'config.data.split_proportion':
                            config.data.split_proportion,
                        'config.data.initial_sample_seed':
                            config.data.initial_sample_seed,
                        'config.data.split_seed':
                            config.data.split_seed,
                        'config.training.num_epochs':
                            config.training.num_epochs,
                        'config.data.use_splits':
                            False,
                        'config.train_bias':
                            False,
                        'config.round_idx':
                            round_idx,
                        'config.ids_dir':
                            ids_dir,
                    },
                    executor=executor),
                identity=f'split_{split_name}_round_{round_idx}'))
          await asyncio.gather(
              *(op.wait_until_complete() for op in train_operations))
        prediction_operations = []
        job_args = {
            'config.output_dir': round_dir,
            'config.save_dir': round_dir,
            'config.generate_bias_table': False,
        }
        prediction_operations.append(await experiment.add(
            xm.Job(prediction_executable, args=job_args, executor=executor),
            identity='Calculate predictions'))
        await asyncio.gather(
            *(op.wait_until_complete() for op in prediction_operations))
    experiment.add(
        run_train_rounds(config))


if __name__ == '__main__':
  app.run(main)
