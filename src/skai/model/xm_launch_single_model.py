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
import logging
import os
from typing import Any, Iterable

from absl import app
from absl import flags
import ml_collections
from ml_collections import config_flags
from xmanager import xm
from xmanager import xm_abc
from xmanager.vizier import vizier_abc



FLAGS = flags.FLAGS
flags.DEFINE_string(
    'experiment_name',
    '',
    'Label for XManager experiment to make it easier to find.',
)
flags.DEFINE_bool(
    'use_vizier', False, 'Finds the best hyperparameters using Vizier.'
)
flags.DEFINE_bool(
    'train_as_ensemble',
    False,
    'Trains an ensemble of single '
    'models, as we would for Stage 1 in Introspective Self-Play.',
)
flags.DEFINE_bool('eval_only', False, 'Only runs evaluation, no training.')
flags.DEFINE_list(
    'optimizer_types',
    [],
    'List of optimizers to try in a hyperparameter sweep. For example, '
    '["adam", "sgd"].',
)
flags.DEFINE_list(
    'lr_tune_values',
    [],
    'List of learning rate values to try in a hyperparameter sweep.',
)
flags.DEFINE_list(
    'l2_reg_tune_values',
    [],
    'List of L2 regularization factors to try in a hyperparameter sweep.',
)
flags.DEFINE_list(
    'num_epoch_values',
    [],
    'List of number of epochs for training duration to try in a '
    'hyperparameter sweep.',
)
_ACCELERATOR = flags.DEFINE_string(
    'accelerator', 'tpu', 'Accelerator type for main job: tpu or gpu.'
)
_TPU_TOPOLOGY = flags.DEFINE_string('tpu_topology', '1x1', 'TPU topology.')
_TPU_PLATFORM = flags.DEFINE_string(
    'tpu_platform',
    'dragonfish',
    'TPU platform, such as dragonfish, dragondonut, jellyfish, jellydonut,'
    ' pufferfish, and puffylite.',
)
_NUM_GPUS = flags.DEFINE_integer('num_gpus', 8, 'Number of gpus.')
_GPU_TYPES = flags.DEFINE_string(
    'gpu_types', 'v100', 'GPU type (only for GPU worker).'
)
config_flags.DEFINE_config_file('config')




def _sweep(
    hyperparameter_name: str,
    values: Iterable[Any]
) -> Iterable[tuple[str, Any]]:
  for value in values:
    yield (hyperparameter_name, value)


def _product(
    hyperparams: list[Iterable[tuple[str, Any]]]
) -> Iterable[dict[str, Any]]:
  for combo in itertools.product(*hyperparams):
    yield dict(combo)


def main(_) -> None:
  config = FLAGS.config

  if not FLAGS.optimizer_types:
    FLAGS.optimizer_types.append(config.optimizer.type)
  if not FLAGS.lr_tune_values:
    FLAGS.lr_tune_values.append(config.optimizer.learning_rate)
  if not FLAGS.l2_reg_tune_values:
    FLAGS.l2_reg_tune_values.append(config.model.l2_regularization_factor)
  if not FLAGS.num_epoch_values:
    FLAGS.num_epoch_values.append(config.training.num_epochs)

  with xm_abc.create_experiment(
      experiment_title=(
          f'{FLAGS.experiment_name} {config.data.name}_{config.model.name}'
      ),
      attribution_urls=['rh/efforts/1910'],
  ) as experiment:
    if _ACCELERATOR.value == 'gpu':
      builder = _mirrored_strategy_builder(experiment)
    elif _ACCELERATOR.value == 'tpu':
      builder = _tpu_strategy_builder(experiment)
    else:
      raise EnvironmentError(
          f'Unsupported accelerator type: {_ACCELERATOR.value}')
    job_args = {
        'config.output_dir': config.output_dir,
        'config.train_bias': config.train_bias,
        'config.train_stage_2_as_ensemble': False,
        'config.round_idx': 0,
        'config.data.initial_sample_proportion': 1.0,
        'config.training.early_stopping': config.training.early_stopping,
        'config.model.load_pretrained_weights': (
            config.model.load_pretrained_weights
        ),
        'config.model.use_pytorch_style_resnet': (
            config.model.use_pytorch_style_resnet
        ),
        'config.data.labeled_train_pattern': (
            config.data.labeled_train_pattern
        ),
        'config.data.unlabeled_train_pattern': (
            config.data.unlabeled_train_pattern
        ),
        'config.data.validation_pattern': config.data.validation_pattern,
    }
    if config.data.name == 'skai':
      job_args.update({
          'config.data.use_post_disaster_only': (
              config.data.use_post_disaster_only
          ),
          'config.data.tfds_dataset_name': config.data.tfds_dataset_name,
          'config.data.tfds_data_dir': config.data.tfds_data_dir,
      })

    if FLAGS.use_vizier:  # Tune hyperparameters with Vizier.
      job_args['config.training.save_model_checkpoints'] = False
      job_args['config.training.save_best_model'] = True
      study_factory = vizier_abc.NewStudy(get_study_config())
      async def gen_work_unit(work_unit: xm.WorkUnit, **hparams):
        job_args.update(**hparams)
        job = builder.create_job_group(work_unit, job_args)
        work_unit.add(job)
      experiment.add(
          vizier_abc.vizier_controller(
              gen_work_unit,
              study_factory,
              num_parallel_work_units=10,
              adhoc_import_modules=[
                  'google3.learning.deepmind.xmanager2.contrib.xm_sync'
              ],
          ),
      )
    else:
      async def run_train(
          experiment: xm.Experiment,
          config: ml_collections.ConfigDict,
          train_as_ensemble: bool,
          job_args: dict[str, Any],
          optimizer_types: list[str],
          lr_tune_values: list[float],
          l2_reg_tune_values: list[float],
          num_epoch_values: list[float],
      ) -> None:
        sweep = _product([
            _sweep('config.optimizer.type', optimizer_types),
            _sweep('config.optimizer.learning_rate', lr_tune_values),
            _sweep('config.model.l2_regularization_factor', l2_reg_tune_values),
            _sweep('config.training.num_epochs', num_epoch_values)
        ])
        for args in sweep:
          output_dir = config.output_dir
          for hyperparam in sorted(args.keys()):
            hyperparam_name = hyperparam.split('.')[-1]
            output_dir = os.path.join(
                output_dir, f'{hyperparam_name}_{args[hyperparam]}'
            )
          job_args['config.output_dir'] = output_dir
          job_args.update(args)

          if train_as_ensemble:
            num_splits = config.data.num_splits
            num_ood_splits = int(num_splits * config.data.ood_ratio)
            num_id_splits = num_splits - num_ood_splits
            train_combos = [
                list(c) for c in list(
                    itertools.combinations(range(num_splits), num_id_splits))
            ]
            two_head_ensemble_dir = os.path.join(config.output_dir, 'ensemble')
            train_ensemble_operations = []
            for combo in train_combos:
              combo_name = '_'.join(map(str, combo))
              combo_dir = os.path.join(two_head_ensemble_dir, combo_name)
              logging.info('Training two-headed model on combo %s', combo_name)
              combo_tuple = '(' + ','.join(map(str, combo)) + ')'
              job_args['config.output_dir'] = combo_dir
              job_args['config.data.included_splits_idx'] = combo_tuple
              train_ensemble_operations.append(
                  await experiment.add(
                      builder.gen_job_group(),
                      job_args,
                      identity=f'two_head_train_{combo_name}',
                  )
              )
            await asyncio.gather(
                *(op.wait_until_complete() for op in train_ensemble_operations)
            )
          else:
            experiment.add(builder.gen_job_group(), job_args)

      experiment.add(
          run_train(
              config,
              FLAGS.train_as_ensemble,
              job_args,
              FLAGS.optimizer_types,
              FLAGS.lr_tune_values,
              FLAGS.l2_reg_tune_values,
              FLAGS.num_epoch_values,
          )
      )


if __name__ == '__main__':
  app.run(main)
