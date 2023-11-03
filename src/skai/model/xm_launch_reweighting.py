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

import os

from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
from xmanager import xm
from xmanager import xm_abc


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
          f'Final Model Sweep {config.data.name}_{config.model.name}'
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

    [train_executable] = experiment.package([
        xm.bazel_binary(
            label='//third_party/py/skai/model:train',
            dependencies=[config_resource],
            bazel_args=xm_abc.bazel_args.gpu(),
            executor_spec=xm_abc.Borg.Spec(),
            args={
                'config': config_resource.get_path(
                    config_filename, xm_abc.Borg.Spec()
                ),
            },
        ),
    ])
    # Sweep over different values of lambda for reweighting.
    job_args = {
        'config.round_idx': config.round_idx,
        'config.train_stage_2_as_ensemble': config.train_stage_2_as_ensemble,
        'config.ids_dir': config.ids_dir,
        'config.path_to_existing_bias_table': (
            config.path_to_existing_bias_table
        ),
        'config.reweighting.do_reweighting': True,
        'config.reweighting.signal': config.reweighting.signal,
        'config.training.early_stopping': config.training.early_stopping,
        'config.training.num_epochs': config.training.num_epochs,
        'config.optimizer.learning_rate': config.optimizer.learning_rate,
    }
    if (
        config.reweighting.signal == 'bias'
        or config.reweighting.signal == 'subgroup_label'
    ):
      for reweighting_lambda in np.linspace(0.0, 1.0, num=11):
        job_args.update({
            'config.output_dir': os.path.join(
                config.output_dir,
                'reweighting_lambda_' + str(reweighting_lambda),
            ),
            'config.reweighting.lambda_value': reweighting_lambda,
        })
        experiment.add(
            xm.Job(train_executable, args=job_args, executor=executor)
        )
    elif config.reweighting.signal == 'error':  # Uses prediction error.
      for reweighting_lambda in np.linspace(0.0, 1.0, num=11):
        for error_percentile_threshold in np.linspace(0.0, 1.0, num=5):
          job_args.update({
              'config.output_dir': os.path.join(
                  config.output_dir,
                  'reweighting_lambda_' + str(reweighting_lambda),
                  'error_percentile_' + str(error_percentile_threshold),
              ),
              'config.reweighting.lambda_value': reweighting_lambda,
              'config.reweighting.error_percentile_threshold': (
                  error_percentile_threshold
              ),
          })
          experiment.add(
              xm.Job(train_executable, args=job_args, executor=executor)
          )


if __name__ == '__main__':
  app.run(main)
