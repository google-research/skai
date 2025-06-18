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

r"""XManager launcher for running SKAI training job.

Example command:

xmanager launch src/skai/model/xm_launch_single_model_vertex.py -- \
    --accelerator=TPU_V3 \
    --accelerator_count=8 \
    --config=src/skai/model/configs/skai_two_tower_config.py \
    --config.data.labeled_train_pattern=$TRAIN_EXAMPLES \
    --config.data.unlabeled_train_pattern=$TRAIN_EXAMPLES \
    --config.data.validation_pattern=$TEST_EXAMPLES \
    --config.output_dir=gs://skai-data/experiments/test_skai \
    --cloud_location='us-central1' \
    --experiment_name=test_skai
"""

import os

from absl import app
from absl import flags
from google.cloud import aiplatform_v1beta1 as aip
from ml_collections import config_flags
from skai.model import docker_instructions
from xmanager import xm
from xmanager import xm_local
from xmanager.cloud import vertex as xm_vertex
from xmanager.vizier import vizier_cloud

parameter_spec = aip.StudySpec.ParameterSpec


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
flags.DEFINE_integer(
    'ram',
    32,
    'Fixed amount of RAM for the work unit in GB',
)
flags.DEFINE_integer(
    'cpu',
    4,
    ('Number of vCPU instances to allocate. If left as None the default'
     ' value set by the cloud AI platform will be used.'
    ),
)
flags.DEFINE_string(
    'cloud_location', None, 'Google Cloud region to run jobs in.'
)
flags.DEFINE_enum(
    'accelerator',
    default=None,
    help='Accelerator to use for faster computations.',
    enum_values=['P100', 'V100', 'P4', 'T4', 'A100', 'TPU_V2', 'TPU_V3'],
)
flags.DEFINE_integer(
    'accelerator_count',
    1,
    (
        'Number of accelerator machines to use. Note that TPU_V2 and TPU_V3 '
        'only support count=8, see '
        'https://github.com/deepmind/xmanager/blob/main/docs/executors.md'
    ),
)
flags.DEFINE_string(
    'tpu', 'local', 'The BNS address of the first TPU worker (if using TPU).'
)
flags.DEFINE_bool(
    'build_docker_image',
    False,
    'If true, build a docker image from source. Otherwise, use a pre-built'
    ' docker image.',
)
flags.DEFINE_string('docker_image', None, 'Pre-built docker image to use.')
flags.DEFINE_bool(
    'use_tfds',
    True,
    'If True the dataset will be read using tensorflow_datasets.'
    'Otherwise, it will be read directly from tfrecord using tf.data API.',
)

config_flags.DEFINE_config_file('config')


def _get_default_docker_image(accelerator_type: str) -> str:
  return f'gcr.io/disaster-assessment/skai-ml-{accelerator_type}:latest'


def get_study_config() -> aip.StudySpec:
  """Get study configs for vizier."""

  return aip.StudySpec(
      parameters=[
          aip.StudySpec.ParameterSpec(
              parameter_id='config.optimizer.learning_rate',
              double_value_spec=parameter_spec.DoubleValueSpec(
                  min_value=1e-4, max_value=1e-2, default_value=1e-3
              ),
              scale_type=parameter_spec.ScaleType(
                  value=2  # 1 = UNIT_LINEAR_SCALE, 2 = UNIT_LOG_SCALE
              ),
          ),
          aip.StudySpec.ParameterSpec(
              parameter_id='config.optimizer.type',
              categorical_value_spec=parameter_spec.CategoricalValueSpec(
                  values=['adam', 'sgd']
              ),
          ),
          aip.StudySpec.ParameterSpec(
              parameter_id='config.model.l2_regularization_factor',
              double_value_spec=parameter_spec.DoubleValueSpec(
                  min_value=0.0, max_value=3.0, default_value=0.5
              ),
              # 1 for 'UNIT_LINEAR_SCALE')
              scale_type=parameter_spec.ScaleType(value=1),
          ),
      ],
      metrics=[
          aip.StudySpec.MetricSpec(
              metric_id='epoch_main_aucpr_1_vs_rest_val',
              goal=aip.StudySpec.MetricSpec.GoalType.MAXIMIZE,
          )
      ],
      decay_curve_stopping_spec=aip.StudySpec.DecayCurveAutomatedStoppingSpec(
          use_elapsed_duration=True
      ),
      measurement_selection_type=aip.StudySpec.MeasurementSelectionType(
          value=1,  # 1 = LAST_MEASUREMENT, 2 = BEST_MEASUREMENT
      ),
  )


def main(_) -> None:
  if FLAGS.cloud_location is None:
    raise ValueError('Google Cloud location must be set')
  xm_vertex.set_default_client(xm_vertex.Client(location=FLAGS.cloud_location))

  config = FLAGS.config
  config_path = config_flags.get_config_filename(FLAGS['config'])

  with xm_local.create_experiment(
      experiment_title=FLAGS.experiment_name
  ) as experiment:
    if FLAGS.accelerator is None:
      accelerator_type = 'cpu'
    elif FLAGS.accelerator in ['P100', 'V100', 'P4', 'T4', 'A100']:
      accelerator_type = 'gpu'
    elif FLAGS.accelerator in ['TPU_V3', 'TPU_V2']:
      accelerator_type = 'tpu'
      if FLAGS.accelerator_count != 8:
        raise ValueError(
            f'The accelerator {FLAGS.accelerator} only support 8 devices.'
        )
    else:
      raise ValueError(f'Unknown accelerator {FLAGS.accelerator}')

    if FLAGS.build_docker_image:
      [train_executable] = experiment.package([
          xm.Packageable(
              executable_spec=docker_instructions.get_xm_executable_spec(
                  FLAGS.accelerator
              ),
              executor_spec=xm_local.Vertex.Spec(),
          ),
      ])
    else:
      [train_executable] = experiment.package([
          xm.container(
              image_path=(
                  FLAGS.docker_image
                  or _get_default_docker_image(accelerator_type)
              ),
              executor_spec=xm_local.Vertex.Spec(),
          ),
      ])

    job_args = {
        'config': config_path,
        'is_vertex': True,
        'accelerator_type': accelerator_type,
        'tpu': FLAGS.tpu,
        'use_tfds': FLAGS.use_tfds,
        'config.output_dir': os.path.join(
            config.output_dir, str(experiment.experiment_id)
        ),
        'config.train_bias': config.train_bias,
        'config.train_stage_2_as_ensemble': False,
        'config.round_idx': 0,
        'config.data.initial_sample_proportion': 1.0,
        'config.data.batch_size': config.data.batch_size,
        'config.training.early_stopping': config.training.early_stopping,
        'config.model.load_pretrained_weights': (
            config.model.load_pretrained_weights
        ),
        'config.model.use_pytorch_style_resnet': (
            config.model.use_pytorch_style_resnet
        ),
    }

    if config.data.name == 'skai':
      job_args.update({
          'config.data.use_post_disaster_only': (
              config.data.use_post_disaster_only
          ),
          'config.data.tfds_dataset_name': config.data.tfds_dataset_name,
          'config.data.tfds_data_dir': config.data.tfds_data_dir,
          'config.data.adhoc_config_name': config.data.adhoc_config_name,
          'config.data.labeled_train_pattern': (
              config.data.labeled_train_pattern
          ),
          'config.data.validation_pattern': config.data.validation_pattern,
          'config.data.unlabeled_train_pattern': (
              config.data.unlabeled_train_pattern
          ),
      })

    job_args['config.training.save_model_checkpoints'] = False
    job_args['config.training.save_best_model'] = True
    job_args['config.training.num_epochs'] = config.training.num_epochs

    xm_args = xm.merge_args(['/skai/src/skai/model/train.py'], job_args)
    resources_args = {'RAM': FLAGS.ram * xm.GiB, 'CPU': FLAGS.cpu * xm.vCPU}
    if FLAGS.accelerator:
      resources_args[FLAGS.accelerator] = FLAGS.accelerator_count

    executor = xm_local.Vertex(
        requirements=xm.JobRequirements(
            service_tier=xm.ServiceTier.PROD,
            location=FLAGS.cloud_location,
            **resources_args,
        ),
    )

    if FLAGS.use_vizier:
      vizier_cloud.VizierExploration(
          experiment=experiment,
          job=xm.Job(
              executable=train_executable, executor=executor, args=xm_args
          ),
          study_factory=vizier_cloud.NewStudy(
              study_config=get_study_config(), location=FLAGS.cloud_location
          ),
          num_trials_total=100,
          num_parallel_trial_runs=3,
      ).launch()
    else:
      experiment.add(
          xm.Job(
              executable=train_executable,
              executor=executor,
              args=xm_args,
          )
      )


if __name__ == '__main__':
  app.run(main)
