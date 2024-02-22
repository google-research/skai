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

r"""XM Launcher."""

import os

from absl import app
from absl import flags
from docker_instructions import get_docker_instructions
from google.cloud import aiplatform_v1beta1 as aip
from ml_collections import config_flags
from xmanager import xm
from xmanager import xm_local
from xmanager.vizier import vizier_cloud


parameter_spec = aip.StudySpec.ParameterSpec

'''
xmanager launch src/skai/model/xm_launch_single_model_vertex.py -- \
    --xm_wrap_late_bindings \
    --xm_upgrade_db=True \
    --config=src/skai/model/configs/skai_config.py \
    --config.data.tfds_dataset_name=skai_dataset \
    --config.data.tfds_data_dir=gs://skai-data/hurricane_ian \
    --config.output_dir=gs://skai-data/experiments/test_skai \
    --cloud_location='us-central1' \
    --experiment_name=test_skai \
    --project_path=~/path/to/skai \
    --accelerator=V100 \
    --accelerator_count=1
'''


FLAGS = flags.FLAGS
flags.DEFINE_string('project_path', '.', 'Path to project')
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
    'cloud_location',
    None,
    'Google Cloud location (region) for vizier jobs'
)
flags.DEFINE_enum(
    'accelerator',
    default=None,
    help='Accelerator to use for faster computations.',
    enum_values=['P100', 'V100', 'P4', 'T4', 'A100', 'TPU_V2', 'TPU_V3']
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
config_flags.DEFINE_config_file('config')


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
  config = FLAGS.config
  config_path = config_flags.get_config_filename(FLAGS['config'])

  with xm_local.create_experiment(
      experiment_title=(
          f'{FLAGS.experiment_name} {config.data.name}_{config.model.name}'
      )
  ) as experiment:
    base_image, docker_instructions = get_docker_instructions(FLAGS.accelerator)

    executable_spec = xm.PythonContainer(
        # Package the current directory that this script is in.
        path=os.path.expanduser(FLAGS.project_path),
        base_image=base_image,
        docker_instructions=docker_instructions,
        entrypoint=xm.CommandList([
            'pip install /skai/src/.',
            'python /skai/src/skai/model/train.py $@',
        ]),
        use_deep_module=True,
    )
    if FLAGS.accelerator is not None:
      if FLAGS.accelerator in ['TPU_V3', 'TPU_V2']:
        if FLAGS.accelerator_count != 8:
          raise ValueError(
              f'The accelerator {FLAGS.accelerator} only support 8 devices.'
          )
        accelerator_type = 'tpu'
      else:
        accelerator_type = 'gpu'
      resources_args = {
          FLAGS.accelerator: FLAGS.accelerator_count,
          'RAM': FLAGS.ram * xm.GiB,
          'CPU': FLAGS.cpu * xm.vCPU,
      }
    else:
      resources_args = {'RAM': FLAGS.ram * xm.GiB, 'CPU': FLAGS.cpu * xm.vCPU}
      accelerator_type = 'cpu'

    executor = xm_local.Vertex(
        requirements=xm.JobRequirements(
            service_tier=xm.ServiceTier.PROD, **resources_args
        ),
    )

    [train_executable] = experiment.package([
        xm.Packageable(
            executable_spec=executable_spec,
            executor_spec=xm_local.Vertex.Spec(),
            args={
                'config': config_path,
                'is_vertex': True,
                'accelerator_type': accelerator_type,
                'tpu': FLAGS.tpu
            },
        ),
    ])

    job_args = {
        'config.output_dir': os.path.join(config.output_dir,
                                          str(experiment.experiment_id)),
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

    if FLAGS.cloud_location is None:
      raise ValueError('Google Cloud location is either None or invalid.')

    if FLAGS.use_vizier:
      vizier_cloud.VizierExploration(
          experiment=experiment,
          job=xm.Job(
              executable=train_executable, executor=executor, args=job_args
          ),
          study_factory=vizier_cloud.NewStudy(
              study_config=get_study_config(),
              location=FLAGS.cloud_location
          ),

          num_trials_total=100,
          num_parallel_trial_runs=3,
      ).launch()
    else:
      experiment.add(
          xm.Job(
              executable=train_executable,
              executor=executor,
              args=job_args,
          )
      )


if __name__ == '__main__':
  app.run(main)
