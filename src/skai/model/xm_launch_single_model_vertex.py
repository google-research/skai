r"""XM Launcher.


# pylint: enable=line-too-long
"""

import os

from absl import app
from absl import flags
from ml_collections import config_flags
from xmanager import xm, xm_local
from google.cloud import aiplatform_v1beta1 as aip
from xmanager.vizier import vizier_cloud
from xm_utils import get_docker_instructions

parameter_spec = aip.StudySpec.ParameterSpec

"""
xmanager launch src/skai/model/xm_launch_single_model_vertex.py -- \
    --xm_wrap_late_bindings \
    --xm_upgrade_db=True \
    --config=src/skai/model/configs/skai_config.py \
    --config.data.tfds_dataset_name=skai_dataset \
    --config.data.tfds_data_dir=gs://skai-project/hurricane_ian \
    --config.output_dir=gs://skai-project/experiments/test_skai \
    --experiment_name=test_skai \
    --project_path=~/path/to/skai
"""


FLAGS = flags.FLAGS
flags.DEFINE_string('project_path', '.', 'Path to project')
flags.DEFINE_string('experiment_name', '', 'Label for XManager experiment to '
                    'make it easier to find.')
flags.DEFINE_bool(
    'use_vizier', False, 'Finds the best hyperparameters using Vizier.')
flags.DEFINE_bool(
    'train_as_ensemble', False, 'Trains an ensemble of single '
    'models, as we would for Stage 1 in Introspective Self-Play.')
flags.DEFINE_bool(
    'eval_only', False, 'Only runs evaluation, no training.')
flags.DEFINE_list(
    'optimizer_types',
    [],
    'List of optimizers to try in a hyperparameter sweep. For example, '
    '["adam", "sgd"].'
)
flags.DEFINE_list(
    'lr_tune_values', [],
    'List of learning rate values to try in a hyperparameter sweep.')
flags.DEFINE_list(
    'l2_reg_tune_values', [],
    'List of L2 regularization factors to try in a hyperparameter sweep.')
flags.DEFINE_list(
    'num_epoch_values',
    [],
    'List of number of epochs for training duration to try in a '
    'hyperparameter sweep.'
)
config_flags.DEFINE_config_file('config')


def get_study_config() -> aip.StudySpec:
    #   algorithm = UNKNOWN, # UNKNOWN Algorithm uses Bayesian Optimization
    return aip.StudySpec(
        parameters=[
            aip.StudySpec.ParameterSpec(
                parameter_id="config.optimizer.learning_rate",
                double_value_spec=parameter_spec.DoubleValueSpec(
                    min_value=1e-6,
                    max_value=1e-2,
                    default_value=1e-3
                ),
                # TODO: Check for availability of LINEAR instead of UNIT_LINEAR
                # value=2 for "UNIT_LOG_SCALE")
                scale_type=parameter_spec.ScaleType(value=2)
            ),
            aip.StudySpec.ParameterSpec(
                parameter_id="config.optimizer.type",
                categorical_value_spec=parameter_spec.CategoricalValueSpec(
                    values=["adam", "sgd"]
                )
            ),
            aip.StudySpec.ParameterSpec(
                parameter_id="config.model.l2_regularization_factor",
                double_value_spec=parameter_spec.DoubleValueSpec(
                    min_value=0.,
                    max_value=3.0,
                    default_value=0.5
                ),
                # 1 for "UNIT_LINEAR_SCALE")
                scale_type=parameter_spec.ScaleType(value=1)
            )
        ],
        metrics=[
            aip.StudySpec.MetricSpec(
                metric_id='epoch_main_aucpr_1_vs_rest_val',
                goal=aip.StudySpec.MetricSpec.GoalType.MAXIMIZE
            )
        ],
        decay_curve_stopping_spec=aip.StudySpec.DecayCurveAutomatedStoppingSpec(
            use_elapsed_duration=True
        ),
        measurement_selection_type=aip.StudySpec.MeasurementSelectionType(
            value=1,  # value = 1 for LAST_MEASUREMENT, value=2 for "BEST_MEASUREMENT"
        ),
    )


def main(_) -> None:

  config = FLAGS.config
  config_path = config_flags.get_config_filename(FLAGS['config'])

  if not FLAGS.optimizer_types:
    FLAGS.optimizer_types.append(config.optimizer.type)
  if not FLAGS.lr_tune_values:
    FLAGS.lr_tune_values.append(config.optimizer.learning_rate)
  if not FLAGS.l2_reg_tune_values:
    FLAGS.l2_reg_tune_values.append(config.model.l2_regularization_factor)
  if not FLAGS.num_epoch_values:
    FLAGS.num_epoch_values.append(config.training.num_epochs)

  with xm_local.create_experiment(
      experiment_title=(
          f'{FLAGS.experiment_name} {config.data.name}_{config.model.name}'
      )
  ) as experiment:

    executable_spec = xm.PythonContainer(
        # Package the current directory that this script is in.
        path=os.path.expanduser(FLAGS.project_path),
        base_image='gcr.io/deeplearning-platform-release/base-gpu',
        docker_instructions=get_docker_instructions(),
        entrypoint=xm.CommandList([
            "pip install /skai/src/.",
            "python /skai/src/skai/model/train.py $@"
        ]),
        use_deep_module=True,
    )

    executor = xm_local.Vertex(
        requirements=xm.JobRequirements(
            location='li',
            service_tier=xm.ServiceTier.PROD,
            cpu=16,
            ram=64 * xm.GiB,
            # tmp_ram_fs=64 * xm.GiB,
            # v100=2
        ),
    )

    [train_executable] = experiment.package([
        xm.Packageable(
            executable_spec=executable_spec,
            executor_spec=xm_local.Vertex.Spec(),
            args={"config": config_path}
        ),
    ])

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
    }

    if config.data.name == 'skai':
      job_args.update(
          {
              'config.data.use_post_disaster_only': (
                  config.data.use_post_disaster_only
              ),
              'config.data.tfds_dataset_name': (
                  config.data.tfds_dataset_name
              ),
              'config.data.tfds_data_dir': (
                  config.data.tfds_data_dir
              ),
          }
      )

    # if FLAGS.use_vizier:  # Tune hyperparameters with Vizier.
    job_args['config.training.save_model_checkpoints'] = False
    job_args['config.training.save_best_model'] = True

    # if FLAGS.use_vizier:
    vizier_cloud.VizierExploration(
        experiment=experiment,
        job=xm.Job(
            executable=train_executable,
            executor=executor,
            args=job_args
        ),
        study_factory=vizier_cloud.NewStudy(
            study_config=get_study_config()),
        num_trials_total=3,
        num_parallel_trial_runs=2,
    ).launch()


if __name__ == '__main__':
  app.run(main)