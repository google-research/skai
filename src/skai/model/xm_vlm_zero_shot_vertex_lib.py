"""A library for launching a zero-shot ensemble experiment.

Works with xm_vlm_zero_shot_vertex.py as the launcher.
"""

import asyncio
import os
from typing import Optional

import pandas as pd
from skai.model import docker_instructions
import tensorflow as tf
from xmanager import xm
from xmanager import xm_local
from xmanager.contrib import parameter_controller


# A temporary directory created on the fly to store a requirements.txt file that
# specifies the dependencies of the auxiliary job when running the ensemble.
_CONTROLLER_TMP_DIR = '/tmp/skai/src'

_GPU_ACCELERATOR_TYPE = 'A100'
_GPU_ACCELERATOR_COUNT = 1

_TPU_ACCELERATOR_TYPE = 'TPU_V3'
_TPU_ACCELERATOR_COUNT = 8


def _ensemble_prediction_csvs(file_paths: list[str]) -> pd.DataFrame:
  """Reads model prediction files and provides an ensembled damage_score.

  The damage_score is the mean of the damage_score across all models.

  Args:
      file_paths: The paths to the CSV files containing the model predictions,
        assuming the SigLIP model output is the first one.

  Returns:
      A pandas DataFrame containing the ensembled data from all CSVs.
  """
  if not file_paths:
    raise ValueError(
        'At least one CSV file must be provided, specifically '
        'the SigLIP model output.'
    )

  siglip_df = pd.read_csv(tf.io.gfile.GFile(file_paths[0], 'r'))
  if (
      'is_cloudy' not in siglip_df.columns
      or 'example_id' not in siglip_df.columns
      or 'damage_score' not in siglip_df.columns
  ):
    raise ValueError(
        'First CSV in file_paths must be the SigLIP model output, '
        'which must contain is_cloudy, example_id, and '
        'damage_score columns.'
    )

  ensemble_df = siglip_df.drop_duplicates(subset=['example_id'])
  ensemble_df.rename(columns={'damage_score': 'damage_score_0'}, inplace=True)
  for i, file_path in enumerate(file_paths[1:]):
    df = pd.read_csv(tf.io.gfile.GFile(file_path, 'r'))
    if 'example_id' not in df.columns or 'damage_score' not in df.columns:
      raise ValueError(
          'Every CSV in file_paths must contain example_id and damage_score'
          f' columns, but {file_path} is missing one or more of them.'
      )
    df = df[['example_id', 'damage_score']].drop_duplicates(
        subset=['example_id']
    )
    if set(siglip_df['example_id']) != set(df['example_id']):
      raise ValueError(
          f'The CSV of {file_path} does not contain the same set of example_ids'
          f' as the SigLIP CSV at {file_paths[0]}.'
      )
    df.rename(columns={'damage_score': f'damage_score_{i+1}'}, inplace=True)
    ensemble_df = ensemble_df.merge(df, on=['example_id'], how='left')

  damage_score_cols = ensemble_df.filter(like='damage_score_').columns
  ensemble_df['damage_score'] = ensemble_df[damage_score_cols].mean(axis=1)
  ensemble_df['label'] = -1.0
  ensemble_df['damage'] = ensemble_df['damage_score'] > 0.5
  return ensemble_df


def _save_ensemble_csv(output_path: str, ensembled_df: pd.DataFrame) -> None:
  """Saves the ensembled CSV to the specified directory."""
  ensembled_df.to_csv(tf.io.gfile.GFile(output_path, 'w'))


def _get_docker_image_name(model_type: str) -> str:
  """Returns the image name for the training job based on model type.

  Args:
    model_type: The model type, e.g. 'siglip' or 'geofm'.

  Returns:
    The name of the Docker image to be used to run the training job.
  """
  # TODO(jlee24): For GeoFM, add TPU support when b/399193238 is resolved.
  if model_type == 'siglip':
    return 'siglip-gpu'
  elif model_type == 'geofm':
    return 'geofm-gpu'
  else:
    raise ValueError(f'Unsupported model type: {model_type}')


def _get_train_executable(
    experiment: xm.Experiment,
    model_type: str,
    build_docker_image: bool,
    pre_built_docker_image: str,
) -> list[xm.Executable]:
  """Returns the executable for a training job based on model type.

  Args:
    experiment: The XManager experiment object.
    model_type: The model type, e.g. 'siglip' or 'geofm'.
    build_docker_image: Whether to build a docker image from source.
    pre_built_docker_image: The path to the pre-built docker image. Only
      used if build_docker_image is False.

  Returns:
    The executable for the training job.
  """
  if build_docker_image:
    [train_executable] = experiment.package([
        xm.Packageable(
            executable_spec=docker_instructions.get_xm_executable_spec(
                _get_docker_image_name(model_type)
            ),
            executor_spec=xm_local.Vertex.Spec(),
        ),
    ])
  else:
    [train_executable] = experiment.package([
        xm.container(
            image_path=pre_built_docker_image,
            executor_spec=xm_local.Vertex.Spec(),
        ),
    ])
  return [train_executable]


def _get_executor(
    docker_image_name: str,
    cloud_location: str,
    num_cpu: int,
    num_ram: int,
) -> xm.Executor:
  """Returns the executor for the training job based on model type.

  Args:
    docker_image_name: The name of the Docker image to be used to run the
      training job, e.g. 'siglip-tpu' or 'geofm-gpu'.
    cloud_location: The cloud location to run the job in.
    num_cpu: The number of CPUs to use for the job.
    num_ram: The number of RAM to use for the job.

  Returns:
    The XM executor for the training job, specifying the required resources.
  """
  job_kwargs = {
      'service_tier': xm.ServiceTier.PROD,
      'location': cloud_location,
      'cpu': num_cpu * xm.vCPU,
      'ram': num_ram * xm.GiB,
  }
  if '-gpu' in docker_image_name:
    job_kwargs[_GPU_ACCELERATOR_TYPE] = _GPU_ACCELERATOR_COUNT
  elif '-tpu' in docker_image_name:
    job_kwargs[_TPU_ACCELERATOR_TYPE] = _TPU_ACCELERATOR_COUNT
  return xm_local.Vertex(
      requirements=xm.JobRequirements(**job_kwargs),
  )


def get_experiment_name(
    dataset_names: list[str],
    model_type: str,
    siglip_model_variant: str,
    image_size: int,
) -> str:
  """Returns the experiment name for the training job."""
  experiment_name = []
  experiment_name.append(model_type)
  if model_type == 'siglip':
    experiment_name.append(siglip_model_variant)
    experiment_name.append(str(image_size))
  if dataset_names:
    experiment_name.extend(dataset_names)
  return '_'.join(experiment_name)


def build_experiment_jobs(
    experiment: xm.Experiment,
    model_type: str,
    cloud_location: str,
    cloud_bucket_name: str,
    cloud_project: str,
    output_dir: str,
    example_patterns: list[str],
    dataset_names: list[str],
    build_docker_image: bool,
    image_feature: str,
    num_ram: int,
    num_cpu: int,
    # SigLIP args.
    use_siglip2: bool,
    siglip_model_variant: str,
    image_size: int,
    negative_labels_filepath: str,
    positive_labels_filepath: str,
    cloud_labels_filepath: str,
    nocloud_labels_filepath: str,
    batch_size: int,
    siglip_docker_image: str,
    # GeoFM args.
    geofm_savedmodel_path: str,
    geofm_docker_image: str,
    output_ensemble_csv_file_name: Optional[str] = None,
) -> None:
  """Launches SigLIP and GeoFM inference jobs in parallel.

  Args:
    experiment: The XManager experiment object.
    model_type: The model type, e.g. 'siglip', 'geofm', or 'ensemble'.
    cloud_location: The cloud location to run the job in.
    cloud_bucket_name: The cloud bucket name to use for the job.
    cloud_project: The cloud project to use for the job.
    output_dir: The output directory to save the results to.
    example_patterns: The list of example patterns to use for the training job.
    dataset_names: The list of dataset names to use for the training job.
    build_docker_image: Whether to build a docker image from source.
    image_feature: The image feature to use for the training job.
    num_ram: The amount of RAM to use for the training job.
    num_cpu: The number of CPUs to use for the training job.
    use_siglip2: Whether to use SigLIP2. If false, use old SigLIP.
    siglip_model_variant: The model name to use for SigLIP.
    image_size: The image size to use for SigLIP.
    negative_labels_filepath: The path to the negative labels file.
    positive_labels_filepath: The path to the positive labels file.
    cloud_labels_filepath: The path to the cloud labels file.
    nocloud_labels_filepath: The path to the nocloud labels file.
    batch_size: The batch size to use for the SigLIP training job.
    siglip_docker_image: The path to the SigLIP docker image.
    geofm_savedmodel_path: The path to the GeoFM savedmodel.
    geofm_docker_image: The path to the GeoFM docker image.
    output_ensemble_csv_file_name: The name of the ensembled predictions CSV
      file that will be saved in the output_dir.

  Returns:
    The XManager experiment object.
  """
  xm_jobs = {}
  job_args = {
      'example_patterns': ','.join(example_patterns),
      'output_dir': output_dir,
      'image_feature': image_feature,
  }
  if dataset_names:
    job_args['dataset_names'] = ','.join(dataset_names)
  if model_type in ['siglip', 'ensemble']:
    siglip_operand = ['/skai/src/skai/model/vlm_zero_shot_vertex.py']
    siglip_job_args = job_args | {
        'use_siglip2': use_siglip2,
        'siglip_model_variant': siglip_model_variant,
        'image_size': image_size,
        'negative_labels_filepath': negative_labels_filepath,
        'positive_labels_filepath': positive_labels_filepath,
        'cloud_labels_filepath': cloud_labels_filepath,
        'nocloud_labels_filepath': nocloud_labels_filepath,
        'batch_size': batch_size
    }
    siglip_job_args = xm.merge_args(siglip_operand, siglip_job_args)
    [siglip_executable] = _get_train_executable(
        experiment, 'siglip', build_docker_image, siglip_docker_image
    )
    siglip_executor = _get_executor(
        siglip_docker_image, cloud_location, num_cpu, num_ram
    )
    siglip_job = xm.Job(
        executable=siglip_executable,
        executor=siglip_executor,
        args=siglip_job_args,
        name='inference_siglip')
    xm_jobs['siglip'] = siglip_job

  if model_type in ['geofm', 'ensemble']:
    geofm_operand = ['/skai/src/skai/model/geofm_zero_shot_vertex.py']
    geofm_job_args = job_args | {
        'geofm_savedmodel_path': geofm_savedmodel_path,
    }
    geofm_job_args = xm.merge_args(geofm_operand, geofm_job_args)
    [geofm_executable] = _get_train_executable(
        experiment, 'geofm', build_docker_image, geofm_docker_image
    )
    geofm_executor = _get_executor(
        geofm_docker_image, cloud_location, num_cpu, num_ram
    )
    geofm_job = xm.Job(
        executable=geofm_executable,
        executor=geofm_executor,
        args=geofm_job_args,
        name='inference_geofm')
    xm_jobs['geofm'] = geofm_job

  if model_type in ['siglip', 'geofm']:
    experiment.add(job=xm_jobs[model_type])

  elif model_type == 'ensemble':
    # Create a temporary requirements.txt file for the auxiliary job. Must be
    # co-located with launcher and src.
    requirements_file_path = os.path.join(
        _CONTROLLER_TMP_DIR, 'requirements.txt'
    )
    with open(requirements_file_path, 'w') as f:
      f.write('cloudpickle\n')
      f.write('gcsfs\n')
      f.write('pandas\n')
      f.write('tensorflow\n')
    @parameter_controller.controller(
        executor=xm_local.Vertex(),
        controller_args={
            'service_tier': xm.ServiceTier.PROD,
            'location': cloud_location,
        },
        controller_env_vars={
            'GOOGLE_CLOUD_BUCKET_NAME': cloud_bucket_name,
            'GOOGLE_CLOUD_LOCATION': cloud_location,
            'GOOGLE_CLOUD_PROJECT': cloud_project,
        },
        package_path=_CONTROLLER_TMP_DIR,
        use_host_db_config=False,
    )
    async def run_ensemble(experiment: xm.Experiment):
      xm_local.Vertex()  # Initialize XM Vertex again in the remote job.
      operations = []
      for job in xm_jobs.values():
        operations.append(await experiment.add(job))
      await asyncio.gather(*(op.wait_until_complete() for op in operations))

      # Combine the ensemble outputs.
      for dataset_name in dataset_names:
        ensemble_df = _ensemble_prediction_csvs([
            f'{output_dir}/{dataset_name}_output.csv',
            f'{output_dir}/{dataset_name}_geofm_output.csv',
        ])
        _save_ensemble_csv(
            os.path.join(output_dir, output_ensemble_csv_file_name),
            ensemble_df,
        )

    experiment.add(run_ensemble())

  else:
    raise ValueError(f'Unknown model type {model_type}')
