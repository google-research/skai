"""A library for launching a zero-shot ensemble experiment.

Works with xm_vlm_zero_shot_vertex.py as the launcher.
"""

import asyncio
import os
import tempfile
from typing import Optional
from skai.model import docker_instructions
from xmanager import xm
from xmanager import xm_local
from xmanager.contrib import parameter_controller


# A temporary directory created on the fly to store a requirements.txt file that
# specifies the dependencies of the auxiliary job when running the ensemble.
_CONTROLLER_TMP_DIR = '/tmp/skai/src'
_SIGLIP_ACCELERATOR_TYPE = 'TPU_V3'
_ACCELERATOR_COUNT = 8


def _get_docker_image_name(
    model_type: str, geofm_accelerator_type: Optional[str] = None
) -> str:
  """Returns the image name for the training job based on model type.

  Args:
    model_type: The model type, e.g. 'siglip' or 'geofm'.
    geofm_accelerator_type: The type of accelerator to use for GeoFM, e.g.
      'A100' or 'CPU'.

  Returns:
    The name of the Docker image to be used to run the training job.
  """
  # TODO(jlee24): For GeoFM, add TPU support when b/399193238 is resolved.
  if model_type == 'siglip':
    docker_image_name = 'siglip-tpu'
  elif model_type == 'geofm':
    docker_image_name = (
        'geofm-gpu' if geofm_accelerator_type == 'A100' else 'geofm-cpu'
    )
  else:
    raise ValueError(f'Unsupported model type: {model_type}')
  return docker_image_name


def _get_train_executable(
    experiment: xm.Experiment, model_type: str, build_docker_image: bool,
    docker_image_name: str,
    siglip_docker_image: Optional[str] = None,
    geofm_docker_image: Optional[str] = None,
) -> list[xm.Executable]:
  """Returns the executable for a training job based on model type.

  Args:
    experiment: The XManager experiment object.
    model_type: The model type, e.g. 'siglip' or 'geofm'.
    build_docker_image: Whether to build a docker image from source.
    docker_image_name: The name of the Docker image to be built. Only used if
      building from source, e.g. 'siglip-tpu' or 'geofm-gpu'.
    siglip_docker_image: The path to the pre-built SigLIP docker image. Only
      used if build_docker_image is False.
    geofm_docker_image: The path to the GeoFM docker image. Only used if
      build_docker_image is False.

  Returns:
    The executable for the training job.
  """
  if build_docker_image:
    [train_executable] = experiment.package([
        xm.Packageable(
            executable_spec=docker_instructions.get_xm_executable_spec(
                docker_image_name
            ),
            executor_spec=xm_local.Vertex.Spec(),
        ),
    ])
  else:
    docker_image = (
        siglip_docker_image if model_type == 'siglip' else geofm_docker_image
    )
    [train_executable] = experiment.package([
        xm.container(
            image_path=docker_image, executor_spec=xm_local.Vertex.Spec()
        ),
    ])
  return [train_executable]


def _get_executor(
    docker_image_name: str,
    cloud_location: str,
    num_cpu: int,
    num_ram: int,
    geofm_accelerator_type: Optional[str] = None,
) -> xm.Executor:
  """Returns the executor for the training job based on model type.

  Args:
    docker_image_name: The name of the Docker image to be used to run the
      training job, e.g. 'siglip-tpu' or 'geofm-gpu'.
    cloud_location: The cloud location to run the job in.
    num_cpu: The number of CPUs to use for the job.
    num_ram: The number of RAM to use for the job.
    geofm_accelerator_type: The type of accelerator to use for GeoFM, e.g.
      'A100' or 'CPU'.

  Returns:
    The XM executor for the training job, specifying the required resources.
  """
  job_kwargs = {
      'service_tier': xm.ServiceTier.PROD,
      'location': cloud_location,
      'cpu': num_cpu * xm.vCPU,
      'ram': num_ram * xm.GiB,
  }
  if docker_image_name.startswith('geofm'):
    if geofm_accelerator_type == 'A100':
      job_kwargs[geofm_accelerator_type] = _ACCELERATOR_COUNT
  else:
    job_kwargs[_SIGLIP_ACCELERATOR_TYPE] = _ACCELERATOR_COUNT
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


def build_experiment_jobs(experiment: xm.Experiment,
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
                          geofm_accelerator_type: str,
                          geofm_docker_image: str) -> None:
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
    siglip_model_variant: The model variant to use for SigLIP.
    image_size: The image size to use for SigLIP.
    negative_labels_filepath: The path to the negative labels file.
    positive_labels_filepath: The path to the positive labels file.
    cloud_labels_filepath: The path to the cloud labels file.
    nocloud_labels_filepath: The path to the nocloud labels file.
    batch_size: The batch size to use for the SigLIP training job.
    siglip_docker_image: The path to the SigLIP docker image.
    geofm_savedmodel_path: The path to the GeoFM savedmodel.
    geofm_accelerator_type: The type of accelerator to use for GeoFM, e.g.
      'A100' or 'CPU'.
    geofm_docker_image: The path to the GeoFM docker image.

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
    siglip_image = _get_docker_image_name('siglip')
    siglip_operand = ['/skai/src/skai/model/vlm_zero_shot_vertex.py']
    siglip_job_args = job_args | {
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
        experiment,
        'siglip',
        build_docker_image,
        siglip_image,
        siglip_docker_image=siglip_docker_image)
    siglip_executor = _get_executor(
        siglip_image,
        cloud_location,
        num_cpu,
        num_ram)
    siglip_job = xm.Job(
        executable=siglip_executable,
        executor=siglip_executor,
        args=siglip_job_args,
        name='inference_siglip')
    xm_jobs['siglip'] = siglip_job

  if model_type in ['geofm', 'ensemble']:
    geofm_image = _get_docker_image_name('geofm')
    geofm_operand = ['/skai/src/skai/model/geofm_zero_shot_vertex.py']
    geofm_job_args = job_args | {
        'geofm_savedmodel_path': geofm_savedmodel_path,
    }
    geofm_job_args = xm.merge_args(geofm_operand, geofm_job_args)
    [geofm_executable] = _get_train_executable(
        experiment,
        'geofm',
        build_docker_image,
        geofm_image,
        geofm_docker_image=geofm_docker_image)
    geofm_executor = _get_executor(
        geofm_image,
        cloud_location,
        num_cpu,
        num_ram,
        geofm_accelerator_type=geofm_accelerator_type)
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
    with tempfile.NamedTemporaryFile() as f:
      f.write(b'cloudpickle')
      os.rename(f.name, os.path.join(_CONTROLLER_TMP_DIR, 'requirements.txt'))
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
      async def run_ensemble(experiment: xm.Experiment) -> None:
        xm_local.Vertex()  # Initialize XM Vertex again in the remote job.
        operations = []
        for job in xm_jobs.values():
          operations.append(
              await experiment.add(job)
          )
        await asyncio.gather(*(op.wait_until_complete() for op in operations))
        # TODO(jlee24): Add a final job to merge the results.

    experiment.add(run_ensemble())

  else:
    raise ValueError(f'Unknown model type {model_type}')
