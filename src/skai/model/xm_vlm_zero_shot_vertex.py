r"""XManager launch script for zero_shot eval on Vertex AI.

This script launches an XManager experiment for zero-shot inference. It launches
the zero-shot inference job on a single TPU.

Usage:

xmanager launch src/skai/model/xm_vlm_zero_shot_vertex.py -- \
    --model_variant='So400m/14' \
    --batch_size=128 \
    --image_size=224 \
    --dataset_names='hurricane_ian' \
    --example_patterns='/path/to/hurricane_ian_dataset' \
    --output_dir='/path/to/output_dir' \
    --source_dir='/tmp/skai'

"""

from absl import flags
from skai.model import docker_instructions
from xmanager import xm
from xmanager import xm_local
from xmanager.cloud import vertex as xm_vertex


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory.', required=True
)

_EXAMPLE_PATTERNS = flags.DEFINE_list(
    'example_patterns',
    None,
    'List of file patterns to the input datasets.',
    required=True,
)

_DATASET_NAMES = flags.DEFINE_list(
    'dataset_names', None, 'List of dataset names.'
)

_POSITIVE_LABELS_FILEPATH = flags.DEFINE_string(
    'positive_labels_filepath',
    'gs://skai-public/VLM/damaged_labels.txt',
    'File path to a text file containing positive labels.',
)

_NEGATIVE_LABELS_FILEPATH = flags.DEFINE_string(
    'negative_labels_filepath',
    'gs://skai-public/VLM/undamaged_labels.txt',
    'File path to a text file containing negative labels.',
)

_CLOUD_LABELS_FILEPATH = flags.DEFINE_string(
    'cloud_labels_filepath',
    'gs://skai-public/VLM/cloud_labels.txt',
    'File path to a text file containing cloud labels.',
)

_NOCLOUD_LABELS_FILEPATH = flags.DEFINE_string(
    'nocloud_labels_filepath',
    'gs://skai-public/VLM/nocloud_labels.txt',
    'File path to a text file containing nocloud labels.',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 128, 'Batch size for the inference.'
)

_IMAGE_FEATURE = flags.DEFINE_string(
    'image_feature',
    'post_image_png_large',
    'Feature to use as the input image.',
)

_MODEL_TYPE = flags.DEFINE_enum(
    'model_type', 'siglip', ['siglip', 'geofm'], 'Specifies which zero-shot '
    'model to use.'
)

_GEOFM_SAVEDMODEL_PATH = flags.DEFINE_string(
    'geofm_savedmodel_path',
    '/cns/od-d/home/skai-dev/experiments/jihyeonlee/80percent/vlp=1x1_exported3/cpu/',
    'Path to the exported GeoFM SavedModel.',
)

_SIGLIP_MODEL_VARIANT = flags.DEFINE_string(
    'siglip_model_variant', 'So400m/14', 'Specifies model variant for SigLIP. '
    'Options are "B/16", "L/16", "So400m/14" and "B/16-i18n". Note that each '
    'siglip_model_variant supports a specific set of image sizes.'
)

_IMAGE_SIZE = flags.DEFINE_integer('image_size', 224, 'Image size.')

_SOURCE_DIR = flags.DEFINE_string(
    'source_dir', None, 'Path to the dirctory containg the skai source code.'
)

_BUILD_DOCKER_IMAGE = flags.DEFINE_bool(
    'build_docker_image',
    False,
    'If true, build a docker image from source. Otherwise, use a pre-built'
    ' docker image.',
)

_DOCKER_IMAGE = flags.DEFINE_string(
    'docker_image', None, 'Pre-built Docker image to use.'
)

_CLOUD_LOCATION = flags.DEFINE_string(
    'cloud_location', None, 'Google Cloud region to run jobs in.'
)

_DEFAULT_DOCKER_IMAGE = 'gcr.io/disaster-assessment/skai-ml-tpu:latest'


def main(_) -> None:
  if _CLOUD_LOCATION.value is None:
    raise ValueError('Google Cloud location must be set')
  xm_vertex.set_default_client(xm_vertex.Client(location=_CLOUD_LOCATION.value))

  experiment_name = []
  experiment_name.append(_MODEL_TYPE.value)
  if _MODEL_TYPE.value == 'siglip':
    # For siglip specifically, model variant must be specified, and each variant
    # supports a specific set of image sizes.
    experiment_name.append(_SIGLIP_MODEL_VARIANT.value)
    experiment_name.append(str(_IMAGE_SIZE.value))
  if _DATASET_NAMES.value:
    experiment_name.extend(_DATASET_NAMES.value)

  experiment_name = '_'.join(experiment_name)

  with xm_local.create_experiment(
      experiment_title=experiment_name
  ) as experiment:
    if _BUILD_DOCKER_IMAGE.value:
      # TODO(jlee24): Add support for TPU when b/399193238 is resolved.
      # TODO(jlee24): Add support for parallel launch of siglip and geofm.
      accelerator = 'tpu' if _MODEL_TYPE.value == 'siglip' else 'cpu'
      [train_executable] = experiment.package([
          xm.Packageable(
              executable_spec=docker_instructions.get_xm_executable_spec(
                  accelerator, _MODEL_TYPE.value
              ),
              executor_spec=xm_local.Vertex.Spec(),
          ),
      ])
    else:
      [train_executable] = experiment.package([
          xm.container(
              image_path=(_DOCKER_IMAGE.value or _DEFAULT_DOCKER_IMAGE),
              executor_spec=xm_local.Vertex.Spec()
          ),
      ])

    job_kwargs = {
        'service_tier': xm.ServiceTier.PROD,
        'location': _CLOUD_LOCATION.value,
        'cpu': 8 * xm.vCPU,
        'ram': 64 * xm.GiB,
    }
    if _MODEL_TYPE.value == 'siglip':
      job_kwargs['TPU_V3'] = 8
    executor = xm_local.Vertex(
        requirements=xm.JobRequirements(**job_kwargs),
    )

    args = {
        'example_patterns': ','.join(_EXAMPLE_PATTERNS.value),
        'output_dir': _OUTPUT_DIR.value,
        'image_feature': _IMAGE_FEATURE.value,
    }
    if _DATASET_NAMES.value:
      args['dataset_names'] = ','.join(_DATASET_NAMES.value)
    if _MODEL_TYPE.value == 'siglip':
      operand = ['/skai/src/skai/model/vlm_zero_shot_vertex.py']
      args.update({
          'siglip_model_variant': _SIGLIP_MODEL_VARIANT.value,
          'image_size': _IMAGE_SIZE.value,
          'negative_labels_filepath': _NEGATIVE_LABELS_FILEPATH.value,
          'positive_labels_filepath': _POSITIVE_LABELS_FILEPATH.value,
          'cloud_labels_filepath': _CLOUD_LABELS_FILEPATH.value,
          'nocloud_labels_filepath': _NOCLOUD_LABELS_FILEPATH.value,
          'batch_size': _BATCH_SIZE.value,
      })
    elif _MODEL_TYPE.value == 'geofm':
      operand = ['/skai/src/skai/model/geofm_zero_shot_vertex.py']
      args.update({
          'geofm_savedmodel_path': _GEOFM_SAVEDMODEL_PATH.value,
      })
    else:
      raise ValueError(f'Unsupported model type: {_MODEL_TYPE.value}')
    xm_args = xm.merge_args(operand, args)

    experiment.add(
        xm.Job(
            executable=train_executable,
            args=xm_args,
            executor=executor,
        )
    )
