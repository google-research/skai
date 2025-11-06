r"""XManager launch script for zero_shot eval on Vertex AI.

This script launches an XManager experiment for zero-shot inference. It launches
the zero-shot inference job on a single TPU.

Example Usage for old SigLIP:

xmanager launch src/skai/model/xm_vlm_zero_shot_vertex.py -- \
    --batch_size=128 \
    --dataset_names='hurricane_ian' \
    --example_patterns='/path/to/hurricane_ian_dataset' \
    --output_dir='/path/to/output_dir'


To launch the ensemble:

xmanager launch src/skai/model/xm_vlm_zero_shot_vertex.py -- \
  --xm_gcp_service_account_name=skai-dataflow \
  --example_patterns=$EXAMPLES_PATTERN \
  --output_dir=$OUTPUT_DIR \
  --cloud_location=your-cloud-location \
  --cloud_bucket_name=your-bucket-name \
  --cloud_project=your-project \
  --geofm_savedmodel_path='gs://path/to/geofm/savedmodel' \
  --model_type=ensemble

"""

from absl import app
from absl import flags
from skai.model import xm_vlm_zero_shot_vertex_lib
from xmanager import xm_local
from xmanager.cloud import vertex as xm_vertex

_JOB_NAME = flags.DEFINE_string(
    'job_name',
    None,
    'Job name.',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory.', required=True
)

_OUTPUT_ENSEMBLE_CSV_FILE_NAME = flags.DEFINE_string(
    'output_ensemble_csv_file_name',
    'ensemble_predictions.csv',
    'Name of the ensembled predictions CSV file that will be saved in the '
    'output_dir. Only used when model_type is "ensemble".'
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
    'model_type',
    'siglip',
    ['siglip', 'geofm', 'ensemble'],
    'Specifies which zero-shot model to use. When using "ensemble", both '
    'siglip and geofm will be used.'
)

_GEOFM_SAVEDMODEL_PATH = flags.DEFINE_string(
    'geofm_savedmodel_path',
    '',
    'Path to the exported GeoFM SavedModel.',
)

_USE_SIGLIP2 = flags.DEFINE_bool(
    'use_siglip2',
    False,
    'If true, use SigLIP2. Otherwise, use old SigLIP.',
)

_SIGLIP_MODEL_VARIANT = flags.DEFINE_string(
    'siglip_model_variant',
    'So400m/14',
    'Specifies model variant for SigLIP. '
    'Options are "B/16", "L/16", "So400m/14" and "B/16-i18n". Note that each '
    'siglip_model_variant supports a specific set of image sizes.',
)

_SIGLIP_IMAGE_SIZE = flags.DEFINE_integer(
    'siglip_image_size', 224, 'Specifies image size for SigLIP. '
)

_SIGLIP2_MODEL_VARIANT = flags.DEFINE_string(
    'siglip2_model_variant',
    'g-opt/16',
    'Specifies model variant for SigLIP2. '
    'Options are "B/16", "L/16", "So400m/14" and "B/16-i18n". Note that each '
    'siglip_model_variant supports a specific set of image sizes.',
)

_SIGLIP2_IMAGE_SIZE = flags.DEFINE_integer(
    'siglip2_image_size', 256, 'Specifies image size for SigLIP2. '
)

_BUILD_DOCKER_IMAGE = flags.DEFINE_bool(
    'build_docker_image',
    False,
    'If true, build a docker image from source. Otherwise, use a pre-built'
    ' docker image.',
)

_SIGLIP_DOCKER_IMAGE = flags.DEFINE_string(
    'siglip_docker_image',
    'gcr.io/disaster-assessment/skai-ml-siglip-gpu:20251103-082405_103606',
    'Pre-built Docker image to use for siglip.',
)

_GEOFM_DOCKER_IMAGE = flags.DEFINE_string(
    'geofm_docker_image',
    'gcr.io/disaster-assessment/skai-ml-geofm-gpu:20250611-180739_681012',
    'Pre-built Docker image to use for geofm.',
)

_CLOUD_LOCATION = flags.DEFINE_string(
    'cloud_location', None, 'Google Cloud region to run jobs in.', required=True
)

_CLOUD_BUCKET_NAME = flags.DEFINE_string(
    'cloud_bucket_name',
    None,
    'Google Cloud bucket in the same region as '
    'cloud_location used to stage artifacts.',
    required=True,
)

_CLOUD_PROJECT = flags.DEFINE_string(
    'cloud_project',
    None,
    'Google Cloud project to be charged for jobs.',
    required=True,
)

_DEFAULT_DOCKER_IMAGE = 'gcr.io/disaster-assessment/skai-ml-tpu:latest'

_NUM_CPU = flags.DEFINE_integer('num_cpu', 1, 'Number of CPUs to use.')

_NUM_RAM = flags.DEFINE_integer('num_ram', 32, 'Number of RAM to use.')

_GEOFM_ACCELERATOR_TYPE = flags.DEFINE_enum(
    'geofm_accelerator',
    'A100',
    ['A100', 'CPU'],
    'Specifies which platform to use for GeoFM model.',
)


def main(_) -> None:
  xm_vertex.set_default_client(xm_vertex.Client(location=_CLOUD_LOCATION.value))
  siglip_model_variant = (
      _SIGLIP2_MODEL_VARIANT.value
      if _USE_SIGLIP2.value
      else _SIGLIP_MODEL_VARIANT.value
  )
  siglip_image_size = (
      _SIGLIP2_IMAGE_SIZE.value
      if _USE_SIGLIP2.value
      else _SIGLIP_IMAGE_SIZE.value
  )
  if not _DATASET_NAMES.value:
    dataset_names = [
        f'dataset_{i}' for i in range(len(_EXAMPLE_PATTERNS.value))
    ]
  else:
    dataset_names = _DATASET_NAMES.value

  if _JOB_NAME.value:
    experiment_name = _JOB_NAME.value
  else:
    experiment_name = xm_vlm_zero_shot_vertex_lib.get_experiment_name(
        dataset_names, _MODEL_TYPE.value, siglip_model_variant,
        siglip_image_size
    )
  with xm_local.create_experiment(
      experiment_title=experiment_name
  ) as experiment:
    siglip_model_variant = (
        _SIGLIP2_MODEL_VARIANT.value
        if _USE_SIGLIP2.value
        else _SIGLIP_MODEL_VARIANT.value
    )
    siglip_image_size = (
        _SIGLIP2_IMAGE_SIZE.value
        if _USE_SIGLIP2.value
        else _SIGLIP_IMAGE_SIZE.value
    )
    xm_vlm_zero_shot_vertex_lib.build_experiment_jobs(
        experiment,
        _MODEL_TYPE.value,
        _CLOUD_LOCATION.value,
        _CLOUD_BUCKET_NAME.value,
        _CLOUD_PROJECT.value,
        _OUTPUT_DIR.value,
        _EXAMPLE_PATTERNS.value,
        dataset_names,
        _BUILD_DOCKER_IMAGE.value,
        _IMAGE_FEATURE.value,
        _NUM_RAM.value,
        _NUM_CPU.value,
        # SigLIP args.
        _USE_SIGLIP2.value,
        siglip_model_variant,
        siglip_image_size,
        _NEGATIVE_LABELS_FILEPATH.value,
        _POSITIVE_LABELS_FILEPATH.value,
        _CLOUD_LABELS_FILEPATH.value,
        _NOCLOUD_LABELS_FILEPATH.value,
        _BATCH_SIZE.value,
        _SIGLIP_DOCKER_IMAGE.value,
        # GeoFM args.
        _GEOFM_SAVEDMODEL_PATH.value,
        _GEOFM_DOCKER_IMAGE.value,
        output_ensemble_csv_file_name=_OUTPUT_ENSEMBLE_CSV_FILE_NAME.value,
    )


if __name__ == '__main__':
  app.run(main)
