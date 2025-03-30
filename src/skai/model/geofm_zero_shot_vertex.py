"""Runs GeoFM zero-shot evaluation on a dataset using Vertex AI.

Currently this module does not support running the inference on a backend other
than CPU, and it can only be run through xm_vlm_zero_shot_vertex module.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from skai.model import geofm_zero_shot_lib
from skai.model import public_vlm_config
import tensorflow as tf

IMAGE_SIZE = 224

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

_IMAGE_FEATURE = flags.DEFINE_string(
    'image_feature',
    'post_image_png_large',
    'Feature to use as the input image.',
)

_GEOFM_SAVEDMODEL_PATH = flags.DEFINE_string(
    'geofm_savedmodel_path',
    None,
    'Path to the exported GeoFM SavedModel.',
)


def _check_example_patterns(patterns: list[str]) -> None:
  if not patterns:
    raise ValueError('example_patterns cannot be empty.')
  for pattern in patterns:
    paths = tf.io.gfile.glob(pattern)
    if not paths:
      raise ValueError(
          f'Examples pattern "{pattern}" does not match any files.'
      )


def main(argv: Sequence[str]) -> None:
  del argv

  _check_example_patterns(_EXAMPLE_PATTERNS.value)

  # Get configuration for the model.
  model_config = public_vlm_config.get_model_config(
      'geofm',
      '',
      IMAGE_SIZE,
      _GEOFM_SAVEDMODEL_PATH.value,
  )

  if not _DATASET_NAMES.value:
    dataset_names = [
        f'dataset_{i}' for i in range(len(_EXAMPLE_PATTERNS.value))
    ]
  else:
    dataset_names = _DATASET_NAMES.value

  geofm_zero_shot_lib.generate_geofm_zero_shot_assessment(
      model_config,
      dataset_names,
      _EXAMPLE_PATTERNS.value,
      _IMAGE_FEATURE.value,
      IMAGE_SIZE,
      _OUTPUT_DIR.value,
  )


if __name__ == '__main__':
  app.run(main)
