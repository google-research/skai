"""Runs VLM zero-shot evaluation on a dataset using Vertex AI.

Currently this module does not support running the inference on a backend other
than TPU and it can only be run through xm_vlm_zero_shot_vertex module.
"""

from collections.abc import Sequence
from absl import app
from absl import flags
from skai.model import public_vlm_config
from skai.model import vlm_zero_shot_lib

import tensorflow as tf

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

_MODEL_VARIANT = flags.DEFINE_string(
    'model_variant', 'So400m/14', 'Specifies model variant. Available model '
    'variants are "B/16", "L/16", "So400m/14" and "B/16-i18n". Note that each '
    'model_variant supports a specific set of image sizes.'
)

_IMAGE_SIZE = flags.DEFINE_integer('image_size', 224, 'Image size.')


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

  # Get configuration VLM model.
  model_config = public_vlm_config.get_model_config(
      _MODEL_VARIANT.value, _IMAGE_SIZE.value
  )

  if not _DATASET_NAMES.value:
    dataset_names = [
        f'dataset_{i}' for i in range(len(_EXAMPLE_PATTERNS.value))
    ]
  else:
    dataset_names = _DATASET_NAMES.value

  vlm_zero_shot_lib.generate_zero_shot_assessment(
      model_config,
      _POSITIVE_LABELS_FILEPATH.value,
      _NEGATIVE_LABELS_FILEPATH.value,
      _CLOUD_LABELS_FILEPATH.value,
      _NOCLOUD_LABELS_FILEPATH.value,
      dataset_names,
      _EXAMPLE_PATTERNS.value,
      _IMAGE_FEATURE.value,
      _BATCH_SIZE.value,
      _OUTPUT_DIR.value
  )


if __name__ == '__main__':
  app.run(main)
