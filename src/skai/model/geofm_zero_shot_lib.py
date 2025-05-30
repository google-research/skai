"""Library for using Vision-Language for zero-shot evaluation of damage."""

import collections
import functools
from typing import Sequence

import ml_collections
import numpy as np
import pandas as pd
import tensorflow as tf

OUTPUT_FEATURES = [
    'building_id',
    'example_id',
    'int64_id',
    'plus_code',
    'label',
]

DAMAGE_THRESHOLD = 0.85
BATCH_SIZE = 1  # Exported GeoFM models for SKAI have batch size 1.


class GeoFM():
  """GeoFM model."""

  def __init__(self, config: ml_collections.ConfigDict):
    self._model = tf.saved_model.load(config.savedmodel_path, tags=['serve'])

  @tf.function(autograph=True, jit_compile=True)
  def predict(self, images: np.ndarray) -> np.ndarray:
    """Generate probability scores for a batch of images."""
    return self._model.signatures['serving_default'](image_batch=images)[
        'output_0'
    ]


def _resize(
    image: tf.Tensor, new_size: int, method: str
) -> tf.Tensor:
  """Resizes and centercrops an image.

  Args:
    image: The image to resize and centercrop.
    new_size: The size to resize the image to.
    method: The method to use for resizing.

  Returns:
    The resized and centercropped image.
  """
  return tf.image.resize(image, [new_size, new_size], method=method)


def _rgb_norm(
    image: tf.Tensor,
    scale: int,
    mean_rgb: Sequence[float],
    stddev_rgb: Sequence[float]
):
  """Normalizes the RGB channels of an image.

  Args:
    image: The image to normalize.
    scale: The scale to apply to the image.
    mean_rgb: The mean RGB values to subtract.
    stddev_rgb: The standard deviation RGB values to divide by.

  Returns:
    The normalized image.
  """
  rank = image.shape.ndims
  shape = [1] * (rank - 1) + [len(mean_rgb)]
  mean_rgb = [i * scale for i in mean_rgb]
  stddev_rgb = [i * scale for i in stddev_rgb]
  image -= tf.constant(mean_rgb, shape=shape, dtype=image.dtype)
  image /= tf.constant(stddev_rgb, shape=shape, dtype=image.dtype)
  return image


def parse_examples(
    record_bytes: tf.train.Example,
    image_feature: str,
    image_size: int,
    mean_rgb: Sequence[float],
    stddev_rgb: Sequence[float],
) -> dict[str, tf.Tensor]:
  """Specifies how to parse a single example.

  Args:
    record_bytes: The record bytes to parse.
    image_feature: String of the feature to use as input image.
    image_size: The size of the input image, e.g. 224.
    mean_rgb: The mean RGB values to subtract.
    stddev_rgb: The standard deviation RGB values to divide by.

  Returns:
    The parsed example.
  """
  example = tf.io.parse_single_example(
      record_bytes,
      {
          image_feature: tf.io.FixedLenFeature([], tf.string),
          'example_id': tf.io.FixedLenFeature([], tf.string),
          'int64_id': tf.io.FixedLenFeature([], tf.int64),
          'coordinates': tf.io.FixedLenFeature([2], tf.float32),
          'encoded_coordinates': tf.io.FixedLenFeature([], tf.string),
          'plus_code': tf.io.FixedLenFeature([], tf.string),
          'label': tf.io.FixedLenFeature([], tf.float32, -1),
      },
  )
  example['building_id'] = example['encoded_coordinates']

  image = tf.io.decode_image(
      example[image_feature],
      channels=3,
      expand_animations=False,
      dtype=tf.uint8,
  )
  resize_fn = functools.partial(
      _resize,
      new_size=image_size,
      method='bicubic',
  )
  image = resize_fn(image)
  image = _rgb_norm(image, 255, mean_rgb, stddev_rgb)
  example['image'] = image
  del example[image_feature]
  return example


def create_geofm_inference_dataset(
    image_size: int, path: str, image_feature: str, mean_rgb: Sequence[float],
    stddev_rgb: Sequence[float],
) -> tf.data.Dataset:
  """Create dataset for GeoFM inference.

  Args:
    image_size: The image size.
    path: Pattern for the dataset filepaths.
    image_feature: Example feature to use as input image.
    mean_rgb: The mean RGB values to subtract.
    stddev_rgb: The standard deviation RGB values to divide by.

  Returns:
    dataset: The dataset.
  """
  paths = tf.io.gfile.glob(path)
  dataset = tf.data.Dataset.from_tensor_slices(paths)
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=len(paths),
      num_parallel_calls=tf.data.AUTOTUNE,
      block_length=50,
      deterministic=False,
  )
  parse_examples_fn = functools.partial(
      parse_examples,
      image_feature=image_feature,
      image_size=image_size,
      mean_rgb=mean_rgb,
      stddev_rgb=stddev_rgb,
  )
  dataset = (
      dataset.map(parse_examples_fn, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(tf.data.AUTOTUNE)
  )
  return dataset


def _dedup_predictions(predictions: pd.DataFrame):
  return predictions.groupby('building_id').agg({
      'damage_score': 'mean',
      'longitude': 'mean',
      'latitude': 'mean',
      'example_id': 'first',
      'int64_id': 'first',
      'plus_code': 'first',
      'label': 'first',
  })


def generate_geofm_zero_shot_assessment(
    model_config: ml_collections.ConfigDict,
    dataset_names: list[str],
    dataset_paths: list[str],
    image_feature: str,
    image_size: int,
    output_dir: str
):
  """Generate zero shot assessment.

  Args:
    model_config: Configuration dict that specifys how the model would be
      loaded.
    dataset_names: List of dataset names.
    dataset_paths: List of dataset filepaths.
    image_feature: Example feature to use as input image.
    image_size: The image size, e.g. 224.
    output_dir: The output directory.
  """
  model = GeoFM(model_config)
  mean_rgb = model_config.mean_norm
  stddev_rgb = model_config.stddev_norm
  for dataset_name, dataset_file_path in zip(dataset_names, dataset_paths):
    dataset = create_geofm_inference_dataset(
        image_size,
        dataset_file_path,
        image_feature,
        mean_rgb,
        stddev_rgb,
    )
    result = collections.defaultdict(list)
    for examples in dataset.as_numpy_iterator():
      scores = model.predict(examples['image']).numpy()
      # Sanity check.
      assert (
          np.allclose(1, scores[:, 0] + scores[:, 1])
      ), 'scores should sum to 1'
      result['damage_score'].extend(scores[:, 1])
      result['longitude'].extend(examples['coordinates'][:, 0])
      result['latitude'].extend(examples['coordinates'][:, 1])
      for key in OUTPUT_FEATURES:
        result[key].extend(examples[key])

    # TODO(mohammedelfatihsalah): Threshold as a flag.
    result['damage'] = [s > DAMAGE_THRESHOLD for s in result['damage_score']]

    output_df = pd.DataFrame(result)
    # Convert bytes columns to str
    for column in output_df.columns:
      if output_df[column].dtype == np.object_:
        output_df[column] = output_df[column].str.decode('utf-8')

    with tf.io.gfile.GFile(
        f'{output_dir}/{dataset_name}_geofm_output.csv', 'w'
    ) as output_csv_file:
      output_df.to_csv(output_csv_file, index=False)

    deduped = _dedup_predictions(output_df)
    with tf.io.gfile.GFile(
        f'{output_dir}/{dataset_name}_geofm_deduped.csv', 'w'
    ) as deduped_file:
      deduped.to_csv(deduped_file, index=False)
