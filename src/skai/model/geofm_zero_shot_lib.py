"""Library for using Vision-Language for zero-shot evaluation of damage."""

import collections
import functools

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


def parse_examples(
    record_bytes: tf.train.Example,
    image_feature: str
) -> dict[str, tf.Tensor]:
  """Specifies how to parse a single example.

  Args:
    record_bytes: The record bytes to parse.
    image_feature: String of the feature to use as input image.

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
  example['image'] = image
  del example[image_feature]
  return example


def create_geofm_inference_dataset(
    path: str, image_feature: str
) -> tf.data.Dataset:
  """Create dataset for GeoFM inference.

  Args:
    path: Pattern for the dataset filepaths.
    image_feature: Example feature to use as input image.

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
    output_dir: str
):
  """Generate zero shot assessment.

  Args:
    model_config: Configuration dict that specifys how the model would be
      loaded.
    dataset_names: List of dataset names.
    dataset_paths: List of dataset filepaths.
    image_feature: Example feature to use as input image.
    output_dir: The output directory.
  """
  model = GeoFM(model_config)
  for dataset_name, dataset_file_path in zip(dataset_names, dataset_paths):
    dataset = create_geofm_inference_dataset(
        dataset_file_path,
        image_feature,
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
