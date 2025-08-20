"""Library for using Vision-Language for zero-shot evaluation of damage."""

import abc
import collections
from typing import Iterator

from big_vision.models.proj.image_text import two_towers
from big_vision.pp import builder as pp_builder
# Below unused imports are needed to construct required global variables.
# pylint:disable=unused-import
import big_vision.pp.ops_general
import big_vision.pp.ops_image
import big_vision.pp.ops_text
# pylint:enable=unused-import
import flax
import jax
import ml_collections
import numpy as np
import pandas as pd
from skai.model import cloud_postprocess_lib
import tensorflow as tf

OUTPUT_FEATURES = [
    'building_id',
    'example_id',
    'int64_id',
    'plus_code',
    'label',
]

DAMAGE_THRESHOLD = 0.85

_CLOUD_DISTANCE_THRESHOLD_METERS = 500


def _batch_array(
    array: np.ndarray, batch_size: int = 32
) -> Iterator[np.ndarray]:
  """Batch a numpy array.

  Args:
    array: The input numpy array to be batched with at least one dimension.
    batch_size: The size of the batch for which the array will be batched.

  Yields:
    A sequence of batched numpy arrays each with batch size of batch_size.
    If number of arrays is not dividable then the last array will have a
    batch size of number_of_arrays % batch_size.
  """
  for i in range(0, array.shape[0], batch_size):
    yield array[i : i + batch_size]


class VLM(abc.ABC):
  """Provide a generic interface for Vision Language models.

  Attributes:
    _label_embeddings: The embeddings of the labels, which is a 2-D array of
      shape (num_labels, embedding_dim).
    _label_embedding_list: A list of embeddings for each label from which
      _label_embeddings is constructed.
  """

  def __init__(self):
    self._label_embeddings = None
    self._label_embedding_list = []

  @property
  def label_embeddings(self) -> np.ndarray:
    """Get label embeddings."""
    if self._label_embeddings is not None:
      return self._label_embeddings
    elif self._label_embedding_list:
      self._label_embeddings = np.stack(self._label_embedding_list, axis=0)
      return self._label_embeddings
    else:
      raise ValueError('Label embeddings are not set.')

  @abc.abstractmethod
  def tokenize(self, texts: list[str]) -> np.ndarray:
    raise NotImplementedError()

  @abc.abstractmethod
  def encode_tokens(self, tokens: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  @abc.abstractmethod
  def encode_images(self, images: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  @abc.abstractmethod
  def get_temperature(self) -> float:
    raise NotImplementedError()

  def set_label_embeddings(
      self,
      labels: list[str]
  ):
    """Set label embeddings.

    Args:
      labels: List of label descriptions
    """

    def _get_embedding(labels):
      embeddings = []
      for labels_batch in _batch_array(np.array(labels)):
        tokens = self.tokenize(labels_batch.tolist())
        embeddings.append(self.encode_tokens(tokens))
      embeddings = np.concatenate(embeddings, axis=0)
      embedding = np.mean(embeddings, axis=0)
      return embedding
    self._label_embeddings = None
    self._label_embedding_list.append(_get_embedding(labels))

  def predict(self, images: np.ndarray) -> np.ndarray:
    """Generate probability scores for a batch of images.

    Args:
      images: A batch of images.

    Returns:
      A 2-D array of shape (batch, 2), where the second dimension contains the
        positive and negative class probabilities respectively.

    Raises:
     ValueError if the connected device is not TPU or labels are not set.
    """
    if self.label_embeddings is None:
      raise ValueError('Label embeddings are not set.')

    if jax.extend.backend.get_backend().platform != 'tpu':
      raise ValueError('Not connected to TPU.')

    batch_size, image_size, _, _ = images.shape
    num_images_to_augment = 0
    if batch_size % jax.local_device_count() != 0:
      num_images_to_augment = jax.local_device_count() - (
          batch_size % jax.local_device_count()
      )
      images_to_augment = np.zeros((num_images_to_augment,) + images.shape[1:])
      images = np.concatenate([images, images_to_augment], axis=0)

    images = images.reshape(
        jax.local_device_count(),
        (batch_size + num_images_to_augment) // jax.local_device_count(),
        image_size,
        image_size,
        3,
    )
    image_embeddings = self.encode_images(images)
    image_embeddings = image_embeddings.reshape(
        batch_size + num_images_to_augment, -1
    )[:batch_size, :]
    sims = image_embeddings @ self.label_embeddings.T * self.get_temperature()
    probability_scores = np.array(jax.nn.softmax(sims, axis=-1))
    return probability_scores


class WebliViT(VLM):
  """WebliViT model."""

  def __init__(self, config):
    """Initialize WebliViT model.

    Args:
      config: Configuration dict that specifys how the model would be loaded.
        The configuration dict should have the following fields:
          - init_shapes: Tuple specifys the expected shape of the image and the
            tokenized text.
          - model: Dict specifys which model architecture to load.
          - model_init:Dict specifys model checkpoints.
          - evals: Dict specifys preprocessing functions for the image and text.
    """
    super().__init__()
    self.temperature = None
    (_, image_size, _, _), _ = config.init_shapes
    self.image_size = image_size
    self._core_model = two_towers.Model(**config.get('model', {}))
    self._params = two_towers.load(None, config.model_init, config.model)
    self._p_params = flax.jax_utils.replicate(self._params)
    self.preprcoess_txt = pp_builder.get_preprocess_fn(
        config.evals.pp_txt
    )
    self._encode = jax.pmap(
        lambda params, images: self._core_model.apply(
            {'params': params}, images, None
        )[0]
    )

  def tokenize(self, texts: list[str]) -> np.ndarray:
    texts = tf.convert_to_tensor(
        [tf.convert_to_tensor([text]) for text in texts], dtype=tf.string
    )
    return tf.map_fn(
        lambda text: self.preprcoess_txt({'texts': text})['labels'],
        texts,
        tf.int32,
    ).numpy()

  def encode_tokens(self, tokens: np.ndarray) -> np.ndarray:
    _, ztxt, _ = self._core_model.apply({'params': self._params}, None, tokens)
    return np.array(ztxt)

  def get_temperature(self) -> float:
    if self.temperature is None:
      _, _, out = self._core_model.apply({'params': self._params}, None, None)
      self.temperature = float(out['t'][0])
    return self.temperature

  def encode_images(self, images: np.ndarray) -> np.ndarray:
    embd = self._encode(self._p_params, images)
    return np.array(embd, np.float32)


def create_inference_dataset(
    image_size: int, pp_img: str, path: str, batch_size: int, image_feature: str
) -> tf.data.Dataset:
  """Create dataset for VLM inference.

  Args:
    image_size: The image size.
    pp_img: String repersent the preprocessing functions for the image.
    path: Pattern for the dataset filepaths.
    batch_size: The size of the batch.
    image_feature: Example feature to use as input image.

  Returns:
    dataset: The dataset.
  """
  paths = tf.io.gfile.glob(path)
  dataset = tf.data.Dataset.from_tensor_slices(paths)
  image_preprocess_fn = pp_builder.get_preprocess_fn(pp_img)
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=len(paths),
      num_parallel_calls=tf.data.AUTOTUNE,
      block_length=50,
      deterministic=False,
  )

  def _parse_examples(record_bytes) -> dict[str, tf.Tensor]:
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
    image = tf.image.resize(
        tf.io.decode_image(
            example[image_feature],
            channels=3,
            expand_animations=False,
            dtype=tf.float32,
        ),
        [image_size, image_size],
    )
    image = tf.cast(image * 255, tf.uint8)
    example['image'] = image
    del example[image_feature]
    return example

  def image_preprocessing(example):
    example['image'] = image_preprocess_fn({'image': example['image']})['image']
    return example

  dataset = (
      dataset.map(_parse_examples, num_parallel_calls=tf.data.AUTOTUNE)
      .map(image_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(batch_size)
      .prefetch(tf.data.AUTOTUNE)
  )
  return dataset


def _dedup_predictions(predictions: pd.DataFrame):
  if 'is_cloudy' in predictions.columns:
    non_cloudy = predictions[predictions['is_cloudy'] == 0]
  else:
    non_cloudy = predictions
  return non_cloudy.groupby('building_id').agg({
      'damage_score': 'mean',
      'cloud_score': 'mean',
      'longitude': 'mean',
      'latitude': 'mean',
      'example_id': 'first',
      'int64_id': 'first',
      'plus_code': 'first',
      'label': 'first',
  }).reset_index()


def generate_zero_shot_assessment(
    model_config: ml_collections.ConfigDict,
    damage_label_file_path: str,
    undamage_label_file_path: str,
    cloud_label_file_path: str,
    nocloud_label_file_path: str,
    dataset_names: list[str],
    dataset_paths: list[str],
    image_feature: str,
    batch_size: int,
    output_dir: str,
    postprocess_clouds: bool,
):
  """Generate zero shot assessment.

  Args:
    model_config: Configuration dict that specifys how the model would be
      loaded.
    damage_label_file_path: Path to file containing the text descriptions of
      damage buildings. Each description is separated by a new line.
    undamage_label_file_path: Path to file containing the text descriptions of
      undamaged buildings. Each description is separated by a new line.
    cloud_label_file_path: Path to file containing the text descriptions of
      buildings covered by clouds. Each description is separated by a new line.
    nocloud_label_file_path: Path to file containing the text descriptions of
      buildings not covered by clouds. Each description is separated by a new
      line.
    dataset_names: List of dataset names.
    dataset_paths: List of dataset filepaths.
    image_feature: Example feature to use as input image.
    batch_size: The size of the batch.
    output_dir: The output directory.
    postprocess_clouds: If true, run postprocessing heuristics for
      identifying cloudy examples.
  """
  label_file_paths = [
      damage_label_file_path,
      undamage_label_file_path,
      cloud_label_file_path,
      nocloud_label_file_path,
  ]
  model = WebliViT(model_config)
  for path in label_file_paths:
    with tf.io.gfile.GFile(path, 'r') as f:
      labels = [label.strip() for label in f.readlines()]
      model.set_label_embeddings(labels)

  for dataset_name, dataset_file_path in zip(dataset_names, dataset_paths):
    image_size = model_config.init_shapes[0][1]
    dataset = create_inference_dataset(
        image_size,
        model_config.evals.pp_img,
        dataset_file_path,
        batch_size,
        image_feature,
    )
    result = collections.defaultdict(list)
    for examples in dataset.as_numpy_iterator():
      scores = model.predict(examples['image'])
      # Normalize damage and cloud scores to be in the range of [0, 1].
      damage_normalizer = scores[:, 0] + scores[:, 1]
      cloud_normalizer = scores[:, 2] + scores[:, 3]
      scores[:, 0] /= damage_normalizer
      scores[:, 1] /= damage_normalizer
      scores[:, 2] /= cloud_normalizer
      scores[:, 3] /= cloud_normalizer
      # Sanity check.
      assert (
          np.allclose(1, scores[:, 0] + scores[:, 1])
      ), 'scores should sum to 1'
      assert (
          np.allclose(1, scores[:, 2] + scores[:, 3])
      ), 'scores should sum to 1'

      result['damage_score'].extend(scores[:, 0])
      result['cloud_score'].extend(scores[:, 2])
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
    if postprocess_clouds:
      output_df = cloud_postprocess_lib.identify_clouds(
          output_df, _CLOUD_DISTANCE_THRESHOLD_METERS
      )

    with tf.io.gfile.GFile(
        f'{output_dir}/{dataset_name}_output.csv', 'w'
    ) as output_csv_file:
      output_df.to_csv(output_csv_file, index=False)

    deduped = _dedup_predictions(output_df)
    with tf.io.gfile.GFile(
        f'{output_dir}/{dataset_name}_deduped.csv', 'w'
    ) as deduped_file:
      deduped.to_csv(deduped_file, index=False)
