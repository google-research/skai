# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for running model inference in beam."""

import enum
import math
import os
import time
from typing import Any, Iterable, Iterator, NamedTuple, Optional, Tuple

import apache_beam as beam
import apache_beam.dataframe.convert
import apache_beam.dataframe.io
from apache_beam.utils import multi_process_shared
import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkb
import shapely.wkt
from skai import utils
from skai.model import data
import tensorflow as tf
# This import is needed for SentencePiece operations.
import tensorflow_text  # pylint: disable=unused-import


class ModelType(enum.Enum):
  """Model types."""

  VLM = 'vlm'
  CLASSIFICATION = 'classification'


class InferenceRow(NamedTuple):
  """A row in the inference output CSV."""
  example_id: int | None
  building_id: str | None
  longitude: float | None
  latitude: float | None
  score: float | None
  plus_code: str | None
  area_in_meters: float | None
  footprint_wkt: str | None
  damaged: bool | None
  damaged_high_precision: bool | None
  damaged_high_recall: bool | None
  label: float | None


def set_gpu_memory_growth() -> None:
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)


set_gpu_memory_growth()


class InferenceModel(object):
  """Abstract base class for an inference model.

  This should be subclassed for each actual model type.
  """

  def prepare_model(self) -> None:
    """Prepares model for inference calls.

    This function will be called in the setup function of the DoFn.
    """
    raise NotImplementedError()

  def predict_scores(self, batch: list[tf.train.Example]) -> np.ndarray:
    """Predicts scores for a batch of input examples."""
    raise NotImplementedError()


def extract_image_or_blank(
    example: tf.train.Example, feature: str, image_size: int
) -> np.ndarray:
  """Extracts an image from a TF Example.

  If the image feature is missing, returns a blank image instead.

  Args:
    example: Example to extract image from.
    feature: Image feature name.
    image_size: Image size.

  Returns:
    Image as a numpy array, or an array of 0s if image feature is missing.
  """
  if feature in example.features.feature:
    image_bytes = utils.get_bytes_feature(example, feature)[0]
    return data.decode_and_resize_image(image_bytes, image_size).numpy()
  return np.zeros((image_size, image_size, 3), dtype=np.float32)


class TF2VLMModel:
  """VLM model wrapper for SKAI TF2 models."""

  def __init__(
      self,
      model: tf.keras.Model,
      text_embeddings: Optional[dict[str, np.ndarray]] = None,
  ):
    self._text_embeddings = np.stack(
        [text_embeddings['neg'], text_embeddings['pos']], axis=0
    )
    self._text_embeddings = tf.convert_to_tensor(
        self._text_embeddings, dtype=tf.float32
    )
    self._model = model

  def __call__(self, batch: dict[str, Any], **kwargs) -> dict[str, tf.Tensor]:
    """Predicts probabilities for a batch of images.

    Args:
      batch: Dictionary that contains the input images. The batch must has
        "large_image" and the image pixels must normalised in the range 0 - 1.0.
      **kwargs: Other keyword arguments.

    Returns:
      a dictionary that contains probabilities of labels for
      each image example.
    """
    # TODO(mohammedelfatihsalah): check the image size requirement of the saved
    # tf model.
    images = tf.convert_to_tensor(
        batch['large_image'] * 255, dtype=tf.float32
    )
    image_embeddings = self._model.signatures['serving_default'](images)[
        'output_0'
    ]
    # TODO(mohammedelfatihsalah): take the temperature value from the saved tf
    # model.
    sims = image_embeddings @ tf.transpose(self._text_embeddings) * 100
    probs = tf.nn.softmax(sims, axis=-1)
    return {'main': probs}


class TF2InferenceModel(InferenceModel):
  """InferenceModel wrapper for SKAI TF2 models."""

  _model_dir: str
  _image_size: int
  _post_image_only: bool
  _model: Any

  def __init__(
      self,
      model_dir: str,
      image_size: int,
      post_image_only: bool,
      model_type: ModelType,
      text_embeddings: Optional[dict[str, np.ndarray]] = None,
  ):
    self._model_dir = model_dir
    self._model_type = model_type
    self._image_size = image_size
    self._post_image_only = post_image_only
    self._model = None
    self._text_embeddings = text_embeddings

  def _make_dummy_input(self):
    num_channels = 3 if self._post_image_only else 6
    image = np.zeros(
        (1, self._image_size, self._image_size, num_channels), dtype=np.float32
    )
    return {'small_image': image, 'large_image': image}

  def _extract_images_or_blanks(
      self,
      example: tf.train.Example,
      pre_image_feature: str,
      post_image_feature: str,
  ) -> np.ndarray:
    """Extracts pre and post disaster images from an example.

    If the image feature is not present, this function will use a blank image
    (all zeros) as a placeholder.

    Args:
      example: Example to extract the image from.
      pre_image_feature: Name of feature storing the pre-disaster image byte.
      post_image_feature: Name of feature storing the post-disaster image byte.

    Returns:
      Numpy array representing the concatenated images.
    """
    post_image = extract_image_or_blank(
        example, post_image_feature, self._image_size
    )
    if self._post_image_only:
      return post_image
    pre_image = extract_image_or_blank(
        example, pre_image_feature, self._image_size
    )
    return np.concatenate([pre_image, post_image], axis=2)

  def prepare_model(self) -> None:
    # Use a shared handle so that the model is only loaded once per worker and
    # shared by all processing threads. For more details, see
    #
    # https://medium.com/google-cloud/cache-reuse-across-dofns-in-beam-a34a926db848
    def load():
      model = tf.saved_model.load(self._model_dir)
      if self._model_type == ModelType.VLM:
        dummy_images = self._make_dummy_input()
        dummy_images = tf.convert_to_tensor(
            dummy_images['large_image'], dtype=tf.float32
        )
        _ = model.signatures['serving_default'](dummy_images)
        vlm_model = TF2VLMModel(model, self._text_embeddings)
        return vlm_model
      else:
        # Call predict once to make sure any hidden lazy initialization is
        # triggered. See https://stackoverflow.com/a/43393252
        _ = model(self._make_dummy_input())
        return model

    self._model = multi_process_shared.MultiProcessShared(
        load, self._model_dir
    ).acquire()

  def predict_scores(
      self,
      batch: list[tf.train.Example],
  ) -> np.ndarray:
    model_input = self._extract_image_arrays(batch)
    if self._model_type == ModelType.VLM:
      outputs = self._model(
          model_input, training=False
      )
    else:
      outputs = self._model(model_input, training=False)
    return outputs['main'][:, 1]

  def _extract_image_arrays(
      self,
      examples: list[tf.train.Example],
  ) -> dict[str, np.ndarray]:
    """Reads images from a batch of examples as numpy arrays."""
    small_images = []
    large_images = []
    for example in examples:
      small_images.append(
          self._extract_images_or_blanks(
              example, 'pre_image_png', 'post_image_png'
          )
      )
      large_images.append(
          self._extract_images_or_blanks(
              example, 'pre_image_png_large', 'post_image_png_large'
          )
      )
    return {
        'small_image': np.stack(small_images),
        'large_image': np.stack(large_images),
    }


class TextTowerInference(beam.DoFn):
  """Text embedding inference DoFn.

  Generate text embedding using text tower from bigvision models.
  """

  def __init__(self, model_checkpoint: str):
    self._model_checkpoint = model_checkpoint

  def setup(self) -> None:
    def _load():
      model = tf.saved_model.load(self._model_checkpoint)
      model.inference = model.signatures['serving_default']
      return model

    self._model = multi_process_shared.MultiProcessShared(
        _load, 'TextTowerModel'
    ).acquire()

  def process(
      self, batch: Tuple[str, list[str]]
  ) -> Iterator[Tuple[str, np.ndarray]]:
    yield (
        batch[0],
        self._model.inference(tf.convert_to_tensor(batch[1], tf.string))[
            'output_0'
        ].numpy(),
    )


class ModelInference(beam.DoFn):
  """Model inference DoFn."""

  def __init__(self, score_feature: str, model: InferenceModel):
    self._score_feature = score_feature
    self._model = model
    self._examples_processed = beam.metrics.Metrics.counter(
        'skai', 'examples_processed'
    )
    self._batches_processed = beam.metrics.Metrics.counter(
        'skai', 'batches_processed'
    )
    self._inference_millis = beam.metrics.Metrics.distribution(
        'skai', 'batch_inference_msec'
    )

  def setup(self) -> None:
    self._model.prepare_model()

  def process(
      self, batch: list[tf.train.Example]
  ) -> Iterator[tf.train.Example]:
    start_time = time.process_time()
    scores = self._model.predict_scores(batch)
    elapsed_millis = (time.process_time() - start_time) * 1000
    self._inference_millis.update(elapsed_millis)
    for example, score in zip(batch, scores):
      output_example = tf.train.Example()
      output_example.CopyFrom(example)
      if self._score_feature == 'score':
        utils.add_float_feature(self._score_feature, score, output_example)
      elif self._score_feature == 'embedding':
        utils.add_float_list_feature('embedding', score, output_example)

      yield output_example

    self._examples_processed.inc(len(batch))
    self._batches_processed.inc(1)


def _merge_examples(
    keyed_examples: tuple[str, Iterable[tf.train.Example]]
) -> tf.train.Example:
  """Merges examples sharing the same coordinates.

  Args:
    keyed_examples: Tuple of encoded coordinates and examples.

  Returns:
    Example with merged scores or embeddings.
  """
  examples = list(keyed_examples[1])
  output_example = tf.train.Example()
  output_example.CopyFrom(examples[0])
  try:  # scores
    scores = [utils.get_float_feature(e, 'score')[0] for e in examples]
    output_example.features.feature['score'].float_list.value[:] = [
        np.mean(scores)
    ]
  except IndexError:  # embeddings
    scores = [utils.get_float_feature(e, 'embedding') for e in examples]
    output_example.features.feature['embedding'].float_list.value.extend(
        np.mean(np.array(scores), axis=0)
    )
  return output_example


def _dedup_scored_examples(
    examples: beam.PCollection
) -> beam.PCollection:
  """Deduplications examples by merging those sharing the same coordinates.

  Args:
    examples: PCollection of examples with scores.

  Returns:
    PCollection of deduplicated examples.
  """
  return (
      examples
      | 'key_examples_by_coords'
      >> beam.Map(_key_example_by_encoded_coordinates)
      | 'group_by_coords' >> beam.GroupByKey()
      | 'merge_examples' >> beam.Map(_merge_examples)
  )


def _example_id_embeddings(examples: beam.PCollection) -> beam.PCollection:
  return (
      examples
      | 'example_id_embeddings' >> beam.Map(_get_example_ids_and_embeddings)
  )


def _get_example_ids_and_embeddings(
    example: tf.train.Example
) -> tuple[int, np.ndarray]:
  try:
    example_id = utils.get_int64_feature(example, 'int64_id')[0]
  except IndexError as e:
    raise IndexError('No example_id was found.') from e
  try:
    embeddings = np.array(utils.get_float_feature(example, 'embedding'))
  except IndexError as e:
    raise IndexError('No embedding was found.') from e
  return example_id, embeddings


def run_inference(
    examples: beam.PCollection,
    score_feature: str,
    batch_size: int,
    model: InferenceModel,
) -> beam.PCollection:
  """Runs inference and augments input examples with inference scores.

  Args:
    examples: PCollection of Tensorflow Examples.
    score_feature: Feature name to use for inference scores.
    batch_size: Batch size.
    model: Inference model to use.

  Returns:
    PCollection of Tensorflow Examples augmented with inference scores.
  """
  scored_examples = (
      examples
      | 'batch'
      >> beam.transforms.util.BatchElements(
          min_batch_size=batch_size, max_batch_size=batch_size
      )
      | 'inference' >> beam.ParDo(ModelInference(score_feature, model))
  )
  if score_feature == 'embedding':
    return _example_id_embeddings(scored_examples)
  return _dedup_scored_examples(scored_examples)


def _key_example_by_encoded_coordinates(
    example: tf.train.Example,
) -> tuple[str, tf.train.Example]:
  encoded_coordinates = utils.get_bytes_feature(example, 'encoded_coordinates')[
      0
  ]
  return (
      encoded_coordinates.decode(),
      example,
  )


def example_to_row(
    example: tf.train.Example,
    threshold: float,
    high_precision_threshold: float,
    high_recall_threshold: float,
) -> InferenceRow:
  """Convert an example into an inference row.

  Args:
    example: Input example.
    threshold: Damaged score threshold.
    high_precision_threshold: Damaged score threshold for high precision.
    high_recall_threshold: Damaged score threshold for high recall.

  Returns:
    Inference row.
  """

  example_id = utils.get_int64_feature(example, 'int64_id')[0]
  building_id = utils.get_bytes_feature(example, 'encoded_coordinates')[
      0
  ].decode()
  longitude, latitude = utils.get_float_feature(example, 'coordinates')
  try:
    score = utils.get_float_feature(example, 'score')[0]
  except IndexError as e:
    raise KeyError('No score was found.') from e
  try:
    plus_code = utils.get_bytes_feature(example, 'plus_code')[0].decode()
  except IndexError:
    plus_code = ''
  try:
    area = utils.get_float_feature(example, 'area_in_meters')[0]
  except IndexError:
    area = None
  try:
    footprint_wkb = utils.get_bytes_feature(example, 'footprint_wkb')[0]
    footprint_wkt = shapely.wkt.dumps(shapely.wkb.loads(footprint_wkb))
  except IndexError:
    footprint_wkt = None

  try:
    label = utils.get_float_feature(example, 'label')[0]
  except IndexError:
    label = None

  return InferenceRow(
      label=label,
      example_id=example_id,
      building_id=building_id,
      longitude=longitude,
      latitude=latitude,
      score=score,
      plus_code=plus_code,
      area_in_meters=area,
      footprint_wkt=footprint_wkt,
      damaged=(score >= threshold),
      damaged_high_precision=(score >= high_precision_threshold),
      damaged_high_recall=(score >= high_recall_threshold)
  )


def examples_to_csv(
    examples: beam.PCollection,
    threshold: float,
    high_precision_threshold: float,
    high_recall_threshold: float,
    output_prefix: str,
) -> None:
  """Converts TF Examples to CSV lines and writes out to file.

  Args:
    examples: PCollection of Tensorflow Examples.
    threshold: Damaged score threshold.
    high_precision_threshold: Damaged score threshold for high precision.
    high_recall_threshold: Damaged score threshold for high recall.
    output_prefix: CSV output prefix.
  """
  rows = (
      examples
      | 'reshuffle_for_output' >> beam.Reshuffle()
      | 'examples_to_rows'
      >> beam.Map(
          example_to_row,
          threshold=threshold,
          high_precision_threshold=high_precision_threshold,
          high_recall_threshold=high_recall_threshold,
      )
  )
  df = apache_beam.dataframe.convert.to_dataframe(rows)
  apache_beam.dataframe.io.to_csv(df, output_prefix, index=False)


def embeddings_to_row(example_id_embeddings: tuple[int, np.ndarray]):
  example_id, embeddings = example_id_embeddings
  embeddings_str = ', '.join(f'{val:.16f}' for val in embeddings)
  return f'{str(example_id)}, {embeddings_str}'


def embeddings_examples_to_csv(
    embeddings: beam.PCollection, output_prefix: str
) -> None:
  """Converts embeddings to CSV lines and writes out to file.

  Args:
    embeddings: PCollection of embeddings.
    output_prefix: CSV output prefix.
  """
  embedding_len = 64
  cols = ['example_id']
  cols.extend([f'embedding_{i}' for i in range(embedding_len)])
  _ = (
      embeddings
      | 'reshuffle_for_output' >> beam.Reshuffle()
      | 'embeddings_to_row' >> beam.Map(embeddings_to_row)
      | 'write_to_file'
      >> beam.io.textio.WriteToText(output_prefix, header=','.join(cols))
  )


def postprocess(
    temp_prefix: str, output_path: str, inference_feature: str
) -> None:
  """Postprocess Dataflow output.

  - Combines individual CSV shards into a single CSV.
  - Creates a GeoPackage file.

  Args:
    temp_prefix: Prefix used in naming temporary shards.
    output_path: CSV output path.
    inference_feature: Describe the possible outputs of an inference process,
      including 'embeddings' or 'score'.
  """
  shards = []
  temp_files = tf.io.gfile.glob(f'{temp_prefix}*')
  for path in temp_files:
    with tf.io.gfile.GFile(path, 'r') as f:
      shards.append(pd.read_csv(f))
  df = pd.concat(shards, ignore_index=True)
  with tf.io.gfile.GFile(output_path, 'w') as f:
    df.to_csv(f, index=False)

  # Delete all temp files.
  for path in temp_files:
    tf.io.gfile.remove(path)

  if inference_feature == 'embedding':
    return

  if 'GPKG' in fiona.supported_drivers:
    # Output GeoPackage if available.
    geometries = []
    for wkt, lon, lat in zip(
        df['footprint_wkt'], df['longitude'], df['latitude']
    ):
      if not isinstance(wkt, str) and math.isnan(wkt):
        geometries.append(shapely.geometry.Point(lon, lat))
      else:
        geometries.append(shapely.wkt.loads(wkt))
    gdf = gpd.GeoDataFrame(
        df.drop(columns=['footprint_wkt']), geometry=geometries
    )
    output_dir, output_file = os.path.split(output_path)
    gpkg_path = os.path.join(output_dir, f'{output_file}.gpkg')
    with tf.io.gfile.GFile(gpkg_path, 'wb') as f:
      gdf.to_file(f, driver='GPKG')


def _do_batch(labels: list[str], batch_size: int) -> list[list[str]]:
  """batch labels."""
  num_batches = (len(labels) + batch_size - 1) // batch_size
  batches = [
      labels[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
  ]
  return batches


def _get_embedding_mean(
    batch: Tuple[str, Iterable[np.ndarray]]
) -> Tuple[str, np.ndarray]:
  key, embeddings = batch
  return key, np.mean(np.concatenate(embeddings, axis=0), axis=0)


def _write_embedding_mean(
    batch: Tuple[str, Iterable[np.ndarray]],
    positive_output_path: str,
    negative_output_path: str,
):
  """Writes embedding mean to file."""
  key, embeddings = batch
  mean = np.mean(np.concatenate(embeddings, axis=0), axis=0)
  if key == 'pos':
    with tf.io.gfile.GFile(positive_output_path, 'wb') as f:
      np.save(f, mean)
  elif key == 'neg':
    with tf.io.gfile.GFile(negative_output_path, 'wb') as f:
      np.save(f, mean)
  else:
    raise ValueError(f'Unrecognized embedding key "{key}"')


def _run_text_tower_inference(
    root: beam.PCollection,
    positive_labels: list[str],
    negative_labels: list[str],
    model_checkpoint: str,
    positive_embedding_path: str,
    negative_embedding_path: str):
  """Runs text tower inference.

  Args:
    root: Beam Collection.
    positive_labels: List of positive labels.
    negative_labels: List of negative labels.
    model_checkpoint: File path for the text tower model.
    positive_embedding_path: Path to file in which the positive embedding will
      be saved.
    negative_embedding_path: Path to file in which the negative embedding will
      be saved.

  Returns:
    text_embeddings: Dict that hold the positive embedding mean and the
        negative embedding mean.
  """
  labels = [
      ('pos', positive_label)
      for positive_label in _do_batch(positive_labels, 32)
  ]
  labels += [
      ('neg', negative_label)
      for negative_label in _do_batch(negative_labels, 32)
  ]
  _ = (
      root
      | 'Read labels' >> beam.Create(labels)
      | 'Generate label embeddings'
      >> beam.ParDo(TextTowerInference(model_checkpoint))
      | 'Group embeddings by pos/neg' >> beam.GroupByKey()
      | 'Calculate the embedding mean' >> beam.Map(
          _write_embedding_mean,
          positive_embedding_path,
          negative_embedding_path,
      )
  )


def _get_text_embeddings(
    pipeline_options: beam.PCollection,
    positive_labels_filepath: list[str],
    negative_labels_filepath: list[str],
    model_checkpoint: str,
    positive_embedding_path: str,
    negative_embedding_path: str,
) -> dict[str, np.ndarray]:
  """Get text embeddings.

  Args:
    pipeline_options: Dataflow pipeline options.
    positive_labels_filepath: File path to a text file containing positive
        labels.
    negative_labels_filepath: File path to a text file containing negative
        labels.
    model_checkpoint: File path for the text tower model.
    positive_embedding_path: Path to file in which the positive embedding will
      be saved.
    negative_embedding_path: Path to file in which the negative embedding will
      be saved.

  Returns:
    text_embeddings: Dict that hold the positive embedding mean and the
        negative embedding mean.
  """
  with tf.io.gfile.GFile(positive_labels_filepath, 'r') as f:
    positive_labels = [label.strip() for label in f.readlines()]

  with tf.io.gfile.GFile(negative_labels_filepath, 'r') as f:
    negative_labels = [label.strip() for label in f.readlines()]

  with beam.Pipeline(options=pipeline_options) as root:
    _run_text_tower_inference(
        root,
        positive_labels,
        negative_labels,
        model_checkpoint,
        positive_embedding_path,
        negative_embedding_path,
    )
  with tf.io.gfile.GFile(positive_embedding_path, 'rb') as f:
    positive_embeddings = np.load(f)
  with tf.io.gfile.GFile(negative_embedding_path, 'rb') as f:
    negative_embeddings = np.load(f)
  return {'pos': positive_embeddings, 'neg': negative_embeddings}


def run_tf2_inference_with_csv_output(
    examples_pattern: str,
    image_model_dir: str,
    text_model_dir: str,
    output_path: str,
    image_size: int,
    post_image_only: bool,
    batch_size: int,
    positive_labels_filepath: list[str] | None,
    negative_labels_filepath: list[str] | None,
    model_type: ModelType,
    threshold: float,
    high_precision_threshold: float,
    high_recall_threshold: float,
    generate_embeddings: bool,
    pipeline_options,
):
  """Runs example generation pipeline using TF2 model and outputs to CSV.

  Args:
    examples_pattern: Pattern for input TFRecords.
    image_model_dir: Model directory for the image checkpoint.
    text_model_dir: Model directory for the text checkpoint.
    output_path: CSV output path.
    image_size: Image width and height.
    post_image_only: Model expects only post-disaster images.
    batch_size: Batch size.
    positive_labels_filepath: File path to a text file containing positive
        labels. The file path is required only when using VLM, otherwise it can
        be set to empty list or None.
    negative_labels_filepath: File path to a text file containing negative
        labels. The file path is required only when using VLM, otherwise it can
        be set to empty list or None.
    model_type: Indentify the type of the model being used.
    threshold: Damaged score threshold.
    high_precision_threshold: Damaged score threshold for high precision.
    high_recall_threshold: Damaged score threshold for high recall.
    generate_embeddings: Generate embeddings.
    pipeline_options: Dataflow pipeline options.
  """

  tempfile_prefix = f'{output_path}.tmp/output'

  if model_type == ModelType.VLM:
    positive_embedding_path = f'{output_path}.positive_label_embedding.npy'
    negative_embedding_path = f'{output_path}.negative_label_embedding.npy'
    text_embeddings = _get_text_embeddings(
        pipeline_options,
        positive_labels_filepath,
        negative_labels_filepath,
        text_model_dir,
        positive_embedding_path,
        negative_embedding_path,
    )
  else:
    text_embeddings = None

  with beam.Pipeline(options=pipeline_options) as pipeline:
    examples = (
        pipeline
        | 'read_tfrecords'
        >> beam.io.tfrecordio.ReadFromTFRecord(
            examples_pattern, coder=beam.coders.ProtoCoder(tf.train.Example)
        )
        | 'reshuffle_input' >> beam.Reshuffle()
    )
    model = TF2InferenceModel(
        image_model_dir,
        image_size,
        post_image_only,
        model_type,
        text_embeddings,
    )
    inference_feature = 'embedding' if generate_embeddings else 'score'
    scored_examples = run_inference(
        examples, inference_feature, batch_size, model
    )
    if generate_embeddings:
      embeddings_examples_to_csv(scored_examples, tempfile_prefix)
    else:
      examples_to_csv(
          scored_examples,
          threshold,
          high_precision_threshold,
          high_recall_threshold,
          tempfile_prefix,
      )

  postprocess(tempfile_prefix, output_path, inference_feature)
