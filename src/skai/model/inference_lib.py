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

import collections
import enum
import math
import time
from typing import Any, Iterable, Iterator, NamedTuple

import apache_beam as beam
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

# Coordinate reference system using Longitude and Latitude.
_EPSG_4326 = 'EPSG:4326'

# Minimum ratio of two overlapping footprints to be considered duplicates.
_POST_FOOTPRINT_DEDUP_AREA_RATIO_THRESHOLD = 0.5

# Minimum ratio of the overlap area to the footprint area to be considered
# duplicates.
_POST_FOOTPRINT_DEDUP_OVERLAP_THRESHOLD = 0.5


class ModelType(enum.Enum):
  """Model types."""

  VLM = 'vlm'
  CLASSIFICATION = 'classification'


class InferenceRow(NamedTuple):
  """A row in the inference output CSV."""
  example_id: str | None
  int64_id: int | None
  building_id: str | None
  longitude: float | None
  latitude: float | None
  score: float | None
  plus_code: str | None
  area_in_meters: float | None
  footprint_wkt: str | None
  post_footprint_wkt: str | None
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
  ):
    self._model_dir = model_dir
    self._image_size = image_size
    self._post_image_only = post_image_only
    self._model = None

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
    keyed_examples: tuple[str, Iterable[tf.train.Example]],
    post_image_order: list[str],
) -> tf.train.Example:
  """Merges examples sharing the same coordinates.

  The score of the output example is the average of all example scores.

  Args:
    keyed_examples: Tuple of encoded coordinates and examples.
    post_image_order: List of post-disaster image ids in descending priority
      order.

  Returns:
    Example with merged scores or embeddings.
  """

  def _sort_key(example: tf.train.Example) -> tuple[int, int, str]:
    post_image_id = utils.get_bytes_feature(example, 'post_image_id')[
        0
    ].decode()
    try:
      order = post_image_order.index(post_image_id)
    except ValueError:
      order = len(post_image_order)

    pre_image_id = utils.get_bytes_feature(example, 'pre_image_id')[0].decode()
    if 'building_image_id' in example.features.feature:
      building_image_id = utils.get_bytes_feature(example, 'building_image_id')[
          0
      ].decode()
    else:
      building_image_id = None
    return (0 if pre_image_id == building_image_id else 1, order, post_image_id)

  examples = list(keyed_examples[1])
  examples.sort(key=_sort_key)
  output_example = tf.train.Example()
  output_example.CopyFrom(examples[0])
  try:  # scores
    scores = [utils.get_float_feature(e, 'score')[0] for e in examples]
    output_example.features.feature['score'].float_list.value[:] = [
        np.mean(scores)
    ]
  except IndexError:  # embeddings
    embeddings = [utils.get_float_feature(e, 'embedding') for e in examples]
    output_example.features.feature['embedding'].float_list.value.extend(
        np.mean(np.array(embeddings), axis=0)
    )
  return output_example


def _dedup_scored_examples(
    examples: beam.PCollection, post_image_order: list[str]
) -> beam.PCollection:
  """Deduplicates examples by merging those sharing the same coordinates.

  Args:
    examples: PCollection of examples with scores.
    post_image_order: List of post-disaster image ids in descending priority
      order.

  Returns:
    PCollection of deduplicated examples.
  """
  return (
      examples
      | 'key_examples_by_coords'
      >> beam.Map(_key_example_by_encoded_coordinates)
      | 'group_by_coords' >> beam.GroupByKey()
      | 'merge_examples' >> beam.Map(_merge_examples, post_image_order)
  )


def _example_id_embeddings(examples: beam.PCollection) -> beam.PCollection:
  return (
      examples
      | 'example_id_embeddings' >> beam.Map(_get_example_ids_and_embeddings)
  )


def _get_example_ids_and_embeddings(
    example: tf.train.Example
) -> tuple[str, np.ndarray]:
  try:
    example_id = utils.get_bytes_feature(example, 'int64_id')[0].decode()
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
    deduplicate: bool,
    post_image_order: list[str],
) -> beam.PCollection:
  """Runs inference and augments input examples with inference scores.

  Args:
    examples: PCollection of Tensorflow Examples.
    score_feature: Feature name to use for inference scores.
    batch_size: Batch size.
    model: Inference model to use.
    deduplicate: If true, examples of the same building are merged.
    post_image_order: List of post-disaster image ids in descending priority
      order for use in deduplicating examples.

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
  if deduplicate:
    return _dedup_scored_examples(scored_examples, post_image_order)
  return scored_examples


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


def _wkb_to_wkt(wkb: bytes) -> str:
  return shapely.wkt.dumps(shapely.wkb.loads(wkb))


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

  example_id = utils.get_bytes_feature(example, 'example_id')[0].decode()
  int64_id = utils.get_int64_feature(example, 'int64_id')[0]
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
    footprint_wkt = _wkb_to_wkt(footprint_wkb)
  except IndexError:
    footprint_wkt = None
  try:
    post_footprint_wkb = utils.get_bytes_feature(example, 'post_footprint_wkb')[
        0
    ]
    post_footprint_wkt = _wkb_to_wkt(post_footprint_wkb)
  except IndexError:
    post_footprint_wkt = None

  try:
    label = utils.get_float_feature(example, 'label')[0]
  except IndexError:
    label = None

  return InferenceRow(
      label=label,
      example_id=example_id,
      int64_id=int64_id,
      building_id=building_id,
      longitude=longitude,
      latitude=latitude,
      score=score,
      plus_code=plus_code,
      area_in_meters=area,
      footprint_wkt=footprint_wkt,
      post_footprint_wkt=post_footprint_wkt,
      damaged=(score >= threshold),
      damaged_high_precision=(score >= high_precision_threshold),
      damaged_high_recall=(score >= high_recall_threshold)
  )


def write_examples_to_files(
    examples: beam.PCollection,
    threshold: float,
    high_precision_threshold: float,
    high_recall_threshold: float,
    output_prefix: str,
) -> None:
  """Writes examples to CSV and GeoPackage files.

  Args:
    examples: PCollection of Tensorflow Examples.
    threshold: Damaged score threshold.
    high_precision_threshold: Damaged score threshold for high precision.
    high_recall_threshold: Damaged score threshold for high recall.
    output_prefix: Path prefix for output files.
  """
  _ = (
      examples
      | 'reshuffle_for_output' >> beam.Reshuffle()
      | 'examples_to_rows'
      >> beam.Map(
          example_to_row,
          threshold=threshold,
          high_precision_threshold=high_precision_threshold,
          high_recall_threshold=high_recall_threshold,
      )
      | 'combine_rows' >> beam.transforms.combiners.ToList()
      | 'write_output' >> beam.Map(write_outputs, output_prefix=output_prefix)
  )


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


def _to_geodataframe(
    df: pd.DataFrame, footprint_column: str
) -> gpd.GeoDataFrame:
  """Converts a DataFrame to a GeoDataFrame."""
  geometries = []
  for _, row in df.iterrows():
    wkt = row[footprint_column]
    if not isinstance(wkt, str) and math.isnan(wkt):
      geometries.append(
          shapely.geometry.Point(row['longitude'], row['latitude'])
      )
    else:
      geometries.append(shapely.wkt.loads(wkt))
  return gpd.GeoDataFrame(
      df.drop(columns=[footprint_column]), geometry=geometries, crs=_EPSG_4326
  )


def _dedup_post_footprints(
    predictions: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
  """Deduplicate inference footprints that have been aligned with post imagery.

  Args:
    predictions: GeoDataFrame containing inference results.

  Returns:
    A copy of the predictions GeoDataFrame with 2 additional columns: duplicate
    and deduped_score.
  """
  utm_crs = utils.get_utm_crs(
      predictions['longitude'].mean(), predictions['latitude'].mean()
  )
  utm_gdf = predictions.to_crs(utm_crs)
  utm_gdf['area'] = utm_gdf.area
  other_gdf = gpd.GeoDataFrame(
      {
          'other_example_id': utm_gdf['example_id'],
          'other_area': utm_gdf['area'],
          'other_score': utm_gdf['score'],
      },
      geometry=utm_gdf.geometry,
  )
  intersections = utm_gdf.overlay(
      other_gdf, how='intersection', keep_geom_type=False
  )
  # Remove self-intersections and reflections.
  intersections = intersections[
      intersections['example_id'] < intersections['other_example_id']
  ]

  # Remove intersections that are not a significant portion of the footprints
  # and intersections between footprints that are very different in size.
  min_areas = intersections[['area', 'other_area']].min(axis=1)
  max_areas = intersections[['area', 'other_area']].max(axis=1)
  intersections['area_ratio'] = min_areas / max_areas
  intersections['overlap_ratio'] = intersections.area / intersections['area']
  duplicates = intersections[
      (intersections['area_ratio'] > _POST_FOOTPRINT_DEDUP_AREA_RATIO_THRESHOLD)
      & (
          intersections['overlap_ratio']
          > _POST_FOOTPRINT_DEDUP_OVERLAP_THRESHOLD
      )
  ].copy()
  duplicates.sort_values(by='overlap_ratio', ascending=False, inplace=True)
  discarded_examples = set()
  damage_scores = collections.defaultdict(list)
  for _, row in predictions.iterrows():
    damage_scores[row['example_id']].append(row['score'])
  for _, row in duplicates.iterrows():
    example_id = row['example_id']
    other_example_id = row['other_example_id']
    if (
        example_id in discarded_examples
        or other_example_id in discarded_examples
    ):
      continue
    # TODO(jzxu): The example to keep should be the one with higher building
    # confidence, but that information is currently not piped into the
    # inference output. Arbitrarily choose the building with the smaller area
    # for now.
    damage_scores[example_id].append(row['other_score'])
    discarded_examples.add(other_example_id)

  predictions_copy = predictions.copy()
  predictions_copy['deduped_score'] = [
      np.mean(damage_scores[i]) for i in predictions_copy['example_id']
  ]
  predictions_copy['duplicate'] = [
      i in discarded_examples for i in predictions_copy['example_id']
  ]
  return predictions_copy


def _write_geopackage_files(
    inference_df: pd.DataFrame,
    output_prefix: str,
) -> None:
  """Writes inference results as GeoPackage files.

  Writes out two files: one using non-post-aligned footprints, and the other
  using post-aligned footprints.

  Args:
    inference_df: Inference results dataframe.
    output_prefix: Output path prefix.
  """
  geometries = []
  post_geometries = []
  for _, row in inference_df.iterrows():
    longitude = row['longitude']
    latitude = row['latitude']
    if 'footprint_wkt' not in row or row['footprint_wkt'] is None:
      footprint = shapely.geometry.Point(longitude, latitude)
    else:
      footprint = shapely.wkt.loads(row['footprint_wkt'])
    geometries.append(footprint)
    if 'post_footprint_wkt' not in row or row['post_footprint_wkt'] is None:
      post_footprint = shapely.geometry.Point(longitude, latitude)
    else:
      post_footprint = shapely.wkt.loads(row['post_footprint_wkt'])
    post_geometries.append(post_footprint)

  df_without_footprints = inference_df.drop(
      columns=[
          c
          for c in ('footprint_wkt', 'post_footprint_wkt')
          if c in inference_df.columns
      ]
  )

  gdf = gpd.GeoDataFrame(
      df_without_footprints,
      geometry=geometries,
      crs='EPSG:4326',
  )
  if output_prefix.endswith('.csv'):
    output_prefix = output_prefix[:-4]
  with tf.io.gfile.GFile(f'{output_prefix}.gpkg', 'wb') as f:
    gdf.to_file(f, driver='GPKG', engine='fiona')

  post_gdf = _dedup_post_footprints(
      gpd.GeoDataFrame(
          df_without_footprints,
          geometry=post_geometries,
          crs='EPSG:4326',
      )
  )
  with tf.io.gfile.GFile(f'{output_prefix}_post.gpkg', 'wb') as f:
    post_gdf.to_file(f, driver='GPKG', engine='fiona')


def write_outputs(
    rows: list[InferenceRow],
    output_prefix: str,
) -> None:
  """Writes inference results to files.

  Args:
    rows: List of InferenceRows.
    output_prefix: Output prefix.
  """
  df = pd.DataFrame(rows)
  with tf.io.gfile.GFile(f'{output_prefix}.csv', 'w') as f:
    df.to_csv(f, index=False)
  footprint_columns = [
      c for c in ['footprint_wkt', 'post_footprint_wkt'] if c in df.columns
  ]
  if footprint_columns:
    # Also output a version of the CSV with no footprints. The footprints_wkt
    # column is the most memory intensive and sometimes prevents the CSV from
    # being loaded by QGIS if there are too many examples.
    df_no_footprints = df.drop(columns=footprint_columns)
    output_path_no_footprints = f'{output_prefix}_no_footprints.csv'
    with tf.io.gfile.GFile(output_path_no_footprints, 'w') as f:
      df_no_footprints.to_csv(f, index=False)

  if 'GPKG' in fiona.supported_drivers:
    _write_geopackage_files(df, output_prefix)


def _do_batch(labels: list[str], batch_size: int) -> list[list[str]]:
  """batch labels."""
  num_batches = (len(labels) + batch_size - 1) // batch_size
  batches = [
      labels[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
  ]
  return batches


def _get_embedding_mean(
    batch: tuple[str, Iterable[np.ndarray]]
) -> tuple[str, np.ndarray]:
  key, embeddings = batch
  return key, np.mean(np.concatenate(embeddings, axis=0), axis=0)


def _write_embedding_mean(
    batch: tuple[str, Iterable[np.ndarray]],
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


def run_tf2_inference_with_csv_output(
    examples_pattern: str,
    image_model_dir: str,
    output_prefix: str,
    image_size: int,
    post_image_only: bool,
    batch_size: int,
    threshold: float,
    high_precision_threshold: float,
    high_recall_threshold: float,
    deduplicate: bool,
    post_image_order: list[str],
    generate_embeddings: bool,
    wait_for_dataflow: bool,
    pipeline_options: beam.options.pipeline_options.PipelineOptions,
) -> None:
  """Runs example generation pipeline using TF2 model and outputs to CSV.

  Args:
    examples_pattern: Pattern for input TFRecords.
    image_model_dir: Model directory for the image checkpoint.
    output_prefix: Path prefix for output files.
    image_size: Image width and height.
    post_image_only: Model expects only post-disaster images.
    batch_size: Batch size.
    threshold: Damaged score threshold.
    high_precision_threshold: Damaged score threshold for high precision.
    high_recall_threshold: Damaged score threshold for high recall.
    deduplicate: If true, examples of the same building are merged.
    post_image_order: List of post-disaster image ids in descending priority
      order for use in deduplicating examples.
    generate_embeddings: Generate embeddings.
    wait_for_dataflow: If true, wait for the Dataflow job to complete.
    pipeline_options: Dataflow pipeline options.
  """
  pipeline = beam.Pipeline(options=pipeline_options)
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
      post_image_only
  )
  inference_feature = 'embedding' if generate_embeddings else 'score'
  scored_examples = run_inference(
      examples,
      inference_feature,
      batch_size,
      model,
      deduplicate,
      post_image_order,
  )
  if generate_embeddings:
    embeddings_examples_to_csv(scored_examples, output_prefix)
  else:
    write_examples_to_files(
        scored_examples,
        threshold,
        high_precision_threshold,
        high_recall_threshold,
        output_prefix,
    )

  result = pipeline.run()
  if wait_for_dataflow:
    result.wait_until_finish()
