# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for performing data labeling."""

import collections
import concurrent.futures
import dataclasses
import functools
import multiprocessing
import os
import random
from typing import Any, Callable, Iterable

from absl import logging
import geopandas as gpd
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import scipy
from skai import utils
import tensorflow as tf
import tqdm

# Gap to add between before and after images.
BEFORE_AFTER_GAP = 10

# Margin around caption text.
CAPTION_MARGIN = 32

# Size of the reticule that identifies the building being labeled.
RETICULE_HALF_LEN = 32


Example = tf.train.Example
Image = PIL.Image.Image


@dataclasses.dataclass(frozen=True)
class LabelingExample:
  """Information about an example chosen for labeling.
  """
  int64_id: int
  example_id: str
  pre_image_path: str
  post_image_path: str
  combined_image_path: str
  tfrecord_path: str
  serialized_example: bytes
  longitude: float
  latitude: float


def _annotate_image(image: Image, caption: str) -> Image:
  """Adds center square and caption to image.

  Args:
    image: Input image.
    caption: Caption text to add.

  Returns:
    A copy of the input image with annotations added.
  """
  # Copy image into bigger frame to have room for caption.
  annotated_image = PIL.Image.new('RGB',
                                  (image.width, image.height + CAPTION_MARGIN),
                                  (225, 225, 225))
  annotated_image.paste(image, (0, 0))

  # Draw center rectangle.
  cx = image.width // 2
  cy = image.height // 2
  coords = [(cx - RETICULE_HALF_LEN, cy - RETICULE_HALF_LEN),
            (cx + RETICULE_HALF_LEN, cy + RETICULE_HALF_LEN)]
  annotations = PIL.ImageDraw.Draw(annotated_image)
  annotations.rectangle(coords, outline=(255, 0, 0), width=1)

  # Add caption.
  caption_xy = (cx, image.height + 5)
  annotations.text(caption_xy, caption, fill='black', anchor='mt')
  return annotated_image


def _read_example_ids_from_import_file(path: str) -> Iterable[str]:
  with tf.io.gfile.GFile(path, 'r') as import_file:
    df = pd.read_csv(import_file)
    return df['example_id']


def create_labeling_image(
    before_image: Image, after_image: Image, example_id: str, plus_code: str
) -> Image:
  """Creates an image used for labeling.

  The image is composed of the before and after images from the input example
  side-by-side.

  Args:
    before_image: Before image.
    after_image: After image.
    example_id: Example id.
    plus_code: Plus code.

  Returns:
    Annotated and combined image.

  """
  before_annotated = _annotate_image(before_image, 'BEFORE')
  after_annotated = _annotate_image(after_image, 'AFTER')
  width = before_annotated.width + after_annotated.width + 3 * BEFORE_AFTER_GAP
  height = before_annotated.height + 2 * BEFORE_AFTER_GAP
  combined = PIL.Image.new('RGB', (width, height), (225, 225, 225))
  combined.paste(before_annotated, (BEFORE_AFTER_GAP, BEFORE_AFTER_GAP))
  combined.paste(after_annotated,
                 (before_annotated.width + 2 * BEFORE_AFTER_GAP,
                  BEFORE_AFTER_GAP))
  caption = PIL.ImageDraw.Draw(combined)
  bottom_text = f'Example id: {example_id}   Plus code: {plus_code}'
  caption.text(
      (10, combined.height - 10),
      bottom_text,
      fill='black',
      anchor='lb',
  )
  return combined


def get_diffuse_subset(
    points: gpd.GeoDataFrame, buffer_meters: float
) -> gpd.GeoDataFrame:
  """Returns an arbitrary subset of points that are far away from each other.

    Points are kept or dropped based on the row order in the dataframe, so it's
    important for the input to already be randomly shuffled.

  Args:
    points: Points to drop neighbors from.
    buffer_meters: Buffer size in meters.

  Returns:
    Points with neighbors dropped.
  """
  buffers = gpd.GeoDataFrame(geometry=points.geometry.buffer(buffer_meters))
  joined = gpd.GeoDataFrame(geometry=points.geometry).sjoin(
      buffers, how='left'
  )['index_right']
  indexes_to_keep = []
  indexes_to_drop = set()
  for index, nearby_indexes in joined.groupby(level=0):
    if index in indexes_to_drop:
      continue
    indexes_to_keep.append(index)
    indexes_to_drop.update(nearby_indexes.values)
  return points.loc[indexes_to_keep]


def merge_dropping_neighbors(
    points: gpd.GeoDataFrame, new_points: gpd.GeoDataFrame, buffer_meters: float
) -> gpd.GeoDataFrame:
  """Merges new_points into points, dropping neighbors if necessary.

  Args:
    points: Points to merge into.
    new_points: New points to merge into points.
    buffer_meters: Buffer size in meters.

  Returns:
    Merged points.
  """
  buffer_df = gpd.GeoDataFrame(geometry=points.buffer(buffer_meters))
  joined = new_points.sjoin(buffer_df, how='inner')
  indexes_to_drop = list(set(joined.index))
  return pd.concat([points, new_points.drop(indexes_to_drop)])


def sample_with_buffer(
    points: gpd.GeoDataFrame,
    num_points: int,
    buffer_meters: float,
    starting_sample: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
  """Samples num_points from points, dropping neighbors if necessary.

  Args:
    points: Points to sample from.
    num_points: Number of points to sample.
    buffer_meters: Buffer size in meters.
    starting_sample: Points to start sampling from.

  Returns:
    GeoDataFrame containing sampled points.
  """
  unsampled = points.copy()
  if starting_sample is None:
    s = unsampled.sample(min(num_points, len(unsampled)))
    unsampled.drop(s.index, inplace=True)
    sample = get_diffuse_subset(s, buffer_meters)
    target_size = num_points
  else:
    target_size = len(starting_sample) + num_points
    sample = starting_sample.copy()
  while not unsampled.is_empty.all() and len(sample) < target_size:
    num_needed = target_size - len(sample)
    s = unsampled.sample(min(num_needed, len(unsampled)))
    unsampled.drop(s.index, inplace=True)
    s = get_diffuse_subset(s, buffer_meters)
    sample = merge_dropping_neighbors(sample, s, buffer_meters)
  return sample


def _read_sharded_metadata(pattern: str) -> pd.DataFrame:
  """Reads sharded metadata files matching pattern and merges them."""
  paths = tf.io.gfile.glob(pattern)
  if not paths:
    raise ValueError(f'File pattern {pattern} did not match any files.')
  dfs = []
  for path in paths:
    if '.parquet' in path:
      with tf.io.gfile.GFile(path, 'rb') as f:
        f.closed = False
        df = pd.read_parquet(f)
    elif '.csv' in path:
      with tf.io.gfile.GFile(path, 'r') as f:
        df = pd.read_csv(f)
    else:
      raise ValueError(f'Unsupported metadata file type: {path}')
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


def _read_sharded_csvs(pattern: str) -> pd.DataFrame:
  """Reads CSV shards matching pattern and merges them."""
  paths = tf.io.gfile.glob(pattern)
  if not paths:
    raise ValueError(f'File pattern {pattern} did not match any files.')
  dfs = []
  expected_columns = None
  for path in paths:
    with tf.io.gfile.GFile(path, 'r') as f:
      df = pd.read_csv(f)
      if expected_columns is None:
        expected_columns = set(df.columns)
      else:
        actual_columns = set(df.columns)
        if actual_columns != expected_columns:
          raise ValueError(f'Inconsistent columns in file {path}')
      dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


def get_buffered_example_ids(
    metadata_pattern: str,
    buffered_sampling_radius: float,
    excluded_example_ids: set[str],
    max_examples: int | None = None,
) -> set[str]:
  """Gets a set of allowed example ids.

  Args:
    metadata_pattern: File pattern for input metadata files.
    buffered_sampling_radius: The minimum distance between two examples for the
      two examples to be in the labeling task.
    excluded_example_ids: Set of example ids to be excluded prior to generating
      buffered example ids.
    max_examples: Maximum number of images to create.
  Returns:
    Set of allowed example ids.
  """
  metadata = _read_sharded_metadata(metadata_pattern)
  metadata = metadata[
      ~metadata['example_id'].isin(excluded_example_ids)
  ].reset_index(drop=True)

  logging.info(
      'Randomly searching for buffered samples with buffer radius %.2f'
      ' metres...',
      buffered_sampling_radius,
  )
  points = utils.convert_to_utm(
      gpd.GeoDataFrame(
          {
              'geometry': gpd.points_from_xy(
                  metadata['longitude'], metadata['latitude']
              )
          },
          crs=4326,
      )
  )['geometry']
  gpd_df = gpd.GeoDataFrame(metadata, geometry=points)
  max_examples = len(gpd_df) if max_examples is None else max_examples
  df_buffered_samples = sample_with_buffer(
      gpd_df, max_examples, buffered_sampling_radius
  )
  allowed_example_ids = set(df_buffered_samples['example_id'].unique())
  logging.info(
      'Allowing %d example ids with buffer radius: %.2f metres',
      len(allowed_example_ids),
      buffered_sampling_radius,
  )

  return allowed_example_ids


def _deduplicate_labeling_examples(
    examples: list[LabelingExample],
) -> list[LabelingExample]:
  example_ids = set()
  deduped_examples = []
  for example in examples:
    if example.example_id not in example_ids:
      example_ids.add(example.example_id)
      deduped_examples.append(example)
  return deduped_examples


def create_labeling_images(
    metadata_pattern: str,
    images_dir: str | None,
    examples_pattern: str | None,
    max_images: int,
    allowed_example_ids_path: str,
    excluded_import_file_patterns: list[str],
    output_dir: str,
    use_multiprocessing: bool,
    multiprocessing_context: Any,
    max_processes: int,
    buffered_sampling_radius: float,
    score_bins_to_sample_fraction: dict[tuple[float, float], float],
    scores_path: str | None,
    filter_by_column: str | None,
) -> int:
  """Samples labeling examples and creates PNG images for the labeling task.

  Example sampling works in 3 possible modes, depending on the parameters
  specified:
  - If allowed_example_ids_path is provided, then choose the examples with the
    example ids specified in this file. No random sampling occurs.
  - If scores_path is provided, then read the scores for examples from this file
    and sample the examples based on their scores. The rate at which different
    score ranges are sampled can be specified in the
    score_bins_to_sample_fraction parameter.
  - If neither of the above parameters are provided, then sample uniformly from
    all examples.

  For all 3 sampling modes, the parameter excluded_import_file_patterns can be
  provided to prevent sampling examples with specific example ids.

  This function will write 3 kinds of files to the output directory:
  - A set of PNG images for use in the labeling task. The files will be named
    after the example id of each example.
  - A text file containing the absolute paths of all the generated PNG files.
    This file can be used as the "import file" to specify which images to
    upload to the Vertex AI labeling service. This file will be named
    "import_file.csv", although it is not actually in CSV format, as it does not
    contain a header.
  - A CSV file with metadata information about each chosen example. The columns
    in this file are:
    - id: The int64 id of the example.
    - int64_id: The int64 id of the example (repeated for convenience).
    - example_id: The string example id of the example.
    - image: The path to the image file, in the format "file://<filepath>".
    - image_source_path: The path to the image file, without the "file://"
      prefix. Repeated for convenience.
    - tfrecord_source_path: Path to the TFRecord file that contains the example.

  Args:
    metadata_pattern: The file pattern of either CSV or parquet example metadata
      files created by example generation pipeline.
    images_dir: Path to directory containing pre/post PNG images. If not none,
      this directory will be preferred over the examples pattern.
    examples_pattern: File pattern for input TFRecords or parquet files
      containing examples. Only used if images_dir is None.
    max_images: Maximum number of images to create.
    allowed_example_ids_path: Path to CSV containing example ids that are
      allowed to be in the labeling set. The CSV must have an "example_id"
      column. Other columns are ignored.
    excluded_import_file_patterns: List of import file patterns containing
      images to exclude.
    output_dir: Output directory.
    use_multiprocessing: If true, create multiple processes to create labeling
      images.
    multiprocessing_context: Context to spawn processes with when using
      multiprocessing.
    max_processes: Maximum number of processes.
    buffered_sampling_radius: The minimum distance between two examples for the
      two examples to be in the labeling task.
    score_bins_to_sample_fraction: Dictionary of bins for selecting labeling
      examples.
    scores_path: File containing scores obtained from pre-trained models.
    filter_by_column: If specified, the name of the column in the scores CSV
      file to use as a filter. The column must contain binary values, either
      true/false or 0/1. Rows with positive values in this column are then
      filtered out.

  Returns:
    Number of images written.
  """
  if scores_path and allowed_example_ids_path:
    raise ValueError(
        'scores_path and allowed_example_ids_path cannot be set at the same'
        ' time.'
    )
  excluded_example_ids = set()
  if excluded_import_file_patterns:
    for pattern in excluded_import_file_patterns:
      for path in tf.io.gfile.glob(pattern):
        logging.info('Excluding example ids from "%s"', path)
        excluded_example_ids.update(_read_example_ids_from_import_file(path))
    logging.info('Excluding %d example ids', len(excluded_example_ids))

  if allowed_example_ids_path:
    with tf.io.gfile.GFile(allowed_example_ids_path, 'r') as f:
      allowed_ids_df = pd.read_csv(f)
      if 'example_id' in allowed_ids_df.columns:
        allowed_example_ids = set(allowed_ids_df['example_id'])
      else:
        raise ValueError(
            'allowed_example_ids_path must contain a column named "example_id"'
        )
  elif scores_path:
    allowed_example_ids = []
    with tf.io.gfile.GFile(scores_path, 'r') as f:
      scores_df = pd.read_csv(f)
      if 'damage_score' not in scores_df.columns:
        raise ValueError(f'damage_score column not found in {scores_path}')
      if filter_by_column:
        if filter_by_column not in scores_df.columns:
          raise ValueError(
              f'{filter_by_column} column not found in {scores_path}'
          )
        scores_df = scores_df[scores_df[filter_by_column] == 0]
      scores_df = scores_df[~scores_df['example_id'].isin(excluded_example_ids)]

    num_total_images = min(max_images, len(scores_df))
    for (
        low,
        high,
    ), percentage_examples in score_bins_to_sample_fraction.items():
      num_examples = int(percentage_examples * num_total_images)
      df_example_ids = scores_df[
          (scores_df['damage_score'] >= low)
          & (scores_df['damage_score'] < high)
      ]
      example_ids = list(df_example_ids['example_id'])
      random.shuffle(example_ids)
      allowed_example_ids.extend(example_ids[:num_examples])

  else:
    allowed_example_ids = get_buffered_example_ids(
        metadata_pattern,
        buffered_sampling_radius,
        excluded_example_ids,
        max_images,
    )

  if images_dir:
    labeling_examples = _create_labeling_assets_from_metadata(
        metadata_pattern, images_dir, output_dir, allowed_example_ids
    )
  else:
    if examples_pattern is None:
      raise ValueError('examples_pattern must be set if images_dir is None.')
    example_files = tf.io.gfile.glob(examples_pattern)
    if not example_files:
      raise ValueError(
          f'Example pattern {examples_pattern} did not match any files.'
      )
    if all(f.endswith('.parquet') for f in example_files):
      labeling_examples = _create_labeling_assets_from_parquet_files(
          example_files, output_dir, allowed_example_ids
      )
    else:
      labeling_examples = _process_example_files(
          example_files,
          output_dir,
          use_multiprocessing,
          multiprocessing_context,
          max_processes,
          allowed_example_ids,
          _create_labeling_assets_from_example_file,
      )
  labeling_examples = _deduplicate_labeling_examples(labeling_examples)

  image_metadata_csv = os.path.join(
      output_dir, 'image_metadata.csv'
  )

  image_metadata_df = pd.DataFrame([
      {
          'id': i.int64_id,
          'int64_id': i.int64_id,
          'example_id': i.example_id,
          'image': (
              f'file://{i.combined_image_path.replace("gs://", "/bigstore/")}'
          ),
          'image_source_path': i.combined_image_path,
          'pre_image_path': i.pre_image_path,
          'post_image_path': i.post_image_path,
          'tfrecord_source_path': i.tfrecord_path,
          'longitude': i.longitude,
          'latitude': i.latitude,
      }
      for i in labeling_examples
  ])
  with tf.io.gfile.GFile(image_metadata_csv, 'w') as f:
    image_metadata_df.to_csv(f, index=False)

  import_file_csv = os.path.join(output_dir, 'import_file.csv')
  with tf.io.gfile.GFile(import_file_csv, 'w') as f:
    f.write('\n'.join([e.combined_image_path for e in labeling_examples]))

  tfrecord_path = os.path.join(output_dir, 'labeling_examples.tfrecord')
  with tf.io.TFRecordWriter(tfrecord_path) as writer:
    for labeling_example in labeling_examples:
      writer.write(labeling_example.serialized_example)

  return len(labeling_examples)


def create_buffered_tfrecords(
    metadata_pattern: str,
    examples_pattern: str,
    output_dir: str,
    use_multiprocessing: bool,
    multiprocessing_context: Any,
    excluded_import_file_patterns: list[str],
    max_processes: int,
    buffered_sampling_radius: float,
):
  """Creates filtered TFRecords.

  Args:
    metadata_pattern: File pattern for input metadata files.
    examples_pattern: File pattern for input TFRecords.
    output_dir: Output directory.
    use_multiprocessing: If true, create multiple processes to create labeling
      images.
    multiprocessing_context: Context to spawn processes with when using
      multiprocessing.
    excluded_import_file_patterns: List of import file patterns containing
      images to exclude.
    max_processes: Maximum number of processes.
    buffered_sampling_radius: The minimum distance between two examples for the
      two examples to be in the labeling task.

  """
  example_files = tf.io.gfile.glob(examples_pattern)
  if not example_files:
    raise ValueError(
        f'Example pattern {examples_pattern} did not match any files.'
    )

  excluded_example_ids = set()
  if excluded_import_file_patterns:
    for pattern in excluded_import_file_patterns:
      for path in tf.io.gfile.glob(pattern):
        logging.info('Excluding example ids from "%s"', path)
        excluded_example_ids.update(_read_example_ids_from_import_file(path))
    logging.info('Excluding %d example ids', len(excluded_example_ids))
  allowed_example_ids = get_buffered_example_ids(
      metadata_pattern, buffered_sampling_radius, excluded_example_ids
  )

  _ = _process_example_files(
      example_files,
      output_dir,
      use_multiprocessing,
      multiprocessing_context,
      max_processes,
      allowed_example_ids,
      filter_examples_from_allowed_ids,
  )

  logging.info(
      'Filtered out %d images to tfrecords to %s',
      len(allowed_example_ids),
      output_dir,
  )


def _process_example_files(
    example_files: str,
    output_dir: str,
    use_multiprocessing: bool,
    multiprocessing_context: Any,
    max_processes: int,
    allowed_example_ids: set[str],
    processing_function: Callable[[str, str, set[str]], list[Any]],
) -> list[Any]:
  """Run a processing function on a list of files.

  Supports processing the files in parallel using multi-processing or
  sequentially.

  Args:
    example_files: List of input TFRecords.
    output_dir: Output directory.
    use_multiprocessing: If true, create multiple processes to create labeling
      images.
    multiprocessing_context: Context to spawn processes with when using
      multiprocessing.
    max_processes: Maximum number of processes.
    allowed_example_ids: Set of example_id from which a subset will be used in
      for filtering or creating a labeling file.
    processing_function: Function to be executed.

  Returns:
    Concatenated list of results from running the processing function on each
    file.
  """
  all_results = []
  if use_multiprocessing:
    def accumulate(results: list[Any]) -> None:
      all_results.extend(results)
    num_workers = min(
        multiprocessing.cpu_count(), len(example_files), max_processes
    )
    if multiprocessing_context:
      pool = multiprocessing_context.Pool(num_workers)
    else:
      pool = multiprocessing.Pool(num_workers)
    for example_file in example_files:
      pool.apply_async(
          processing_function,
          args=(example_file, output_dir, allowed_example_ids),
          callback=accumulate,
          error_callback=print,
      )
    pool.close()
    pool.join()
  else:
    for example_file in example_files:
      all_results.extend(processing_function(
          example_file, output_dir, allowed_example_ids
      ))

  return all_results


def _tfrecord_iterator(path: str) -> Example:
  """Creates an iterator over TFRecord files.

  Supports both eager and non-eager execution.

  Args:
    path: Path to TFRecord file.

  Yields:
    Examples from the TFRecord file.
  """
  ds = tf.data.TFRecordDataset([path]).prefetch(tf.data.AUTOTUNE)
  if tf.executing_eagerly():
    for record in ds:
      example = Example()
      example.ParseFromString(record.numpy())
      yield example
  else:
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    next_element = iterator.get_next()
    with tf.compat.v1.Session() as sess:
      while True:
        try:
          value = sess.run(next_element)
        except tf.errors.OutOfRangeError:
          return
        example = Example()
        example.ParseFromString(value)
        yield example


def _create_labeling_assets_from_example_file(
    example_file: str,
    output_dir: str,
    allowed_example_ids: set[str],
) -> list[LabelingExample]:
  """Creates assets needed for a labeling task from TFRecords.

  Creates combined and separate pre/post PNGs used by labeling tools.

  Args:
    example_file: Path to file containing TF records.
    output_dir: Directory to write assets to.
    allowed_example_ids: Set of example_id from which a subset will be used in
      creating labeling task.

  Returns:
    List of LabelingExamples containing information about the created assets.
  """
  labeling_examples = []
  for example in _tfrecord_iterator(example_file):
    if 'example_id' in example.features.feature:
      example_id = (
          example.features.feature['example_id'].bytes_list.value[0].decode()
      )
    else:
      # If the example doesn't have an "example_id" feature, fall back on using
      # "encoded_coordinates". This maintains backwards compatibility with
      # older datasets.
      # TODO(jzxu): Remove this branch when backward compatibility is no longer
      # needed.
      example_id = (
          example.features.feature['encoded_coordinates']
          .bytes_list.value[0]
          .decode()
      )

    if example_id not in allowed_example_ids:
      continue

    try:
      int64_id = utils.get_int64_feature(example, 'int64_id')[0]
    except IndexError as error:
      raise ValueError('Examples do not have int64_id feature') from error

    if 'plus_code' in example.features.feature:
      plus_code = (
          example.features.feature['plus_code'].bytes_list.value[0].decode()
      )
    else:
      plus_code = 'unknown'

    longitude, latitude = example.features.feature[
        'coordinates'
    ].float_list.value

    before_image = utils.deserialize_image(
        example.features.feature['pre_image_png_large'].bytes_list.value[0],
        'png',
    )
    after_image = utils.deserialize_image(
        example.features.feature['post_image_png_large'].bytes_list.value[0],
        'png',
    )
    combined_image = create_labeling_image(
        before_image, after_image, example_id, plus_code
    )

    pre_image_path = os.path.join(output_dir, f'{example_id}_pre.png')
    with tf.io.gfile.GFile(pre_image_path, 'wb') as f:
      f.write(
          example.features.feature['pre_image_png_large'].bytes_list.value[0]
      )
    post_image_path = os.path.join(output_dir, f'{example_id}_post.png')
    with tf.io.gfile.GFile(post_image_path, 'wb') as f:
      f.write(
          example.features.feature['post_image_png_large'].bytes_list.value[0]
      )
    combined_image_path = os.path.join(output_dir, f'{example_id}.png')
    with tf.io.gfile.GFile(combined_image_path, 'wb') as f:
      f.write(utils.serialize_image(combined_image, 'png'))

    labeling_examples.append(
        LabelingExample(
            int64_id=int64_id,
            example_id=str(example_id),
            pre_image_path=pre_image_path,
            post_image_path=post_image_path,
            combined_image_path=combined_image_path,
            tfrecord_path=example_file,
            serialized_example=example.SerializeToString(),
            longitude=longitude,
            latitude=latitude,
        )
    )

  return labeling_examples


def _dataframe_row_to_example(row: pd.Series) -> bytes:
  """Converts a dataframe row into a serialized TF example.

  Args:
    row: The input row.

  Returns:
    Serialized TF Example with features populated from the row's column values.
  """
  example = tf.train.Example()
  utils.add_bytes_feature('example_id', row['example_id'].encode(), example)
  utils.add_bytes_feature(
      'encoded_coordinates', row['encoded_coordinates'].encode(), example
  )
  utils.add_bytes_feature('plus_code', row['plus_code'].encode(), example)
  utils.add_bytes_feature('pre_image_id', row['pre_image_id'].encode(), example)
  utils.add_bytes_feature(
      'post_image_id', row['post_image_id'].encode(), example
  )
  utils.add_bytes_feature('pre_image_png', row['pre_image_png'], example)
  utils.add_bytes_feature('post_image_png', row['post_image_png'], example)
  utils.add_bytes_feature(
      'pre_image_png_large', row['pre_image_png_large'], example
  )
  utils.add_bytes_feature(
      'post_image_png_large', row['post_image_png_large'], example
  )
  utils.add_int64_feature('int64_id', row['int64_id'], example)
  longitude, latitude = utils.decode_coordinates(row['encoded_coordinates'])
  utils.add_float_list_feature('coordinates', [longitude, latitude], example)
  return example.SerializeToString()


def _write_labeling_images_from_dataframe_row(
    row: pd.Series, output_dir: str
) -> tuple[str, str, str]:
  """Writes labeling images from a dataframe row.

  Args:
    row: Dataframe row containing images.
    output_dir: Output directory.

  Returns:
    Tuple of pre image path, post image path, combined image path.
  """
  example_id = row['example_id']
  pre_image_path = os.path.join(output_dir, f'{example_id}_pre.png')
  with tf.io.gfile.GFile(pre_image_path, 'wb') as f:
    f.write(row['pre_image_png_large'])
  post_image_path = os.path.join(output_dir, f'{example_id}_post.png')
  with tf.io.gfile.GFile(post_image_path, 'wb') as f:
    f.write(row['post_image_png_large'])

  before_image = utils.deserialize_image(
      row['pre_image_png_large'],
      'png',
  )
  after_image = utils.deserialize_image(
      row['post_image_png_large'],
      'png',
  )
  combined_image = create_labeling_image(
      before_image, after_image, example_id, row['plus_code']
  )
  combined_image_path = os.path.join(output_dir, f'{example_id}.png')
  with tf.io.gfile.GFile(combined_image_path, 'wb') as f:
    f.write(utils.serialize_image(combined_image, 'png'))
  return pre_image_path, post_image_path, combined_image_path


def _read_parquet(
    path: str, output_dir: str, allowed_example_ids: set[str]
) -> list[LabelingExample]:
  """Extracts labeling images from Parquet files.

  Args:
    path: Path to Parquet file.
    output_dir: Output directory.
    allowed_example_ids: Set of example ids that should be included.

  Returns:
    List of LabelingExample obtions.
  """
  labeling_examples = []
  df = pd.read_parquet(
      path,
      filters=[('example_id', 'in', allowed_example_ids)],
      engine='pyarrow',
  )
  for _, row in df.iterrows():
    pre_image_path, post_image_path, combined_image_path = (
        _write_labeling_images_from_dataframe_row(row, output_dir)
    )
    longitude, latitude = utils.decode_coordinates(row['encoded_coordinates'])
    labeling_examples.append(
        LabelingExample(
            int64_id=row['int64_id'],
            example_id=str(row['example_id']),
            pre_image_path=pre_image_path,
            post_image_path=post_image_path,
            combined_image_path=combined_image_path,
            tfrecord_path=path,
            serialized_example=_dataframe_row_to_example(row),
            longitude=longitude,
            latitude=latitude,
        )
    )
  return labeling_examples


def _create_labeling_assets_from_parquet_files(
    parquet_paths: list[str],
    output_dir: str,
    allowed_example_ids: set[str],
) -> list[LabelingExample]:
  """Creates assets needed for a labeling task from examples stored in Parquet.

  Writes combined and separate pre/post PNGs used by labeling tools.

  Args:
    parquet_paths: List of Parquet files.
    output_dir: Directory to write assets to.
    allowed_example_ids: Set of example_id from which a subset will be used in
      creating labeling task.

  Returns:
    List of LabelingExamples containing information about the created assets.
  """
  labeling_examples = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {
        executor.submit(_read_parquet, p, output_dir, allowed_example_ids): p
        for p in parquet_paths
    }
    with tqdm.tqdm(
        desc='examples', total=len(allowed_example_ids)
    ) as progress_bar:
      for future in concurrent.futures.as_completed(futures):
        examples = future.result()
        progress_bar.update(len(examples))
        labeling_examples.extend(examples)
  return labeling_examples


def _read_file(path: str) -> bytes:
  """Reads file contents."""
  with tf.io.gfile.GFile(path, 'rb') as f:
    return f.read()


def _read_files_concurrently(
    images_dir: str, example_ids: set[str]
) -> dict[tuple[str, str], bytes]:
  """Reads file contents."""
  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {}
    for example_id in example_ids:
      for subdir in ['pre', 'post', 'large_pre', 'large_post']:
        path = os.path.join(images_dir, subdir, f'{example_id}.png')
        futures[executor.submit(_read_file, path)] = (example_id, subdir)

    results = {}
    with tqdm.tqdm(
        desc='Read Images (4 per example)', total=len(example_ids) * 4
    ) as progress_bar:
      for future in concurrent.futures.as_completed(futures):
        example_id, subdir = futures[future]
        image_bytes = future.result()
        results[(example_id, subdir)] = image_bytes
        progress_bar.update(1)
  return results


def _write_file(path_and_content: tuple[str, bytes]) -> None:
  """Writes file contents."""
  with tf.io.gfile.GFile(path_and_content[0], 'wb') as f:
    f.write(path_and_content[1])


def _create_labeling_assets_from_metadata(
    metadata_pattern: str,
    images_dir: str,
    output_dir: str,
    allowed_example_ids: set[str],
) -> list[LabelingExample]:
  """Creates assets needed for a labeling task from metadata files and images.

  Args:
    metadata_pattern: Pattern matching paths to metadata file.
    images_dir: Directory containing pre/post images.
    output_dir: Directory to write assets to.
    allowed_example_ids: Set of example_id from which a subset will be used in
      creating labeling task.

  Returns:
    List of LabelingExamples containing information about the created assets.
  """
  metadata_df = _read_sharded_metadata(metadata_pattern)
  images = _read_files_concurrently(images_dir, allowed_example_ids)
  labeling_examples = []
  images_to_write = []
  for example_id in allowed_example_ids:
    matched_rows = metadata_df[metadata_df['example_id'] == example_id]
    if matched_rows.empty:
      raise ValueError(f'Example id {example_id} not found in metadata file.')
    if len(matched_rows) > 1:
      raise ValueError(
          f'Example id {example_id} found multiple times in metadata file.'
      )
    row = matched_rows.iloc[0].copy()
    row['pre_image_png'] = images[(example_id, 'pre')]
    row['post_image_png'] = images[(example_id, 'post')]
    row['pre_image_png_large'] = images[(example_id, 'large_pre')]
    row['post_image_png_large'] = images[(example_id, 'large_post')]

    pre_image_path = os.path.join(output_dir, f'{example_id}_pre.png')
    post_image_path = os.path.join(output_dir, f'{example_id}_post.png')
    combined_image_path = os.path.join(output_dir, f'{example_id}.png')
    before_image = utils.deserialize_image(
        row['pre_image_png_large'],
        'png',
    )
    after_image = utils.deserialize_image(
        row['post_image_png_large'],
        'png',
    )
    combined_image = utils.serialize_image(create_labeling_image(
        before_image, after_image, example_id, row['plus_code']
    ), 'png')
    images_to_write.extend([
        (pre_image_path, row['pre_image_png_large']),
        (post_image_path, row['post_image_png_large']),
        (combined_image_path, combined_image),
    ])
    longitude, latitude = utils.decode_coordinates(row['encoded_coordinates'])
    labeling_examples.append(
        LabelingExample(
            int64_id=row['int64_id'],
            example_id=str(example_id),
            pre_image_path=pre_image_path,
            post_image_path=post_image_path,
            combined_image_path=combined_image_path,
            tfrecord_path='',
            serialized_example=_dataframe_row_to_example(row),
            longitude=longitude,
            latitude=latitude,
        )
    )

  with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(_write_file, images_to_write)

  return labeling_examples


def _write_tfrecord(examples: Iterable[Example], path: str) -> None:
  """Writes a list of examples to a TFRecord file."""
  output_dir = os.path.dirname(path)
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  with tf.io.TFRecordWriter(path) as writer:
    for example in examples:
      writer.write(example.SerializeToString())


def filter_examples_from_allowed_ids(
    example_file: str, output_path: str, allowed_example_ids: set[str]
):
  """Filters examples based on allowed ids.

  Args:
    example_file: Path to file containing TF records.
    output_path: Path to output filtered examples.
    allowed_example_ids: List of example_id for filtering
  Returns:
    Empty list
  """
  filtered_examples = []
  for record in tf.data.TFRecordDataset([example_file]):
    example = Example()
    example.ParseFromString(record.numpy())
    if 'example_id' in example.features.feature:
      example_id = (
          example.features.feature['example_id'].bytes_list.value[0].decode()
      )
    else:
      # If the example doesn't have an "example_id" feature, fall back on
      # using "encoded_coordinates". This maintains backwards compatibility
      # with older datasets.
      # TODO(jzxu): Remove this branch when backward compatibility is no
      # longer needed.
      example_id = (
          example.features.feature['encoded_coordinates']
          .bytes_list.value[0]
          .decode()
      )
    if example_id in allowed_example_ids:
      filtered_examples.append(example)

  _write_tfrecord(
      filtered_examples, f'{output_path}/{example_file.split("/")[-1]}'
  )
  return []


def get_connection_matrix(
    longitudes: list[float],
    latitudes: list[float],
    encoded_coordinates: list[str],
    connecting_distance_meters: float,
)-> tuple[gpd.GeoDataFrame, np.ndarray]:
  """Gets a connection matrix for a set of points.

  Args:
    longitudes: Longitudes of points.
    latitudes: Latitudes of points.
    encoded_coordinates: Encoded coordinates of points.
    connecting_distance_meters: Maximum distance for two points to be connected.

  Returns:
    Tuple of (GeoDataFrame, connection_matrix).
  """

  gpd_df = utils.convert_to_utm(
      gpd.GeoDataFrame(
          {'encoded_coordinates': encoded_coordinates},
          geometry=gpd.points_from_xy(longitudes, latitudes),
          crs=4326,
      )
  )

  def calculate_euclidean_distance(row):
    return gpd_df.distance(row.geometry)

  distances = np.array(gpd_df.apply(calculate_euclidean_distance, axis=1))
  connection_matrix = (distances < connecting_distance_meters).astype('int')

  assert connection_matrix.shape == (
      len(encoded_coordinates),
      len(encoded_coordinates),
  )

  return gpd_df, connection_matrix


def get_connected_labels(
    connection_matrix: np.ndarray,
) -> list[str]:
  """Gets the labels of connected components.

  Args:
    connection_matrix: Connection matrix.

  Returns:
    List of labels of connected components. Components with the same label are
    connected and are therefore connected.
  """
  graph = scipy.sparse.csr_matrix(connection_matrix)
  _, labels = scipy.sparse.csgraph.connected_components(
      csgraph=graph, directed=False, return_labels=True
  )

  return list(labels)


def _split_examples(
    examples: list[Example],
    test_fraction: float,
    connecting_distance_meters: float,
) -> tuple[list[Example], list[Example]]:
  """Splits a list of examples into training and test sets.

  Examples with the same encoded coordinates will always end up in the same
  split to prevent leaking information between training and test sets. Any two
  examples separated by less than connecting_distance_meters will always be in
  the same split.

  Args:
    examples: Input examples.
    test_fraction: Fraction of examples to use for testing.
    connecting_distance_meters: Maximum distance for two points to be connected.

  Returns:
    Tuple of (training examples, test examples).
  """
  longitudes = []
  latitudes = []
  encoded_coordinates = []
  for example in examples:
    encoded_coordinates.append(utils.get_bytes_feature(
        example, 'encoded_coordinates'
    )[0].decode())
    longitude, latitude = utils.get_float_feature(example, 'coordinates')
    longitudes.append(longitude)
    latitudes.append(latitude)

  gpd_df, connection_matrix = get_connection_matrix(
      longitudes, latitudes, encoded_coordinates, connecting_distance_meters
  )
  labels = get_connected_labels(connection_matrix)
  connected_groups = collections.defaultdict(list)
  for idx, key in enumerate(labels):
    connected_groups[key].append(idx)

  list_of_connected_examples = []
  for _, connected_group in connected_groups.items():
    list_of_connected_examples.append(connected_group)

  num_test = int(len(gpd_df) * test_fraction)
  test_indices = get_testset_indices(num_test, list_of_connected_examples)
  test_examples = [examples[idx] for idx in test_indices]
  train_examples = [
      examples[idx] for idx in range(len(examples)) if idx not in test_indices
  ]

  return train_examples, test_examples


def get_testset_indices(num_test, list_of_connected_examples):
  """Get random list of indices corresponding to test examples.

  Args:
    num_test: Number of test examples.
    list_of_connected_examples: List of connected examples.

  Returns:
    List of indices corresponding test examples.
  """
  max_num_attempts_train_test_splits = 10000
  best_test_indices = []
  min_diff_best_num_test = num_test

  for _ in range(max_num_attempts_train_test_splits):
    # Ensure randomness
    random_list_of_connected_examples = random.sample(
        list_of_connected_examples, len(list_of_connected_examples)
    )
    current_test_indices = []

    for connected_component in random_list_of_connected_examples:
      current_test_indices.extend(connected_component)
      if abs(len(current_test_indices) - num_test) < min_diff_best_num_test:
        best_test_indices = current_test_indices.copy()
        min_diff_best_num_test = abs(len(best_test_indices) - num_test)

        # Stop trials once best best_test_indices is found
        if min_diff_best_num_test == 0:
          return best_test_indices

  return best_test_indices


def _merge_single_example_file_and_labels(
    example_file: str, labels: dict[str, list[tuple[str, float, str]]]
) -> list[Example]:
  """Merges TF records from single example_file with corresponding labels.

  Args:
    example_file: Path to file containing TF records.
    labels: Dictionary of example id to a list of tuples
        (string label, numeric label, source dataset id).

  Returns:
    List of TF examples merged with labels for a single example_file.
  """
  labeled_examples = []
  for example in _tfrecord_iterator(example_file):
    if 'example_id' in example.features.feature:
      example_id = (
          example.features.feature['example_id'].bytes_list.value[0].decode()
      )
    else:
      # If the example doesn't have an "example_id" feature, fall back on
      # using "encoded_coordinates". This maintains backwards compatibility
      # with older datasets.
      # TODO(jzxu): Remove this branch when backward compatibility is no
      # longer needed.
      example_id = (
          example.features.feature['encoded_coordinates']
          .bytes_list.value[0]
          .decode()
      )

    label_tuples = labels.get(example_id, [])
    for string_label, numeric_label, dataset_id_or_label_path in label_tuples:
      labeled_example = Example()
      labeled_example.CopyFrom(example)
      features = labeled_example.features
      features.feature['string_label'].bytes_list.value[:] = [
          string_label.encode()
      ]

      if tf.io.gfile.exists(dataset_id_or_label_path):
        features.feature['label_file_path'].bytes_list.value.append(
            dataset_id_or_label_path.encode()
        )
      else:
        features.feature['label_dataset_id'].bytes_list.value.append(
            dataset_id_or_label_path.encode()
        )

      label_feature = features.feature['label'].float_list
      if not label_feature.value:
        label_feature.value.append(numeric_label)
      else:
        label_feature.value[0] = numeric_label
      labeled_examples.append(labeled_example)

  return labeled_examples


def _match_patterns(patterns: list[str]) -> list[str]:
  paths = []
  for pattern in patterns:
    if not (matches := tf.io.gfile.glob(pattern)):
      raise ValueError(f'File pattern {pattern} did not match anything')
    paths.extend(matches)
  return paths


def _merge_examples_and_labels(
    example_patterns: list[str],
    labels: dict[str, list[tuple[str, float, str]]],
    use_multiprocessing: bool,
    multiprocessing_context: Any,
    max_processes: int,
) -> list[Example]:
  """Merges examples with labels into train and test TFRecords.

  Args:
    example_patterns: File patterns for input examples.
    labels: Dictionary of example ids to a list of tuples
        (string label, numeric label, source dataset id).
    use_multiprocessing: If true, create multiple processes to create labeled
      examples.
    multiprocessing_context: Context to spawn processes with when using
      multiprocessing.
    max_processes: Maximum number of processes.

  Returns:
    A list of labeled examples.
  """

  example_files = _match_patterns(example_patterns)
  if not labels:
    raise ValueError(
        'Dictionary of labels is empty. Ensure that the dictionary of'
        ' labels is not empty'
    )

  if use_multiprocessing:
    num_workers = min(
        multiprocessing.cpu_count(), len(example_files), max_processes
    )
    if multiprocessing_context:
      pool = multiprocessing_context.Pool(num_workers)
    else:
      pool = multiprocessing.Pool(num_workers)

    logging.info('Using multiprocessing with %d processes.', num_workers)
    results = pool.map(
        functools.partial(
            _merge_single_example_file_and_labels, labels=labels
        ),
        example_files,
    )
  else:
    logging.info('Not using multiprocessing.')
    results = [
        _merge_single_example_file_and_labels(example_file, labels)
        for example_file in example_files
    ]

  all_labeled_examples = []
  for result in results:
    all_labeled_examples.extend(result)
  return all_labeled_examples


def _read_label_file(path: str) -> list[tuple[str, str, str]]:
  """Reads a label file.

  The file should be a CSV containing at least an "example_id" column
  and a "string_label" column. In the future example_ids will also be supported.

  Args:
    path: Path to file.

  Returns:
    List of (example id, string label, label file path) tuples.
  """
  with tf.io.gfile.GFile(path) as f:
    df = pd.read_csv(f)

  if 'example_id' not in df.columns:
    raise ValueError('Label file must contain "example_id" column.')
  if 'string_label' not in df.columns:
    raise ValueError('Label file must contain "string_label" column.')

  return [(row.example_id, row.string_label, path) for _, row in df.iterrows()]


def create_labeled_examples(
    label_file_paths: list[str],
    string_to_numeric_labels: list[str],
    example_patterns: list[str],
    splits_path: str | None,
    test_fraction: float,
    train_output_path: str,
    test_output_path: str,
    connecting_distance_meters: float,
    use_multiprocessing: bool,
    multiprocessing_context: Any,
    max_processes: int) -> None:
  """Creates a labeled dataset by merging cloud labels and unlabeled examples.

  Args:
    label_file_paths: Paths to files to read labels from.
    string_to_numeric_labels: List of strings in the form
      "<string label>=<numeric label>", e.g. "no_damage=0"
    example_patterns: Patterns for unlabeled examples.
    splits_path: Path to CSV mapping example ids to train/test splits.
    test_fraction: Fraction of examples to write to test output.
    train_output_path: Path to training examples TFRecord output.
    test_output_path: Path to test examples TFRecord output.
    connecting_distance_meters: Maximum distance for two points to be connected.
    use_multiprocessing: If true, create multiple processes to create labeled
      examples.
    multiprocessing_context: Context to spawn processes with when using
      multiprocessing.
    max_processes: Maximum number of processes.
  """
  string_to_numeric_map = {}
  for string_to_numeric_label in string_to_numeric_labels:
    if '=' not in string_to_numeric_label:
      raise ValueError(
          f'Invalid label mapping "{string_to_numeric_label}", should have '
          'form "label=0 or 1".')
    label, numeric_label = string_to_numeric_label.split('=')
    try:
      string_to_numeric_map[label] = float(numeric_label)
    except TypeError:
      logging.error('Class %s is not numeric.', numeric_label)
      raise

  if label_file_paths:
    labels = []
    for path in label_file_paths:
      labels.extend(_read_label_file(path))

    logging.info('Read %d labels total.', len(labels))
    ids_to_labels = collections.defaultdict(list)
    for example_id, string_label, dataset_id_or_label_path in labels:
      example_labels = ids_to_labels[example_id]
      if string_label in [l[0] for l in example_labels]:
        # Don't add multiple labels with the same value for a single example.
        continue
      numeric_label = string_to_numeric_map.get(string_label, None)
      if numeric_label is None:
        raise ValueError(f'Label "{string_label}" has no numeric mapping.')
      example_labels.append(
          (string_label, numeric_label, dataset_id_or_label_path)
      )

    all_labeled_examples = _merge_examples_and_labels(
        example_patterns,
        ids_to_labels,
        use_multiprocessing,
        multiprocessing_context,
        max_processes
    )
    if not all_labeled_examples:
      raise ValueError('No examples found matching labels.')
  else:
    # If no labels are provided, then assume that input examples already have
    # labels. Simply read them in.
    all_labeled_examples = []
    for example_path in _match_patterns(example_patterns):
      for example in _tfrecord_iterator(example_path):
        if ('string_label' not in example.features.feature) or (
            'label' not in example.features.feature
        ):
          raise ValueError(
              f'An example in file {example_path} does not have a string_label'
              ' or label feature.'
          )
        all_labeled_examples.append(example)

  if splits_path:
    with tf.io.gfile.GFile(splits_path) as f:
      df = pd.read_csv(f)
      if 'example_id' not in df.columns:
        raise ValueError('Splits CSV must contain "example_id" column.')
      if 'split' not in df.columns:
        raise ValueError('Splits CSV must contain "split" column.')
    example_id_to_split = {r.example_id: r.split for r in df.itertuples()}
    train_examples = []
    test_examples = []
    for e in all_labeled_examples:
      example_id = (
          e.features.feature['example_id'].bytes_list.value[0].decode()
      )
      split = example_id_to_split.get(example_id, None)
      if split is None:
        raise ValueError(f'Example id {example_id} not found in splits CSV.')
      if split == 'train':
        train_examples.append(e)
      elif split == 'test':
        test_examples.append(e)
      else:
        raise ValueError(f'Unknown split: {split}')
  else:
    train_examples, test_examples = _split_examples(
        all_labeled_examples, test_fraction, connecting_distance_meters
    )

  _write_tfrecord(train_examples, train_output_path)
  _write_tfrecord(test_examples, test_output_path)
  logging.info(
      'Written %d test examples and %d train examples',
      len(test_examples),
      len(train_examples),
  )
