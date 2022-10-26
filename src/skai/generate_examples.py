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
"""Pipeline for generating tensorflow examples from satellite images."""

import dataclasses
import logging
import os
import pathlib
import random
from typing import Dict, Iterator, List, Optional, Tuple
import apache_beam as beam
import cv2
import geopandas as gpd
import numpy as np
import PIL
import PIL.Image

from skai import beam_utils
from skai import cloud_labeling
from skai import read_raster
from skai import utils
import tensorflow as tf


Example = tf.train.Example
Metrics = beam.metrics.Metrics
PipelineOptions = beam.options.pipeline_options.PipelineOptions

# If more than this fraction of a before or after image is blank, discard this
# example.
_BLANK_THRESHOLD = 0.25

# Technique used for aligning before and after images. See the OpenCV
# documentation on template matching for the list of options.
_ALIGNMENT_METHOD = cv2.TM_CCOEFF_NORMED

# Maximum number of pixels that an image can be displaced during alignment.
_MAX_DISPLACEMENT = 30

# Multi-output tags for GenerateExamplesFn.
_EXAMPLES = 'examples'
_LABELING_IMAGES = 'label_images'


@dataclasses.dataclass
class _Coordinate:
  """Class that encodes a geographic position and a label.

  Attributes:
    longitude: Longitude.
    latitude: Latitude.
    label: Label for for this coordinate.
  """
  longitude: float
  latitude: float
  label: float

  def __post_init__(self):
    # Check if the longitude and latitude are valid
    if not -180 <= self.longitude <= 180:
      raise ValueError(
          f'Invalid longitude, got {self.longitude}'
      )
    if not -90 <= self.latitude <= 90:
      raise ValueError(
          f'Invalid latitude, got {self.latitude}'
      )


def _to_grayscale(image: np.ndarray) -> np.ndarray:
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def align_after_image(before_image: np.ndarray, after_image: np.ndarray):
  """Aligns after image to before image.

  Uses OpenCV template matching algorithm to align before and after
  images. Assumes that after_image is larger than before_image, so that the best
  alignment can be found. If the two images are the same size, then obviously no
  alignment is possible.

  Args:
    before_image: Before image.
    after_image: After image.

  Returns:
    A crop of after_image that is the same size as before_image and is best
    aligned to it.
  """
  result = cv2.matchTemplate(
      _to_grayscale(after_image), _to_grayscale(before_image),
      _ALIGNMENT_METHOD)
  _, _, _, max_location = cv2.minMaxLoc(result)
  j, i = max_location
  rows = before_image.shape[0]
  cols = before_image.shape[1]
  aligned_after = after_image[i:i + rows, j:j + cols, :]
  return aligned_after


def _mostly_blank(image: np.ndarray) -> bool:
  """Determines if an image is mostly blank.

  Assumes that the first dimension of the input data is the channel dimension. A
  pixel is considered blank if it has 0s in all channels.

  Args:
    image: Input image.

  Returns:
    Whether the image has too many blank pixels.
  """
  if image.size == 0:
    return 0

  flattened = image.max(axis=0)
  num_non_blank = np.count_nonzero(flattened)
  blank_fraction = (flattened.size - num_non_blank) / flattened.size
  return blank_fraction >= _BLANK_THRESHOLD


def _create_example(example_id: str,
                    before_image: np.ndarray, after_image: np.ndarray,
                    scalar_features: Dict[str, List[float]]) -> Example:
  """Create Tensorflow Example from inputs.

  Args:
    example_id: Example id.
    before_image: Before disaster image.
    after_image: After disaster image.
    scalar_features: Dict mapping scalar feature names to values.

  Returns:
    Tensorflow Example.
  """
  example = tf.train.Example()
  # TODO(jzxu): Use constants for these feature name strings.

  # For legacy reasons, the example id feature is named "encoded_coordinates".
  # Should change this in the future.
  utils.add_bytes_feature('encoded_coordinates', example_id.encode(), example)
  utils.add_bytes_feature('pre_image_png',
                          tf.io.encode_png(before_image).numpy(), example)
  utils.add_bytes_feature('post_image_png',
                          tf.io.encode_png(after_image).numpy(), example)
  for name, value in scalar_features.items():
    utils.add_float_list_feature(name, value, example)
  return example


def _center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
  """Crops an image into a square of a specified size.

  Args:
    image: Input image array.
    crop_size: Length and width of the cropped image.

  Returns:
    The cropped image.
  """
  rows = image.shape[0]
  cols = image.shape[1]
  i = rows // 2 - crop_size // 2
  j = cols // 2 - crop_size // 2
  return image[i:i + crop_size, j:j + crop_size, :]


class GenerateExamplesFn(beam.DoFn):
  """DoFn that extracts patches from before and after images into examples.

  The DoFn takes as input a list of (longitude, latitude) coordinates and
  extracts patches centered at each coordinate from the before and after images,
  and creates Tensorflow Examples containing these patches.

  The after image is also aligned to the before image during this process. The
  maximum displacement that can occur in alignment is _MAX_DISPLACEMENT pixels.

  Attributes:
    _large_patch_size: Size in pixels of the before and after image patches.
      Typically 256.
    _example_patch_size: Size in pixels of the smaller before and after image
      patches used in TF Examples. This is typically 64.
    _use_before_image: Whether to include before images in the examples.
    _labeling_image_sample_rate: Rate at which to sample labeling images.
  """

  def __init__(self,
               large_patch_size: int,
               example_patch_size: int,
               use_before_image: bool,
               labeling_image_sample_rate: float) -> None:
    self._large_patch_size = large_patch_size
    self._example_patch_size = example_patch_size
    self._use_before_image = use_before_image
    self._labeling_image_sample_rate = labeling_image_sample_rate

    self._example_count = Metrics.counter('skai', 'generated_examples_count')
    self._bad_example_count = Metrics.counter('skai', 'rejected_examples_count')
    self._before_patch_blank_count = Metrics.counter(
        'skai', 'before_patch_blank_count')
    self._after_patch_blank_count = Metrics.counter(
        'skai', 'after_patch_blank_count')

  def process(self, grouped_features) -> Iterator[beam.pvalue.TaggedOutput]:
    """Extract patches from before and after images and output as tf Example.

    Args:
      grouped_features: All features for an example.

    Yields:
      Serialized Tensorflow Example.
    """
    example_id, features = grouped_features
    features = dict(features)
    scalar_features = features.get('scalar_features', {})

    after_image = features.get('after', None)
    if after_image is None:
      self._after_patch_blank_count.inc()
      self._bad_example_count.inc()
      return

    if self._use_before_image:
      before_image = features.get('before', None)
      if before_image is None:
        self._before_patch_blank_count.inc()
        self._bad_example_count.inc()
        return
      # Align after image to before image.
      after_image = align_after_image(before_image, after_image)
    else:
      before_image = np.zeros(
          (self._large_patch_size, self._large_patch_size, 3), dtype=np.uint8)

    before_crop = _center_crop(before_image, self._example_patch_size)
    if self._use_before_image and _mostly_blank(before_crop):
      self._before_patch_blank_count.inc()
      self._bad_example_count.inc()
      return
    after_crop = _center_crop(after_image, self._example_patch_size)
    if _mostly_blank(after_crop):
      self._after_patch_blank_count.inc()
      self._bad_example_count.inc()
      return
    example = _create_example(example_id, before_crop, after_crop,
                              scalar_features)
    self._example_count.inc()
    yield beam.pvalue.TaggedOutput(_EXAMPLES, example.SerializeToString())

    if random.random() < self._labeling_image_sample_rate:
      labeling_image = cloud_labeling.create_labeling_image(
          PIL.Image.fromarray(before_image), PIL.Image.fromarray(after_image))
      serialized_labeling_image = utils.serialize_image(labeling_image, 'png')
      labeling_image_name = f'{example_id}.png'
      yield beam.pvalue.TaggedOutput(
          _LABELING_IMAGES,
          (labeling_image_name, serialized_labeling_image))


def _get_setup_file_path():
  return str(pathlib.Path(__file__).parent.parent / 'setup.py')


def _get_dataflow_pipeline_options(
    job_name: str, project: str, region: str, temp_dir: str,
    dataflow_container_image: str,
    worker_service_account: Optional[str], max_workers: int) -> PipelineOptions:
  """Returns dataflow pipeline options.

  Args:
    job_name: Name of Dataflow job.
    project: GCP project.
    region: GCP region.
    temp_dir: Temporary data location.
    dataflow_container_image: Docker container to use.
    worker_service_account: Email of the service account will launch workers.
        If None, uses the project's default Compute Engine service account
        (<project-number>-compute@developer.gserviceaccount.com).
    max_workers: Maximum number of Dataflow workers.

  Returns:
    Dataflow options.
  """
  options = {
      'job_name': job_name,
      'project': project,
      'region': region,
      'temp_location': temp_dir,
      'runner': 'DataflowRunner',
      'experiment': 'use_runner_v2',
      'sdk_container_image': dataflow_container_image,
      'setup_file': _get_setup_file_path(),
      'max_num_workers': max_workers
  }
  if worker_service_account:
    options['service_account_email'] = worker_service_account
  return PipelineOptions.from_dictionary(options)


def _get_local_pipeline_options() -> PipelineOptions:
  return PipelineOptions.from_dictionary({
      'runner': 'DirectRunner',
      'direct_num_workers': 10,
      'direct_running_mode': 'multi_processing',
  })


def _coordinates_to_scalar_features(longitude: float, latitude: float,
                                    label: float):
  example_id = utils.encode_coordinates(longitude, latitude)
  return (example_id, ('scalar_features', {
      'coordinates': [longitude, latitude],
      'label': [label]
  }))


def _generate_examples(
    pipeline,
    before_image_path: str,
    after_image_path: str,
    large_patch_size: int,
    example_patch_size: int,
    resolution: float,
    unlabeled_coordinates: List[Tuple[float, float]],
    labeled_coordinates: List[Tuple[float, float, float]],
    labeling_image_sample_rate: float,
    gdal_env: Dict[str, str],
    stage_prefix: str) -> Tuple[beam.PCollection, beam.PCollection]:
  """Generates examples and labeling images from source images.

  Args:
    pipeline: Beam pipeline.
    before_image_path: Before image path.
    after_image_path: After image path.
    large_patch_size: Size in pixels of before and after image patches for
      labeling and alignment. Typically 256.
    example_patch_size: Size of patches to extract into examples. Typically 64.
    resolution: Desired resolution of image patches.
    unlabeled_coordinates: List of coordinates (longitude, latitude) to extract
      unlabeled examples for.
    labeled_coordinates: List of coordinates (longitude, latitude, label) to
      extract labeled examples for.
    labeling_image_sample_rate: Rate at which to sample labeling images.
    gdal_env: GDAL environment configuration.
    stage_prefix: Beam stage name prefix.

  Returns:
    PCollection of examples and PCollection of labeling images.
  """

  if unlabeled_coordinates:
    coords_with_ids = [
        (utils.encode_coordinates(x, y), x, y) for x, y in unlabeled_coordinates
    ]
    scalar_features = [
        _coordinates_to_scalar_features(x, y, -1.0)
        for x, y in unlabeled_coordinates
    ]
  elif labeled_coordinates:
    coords_with_ids = [(utils.encode_coordinates(x, y), x, y)
                       for x, y, _ in labeled_coordinates]
    scalar_features = [
        _coordinates_to_scalar_features(x, y, label)
        for x, y, label in labeled_coordinates
    ]

  scalar_features_collection = (
      pipeline
      | stage_prefix + '_make_scalar_features' >> beam.Create(scalar_features))

  input_collections = [scalar_features_collection]
  after_image_size = large_patch_size
  use_before_image = bool(before_image_path)
  if use_before_image:
    before_images = read_raster.extract_patches_from_raster(
        pipeline, before_image_path, 'before', large_patch_size, resolution,
        coords_with_ids, gdal_env, 'before')
    input_collections.append(before_images)
    # Make the after image patch larger than the before image patch by
    # giving it a border of _MAX_DISPLACEMENT pixels. This gives the
    # alignment algorithm at most +/-_MAX_DISPLACEMENT pixels of movement in
    # either dimension to find the best alignment.
    after_image_size += 2 * _MAX_DISPLACEMENT

  after_images = read_raster.extract_patches_from_raster(
      pipeline, after_image_path, 'after', after_image_size, resolution,
      coords_with_ids, gdal_env, 'after')
  input_collections.append(after_images)

  results = (input_collections
             | stage_prefix + '_merge_features' >> beam.Flatten()
             | stage_prefix + '_group_by_example_id' >> beam.GroupByKey()
             | stage_prefix + '_generate_examples' >> beam.ParDo(
                 GenerateExamplesFn(large_patch_size, example_patch_size,
                                    use_before_image,
                                    labeling_image_sample_rate)).with_outputs(
                                        _EXAMPLES, _LABELING_IMAGES))
  examples = results[_EXAMPLES]
  labeling_images = results[_LABELING_IMAGES]
  return examples, labeling_images


def _parse_coords_from_csv_line(line: str) -> _Coordinate:
  x, y = [float(w.strip()) for w in line.split(',')]
  return _Coordinate(x, y, -1.0)


def read_labels_file(
    path: str, label_property: str, labels_to_classes: List[str],
    max_points: int) -> List[Tuple[float, float, float]]:
  """Reads labels from a GIS file.

  If the "label_property" is a string, then it is assumed to be the name of a
  class, e.g. "damaged". In labels_to_classes, user can specify the mapping of
  the class and label, e.g. "damaged=1". If the name is not in
  "labels_to_classes", the example is dropped.

  If the label is a float or integer, it is read as-is.

  Args:
    path: Path to the file to be read.
    label_property: The property to use as the label, e.g. "Main_Damag".
    labels_to_classes: List of string in "class=label" format, e.g.
      ["undamaged=0", "damaged=1", "destroyed=1"].
    max_points: Number of labeled examples to keep

  Returns:
    List of tuples of the form (longitude, latitude, float label).
  """
  # Parse labels_to_classes into dictionary format
  label_to_class_dict = {}
  for label_to_class in labels_to_classes:
    if '=' not in label_to_class:
      raise ValueError(
          f'Invalid label to class mapping "{label_to_class}",'
          f'should have form "label=class".')
    label, _, numeric_class = label_to_class.partition('=')
    try:
      label_to_class_dict[label] = float(numeric_class)
    except TypeError:
      logging.error('Class %s is not numeric.', numeric_class)
      raise

  # Generate coordinates from label file
  df = gpd.read_file(path).to_crs(epsg=4326)
  coordinates = []
  for _, row in df.iterrows():
    centroid = row.geometry.centroid
    label = row[label_property]
    if isinstance(label, str):
      try:
        float_label = label_to_class_dict[label]
      except KeyError:
        logging.warning('Label %s is not recognized.', label)
    elif isinstance(label, (int, float)):
      float_label = float(label)
    else:
      raise ValueError(f'Unrecognized label property type {type(label)}')

    coordinates.append((centroid.x, centroid.y, float_label))

  if max_points:
    coordinates = coordinates[:max_points]

  # logging.info('Read %d labeled coordinates.', len(coordinates))
  return coordinates


def get_dataflow_container_image(py_version: str) -> str:
  """Gets default dataflow image based on Python version.

  Args:
    py_version: Python version
  Returns:
    Dataflow container image path.
  """
  if py_version == '3.7':
    return 'gcr.io/disaster-assessment/dataflow_3.7_image:latest'
  elif py_version == '3.8':
    return 'gcr.io/disaster-assessment/dataflow_3.8_image:latest'
  elif py_version == '3.9':
    return 'gcr.io/disaster-assessment/dataflow_3.9_image:latest'
  else:
    return None


def parse_gdal_env(settings: List[str]) -> Dict[str, str]:
  """Parses a list of GDAL environment variable settings into a dictionary.

  Args:
    settings: A list of environment variable settings in "var=value" format.

  Returns:
    Dictionary with variable as key and assigned value.
  """
  gdal_env = {}
  for setting in settings:
    if '=' not in setting:
      raise ValueError(
          'Each GDAL environment setting should have the form "var=value".')
    var, _, value = setting.partition('=')
    gdal_env[var] = value
  return gdal_env


def generate_examples_pipeline(
    before_image_path: str,
    after_image_path: str,
    large_patch_size: int,
    example_patch_size: int,
    resolution: float,
    output_dir: str,
    num_output_shards: int,
    unlabeled_coordinates: List[Tuple[float, float]],
    labeled_coordinates: List[Tuple[float, float, float]],
    use_dataflow: bool,
    num_labeling_images: int,
    gdal_env: Dict[str, str],
    dataflow_job_name: Optional[str],
    dataflow_container_image: Optional[str],
    cloud_project: Optional[str],
    cloud_region: Optional[str],
    worker_service_account: Optional[str],
    max_workers: int) -> None:
  """Runs example generation pipeline.

  Args:
    before_image_path: Before image path.
    after_image_path: After image path.
    large_patch_size: Size in pixels of before and after image patches for
      labeling and alignment. Typically 256.
    example_patch_size: Size of patches to extract into examples. Typically 64.
    resolution: Desired resolution of image patches.
    output_dir: Parent output directory.
    num_output_shards: Number of output shards.
    unlabeled_coordinates: List of coordinates (longitude, latitude) to extract
      unlabeled examples for.
    labeled_coordinates: List of coordinates (longitude, latitude, label) to
      extract labeled examples for.
    use_dataflow: If true, run pipeline on GCP Dataflow.
    num_labeling_images: Number of labeling images to generate, or 0 to disable.
    gdal_env: GDAL environment configuration.
    dataflow_job_name: Name of dataflow job.
    dataflow_container_image: Container image to use when running Dataflow.
    cloud_project: Cloud project name.
    cloud_region: Cloud region, e.g. us-central1.
    worker_service_account: Email of service account that will launch workers.
    max_workers: Maximum number of workers to use.
  """

  temp_dir = os.path.join(output_dir, 'temp')
  if use_dataflow:
    if cloud_project is None or cloud_region is None:
      raise ValueError(
          'cloud_project and cloud_region must be specified when using '
          'Dataflow.')
    pipeline_options = _get_dataflow_pipeline_options(dataflow_job_name,
                                                      cloud_project,
                                                      cloud_region, temp_dir,
                                                      dataflow_container_image,
                                                      worker_service_account,
                                                      max_workers)
  else:
    pipeline_options = _get_local_pipeline_options()

  with beam.Pipeline(options=pipeline_options) as pipeline:
    if unlabeled_coordinates:
      examples_output_prefix = (
          os.path.join(output_dir, 'examples', 'unlabeled', 'unlabeled'))
      labeling_image_sample_rate = (
          num_labeling_images / len(unlabeled_coordinates))

    elif labeled_coordinates:
      examples_output_prefix = (
          os.path.join(output_dir, 'examples', 'labeled', 'labeled'))
      labeling_image_sample_rate = 0

    examples, labeling_images = _generate_examples(
        pipeline, before_image_path, after_image_path, large_patch_size,
        example_patch_size, resolution, unlabeled_coordinates,
        labeled_coordinates, labeling_image_sample_rate, gdal_env,
        'generate_examples')

    _ = (
        examples
        | 'write_examples' >> beam.io.tfrecordio.WriteToTFRecord(
            examples_output_prefix,
            file_name_suffix='.tfrecord',
            num_shards=num_output_shards))

    if num_labeling_images > 0:
      labeling_images_dir = (
          os.path.join(output_dir, 'examples', 'labeling_images'))
      beam_utils.write_records_as_files(labeling_images, labeling_images_dir,
                                        temp_dir, 'write_labeling_images')
