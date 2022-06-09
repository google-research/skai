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
import os
import pathlib
from typing import Dict, Iterator, List, Optional, Tuple
import apache_beam as beam
import cv2
import numpy as np
import PIL
import PIL.Image
import pyproj
import rasterio
import rasterio.plot
from skai import beam_utils
from skai import cloud_labeling
from skai import utils
import tensorflow as tf

Example = tf.train.Example
Image = PIL.Image.Image
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

# Maximum number of dataflow workers to use.
_MAX_DATAFLOW_WORKERS = 20


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


def _get_blank_fraction(data: np.ndarray) -> float:
  """Get the fraction of blank elements in an array.

  Assumes that the first dimension of the input data is the channel dimension. A
  pixel is considered blank if it has 0s in all channels.

  Args:
    data: Input array.

  Returns:
    Fraction of data that is all 0s.
  """
  if data.size == 0:
    return 0

  flattened = data.max(axis=0)
  num_non_blank = np.count_nonzero(flattened)
  return (flattened.size - num_non_blank) / flattened.size


def _get_patch_at_coordinate(
    raster,
    longitude: float,
    latitude: float,
    patch_size: int,
    resolution: float) -> Optional[np.ndarray]:
  """Extracts image patch from a raster.

  Args:
    raster: Input raster.
    longitude: Longitude of center of patch to extract.
    latitude: Latitude of center of patch to extract.
    patch_size: Patch size.
    resolution: Desired resolution of output patch.

  Returns:
    The image patch, or None if the coordinates are out of the bounds of the
    raster.
  """
  # Set always_xy=True so that transformer always expects longitude, latitude in
  # that order.
  transformer = pyproj.Transformer.from_crs(
      'epsg:4326', raster.crs, always_xy=True)
  x, y = transformer.transform(longitude, latitude, errcheck=True)
  row, col = raster.index(x, y)

  if raster.res[0] != raster.res[1]:
    raise ValueError(
        'Rasters with different x and y resolutions are not supported.')

  raster_res = raster.res[0]
  scale_factor = resolution / raster_res
  input_size = int(patch_size * scale_factor)

  half_size = input_size // 2
  col_off = col - half_size
  row_off = row - half_size
  window = rasterio.windows.Window(col_off, row_off, input_size, input_size)
  # Currently assumes that bands [1, 2, 3] of the input image are the RGB
  # channels.
  window_data = raster.read(
      indexes=[1, 2, 3], window=window, boundless=True, fill_value=-1,
      out_shape=(3, patch_size, patch_size))
  if _get_blank_fraction(window_data) > _BLANK_THRESHOLD:
    return None
  window_data = np.clip(window_data, 0, None)
  return rasterio.plot.reshape_as_image(window_data)


def _create_example(before_image: Image, after_image: Image,
                    longitude: float, latitude: float, label: float) -> Example:
  """Create Tensorflow Example from inputs.

  Args:
    before_image: Before disaster image.
    after_image: After disaster image.
    longitude: Longitude of center of image.
    latitude: Latitude of center of image.
    label: Label for this example.

  Returns:
    Tensorflow Example.
  """
  example = tf.train.Example()
  # TODO: Use constants for these feature name strings.
  utils.add_bytes_feature('pre_image_png',
                          utils.serialize_image(before_image, 'png'), example)
  utils.add_bytes_feature('post_image_png',
                          utils.serialize_image(after_image, 'png'), example)
  utils.add_float_feature('coordinates', longitude, example)
  utils.add_float_feature('coordinates', latitude, example)
  utils.add_bytes_feature('encoded_coordinates',
                          utils.encode_coordinates(longitude, latitude),
                          example)
  utils.add_float_feature('label', label, example)

  return example


def _center_crop(image: np.ndarray, crop_size: int) -> Image:
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
  crop = image[i:i + crop_size, j:j + crop_size, :]
  return PIL.Image.fromarray(crop)


class GenerateExamplesFn(beam.DoFn):
  """DoFn that extracts patches from before and after images into examples.

  The DoFn takes as input a list of (longitude, latitude) coordinates and
  extracts patches centered at each coordinate from the before and after images,
  and creates Tensorflow Examples containing these patches.

  The after image is also aligned to the before image during this process. The
  maximum displacement that can occur in alignment is _MAX_DISPLACEMENT pixels.

  Attributes:
    _before_path: Path to before disaster image.
    _after_path: Path to after disaster image.
    _example_patch_size: Size in pixels of the before and after image patches
      included in the examples.
    _alignment_patch_size: Size in pixels of the before and after image patches
      used during alignment. Must be larger than _patch_size. The reasoning for
      this is that more context is needed to perform a good alignment.
    _labeling_patch_size: Size in pixels of before and after image patches for
      labeling.
    _resolution: The desired resolution (in m/pixel) of the image patches. If
      this is different from the image's native resolution, patches will be
      upsampled or downsampled.
    _gdal_env: GDAL environment configuration.
  """

  def __init__(self,
               before_path: str,
               after_path: str,
               generate_labeling_images: bool,
               example_patch_size: int,
               alignment_patch_size: int,
               labeling_patch_size: int,
               resolution: float,
               gdal_env: Dict[str, str]) -> None:
    self._before_path = before_path
    self._after_path = after_path
    self._generate_labeling_images = generate_labeling_images
    self._example_patch_size = example_patch_size
    self._alignment_patch_size = alignment_patch_size
    self._labeling_patch_size = labeling_patch_size
    self._resolution = resolution
    self._gdal_env = gdal_env

    self._example_count = Metrics.counter('skai', 'generated_examples_count')
    self._bad_example_count = Metrics.counter('skai', 'rejected_examples_count')
    self._before_patch_blank_count = Metrics.counter(
        'skai', 'before_patch_blank_count')
    self._after_patch_blank_count = Metrics.counter(
        'skai', 'after_patch_blank_count')

  def setup(self) -> None:
    """Open before and after image rasters.

    This simply creates raster placeholders in memory. It doesn't actually read
    the raster data from disk.
    """
    with rasterio.Env(**self._gdal_env):
      self._before_raster = rasterio.open(self._before_path)
      self._after_raster = rasterio.open(self._after_path)

  def process(
      self, coordinate: _Coordinate) -> Iterator[beam.pvalue.TaggedOutput]:
    """Extract patches from before and after images and output as tf Example.

    Args:
      coordinate: Longitude and latitude of the center the of patch.

    Yields:
      Serialized Tensorflow Example.
    """
    with rasterio.Env(**self._gdal_env):
      before_patch = _get_patch_at_coordinate(self._before_raster,
                                              coordinate.longitude,
                                              coordinate.latitude,
                                              self._alignment_patch_size,
                                              self._resolution)

      # Make the after image patch larger than the before image patch by giving
      # it a border of _MAX_DISPLACEMENT pixels. This gives the alignment
      # algorithm at most +/-_MAX_DISPLACEMENT pixels of movement in either
      # dimension to find the best alignment.
      after_patch_size = self._alignment_patch_size + 2 * _MAX_DISPLACEMENT
      after_patch = _get_patch_at_coordinate(self._after_raster,
                                             coordinate.longitude,
                                             coordinate.latitude,
                                             after_patch_size,
                                             self._resolution)

      if before_patch is None:
        self._before_patch_blank_count.inc()
        self._bad_example_count.inc()
      elif after_patch is None:
        self._after_patch_blank_count.inc()
        self._bad_example_count.inc()
      else:
        aligned_after_patch = align_after_image(before_patch, after_patch)
        example = _create_example(
            _center_crop(before_patch, self._example_patch_size),
            _center_crop(aligned_after_patch, self._example_patch_size),
            coordinate.longitude, coordinate.latitude, coordinate.label)

        self._example_count.inc()
        yield beam.pvalue.TaggedOutput(_EXAMPLES, example.SerializeToString())

        if self._generate_labeling_images:
          labeling_image = cloud_labeling.create_labeling_image(
              _center_crop(before_patch, self._labeling_patch_size),
              _center_crop(aligned_after_patch, self._labeling_patch_size))
          serialized_labeling_image = utils.serialize_image(
              labeling_image, 'png')
          encoded_coords = utils.encode_coordinates(
              coordinate.longitude, coordinate.latitude).decode()
          labeling_image_name = f'{encoded_coords}.png'
          yield beam.pvalue.TaggedOutput(
              _LABELING_IMAGES,
              (labeling_image_name, serialized_labeling_image))


def _get_setup_file_path():
  return str(pathlib.Path(__file__).parent.parent / 'setup.py')


def _get_dataflow_pipeline_options(
    project: str, region: str, temp_dir: str,
    dataflow_container_image: str,
    worker_service_account: Optional[str]) -> PipelineOptions:
  """Returns dataflow pipeline options.

  Args:
    project: GCP project.
    region: GCP region.
    temp_dir: Temporary data location.
    dataflow_container_image: Docker container to use.
    worker_service_account: Email of the service account will launch workers.
        If None, uses the project's default Compute Engine service account
        (<project-number>-compute@developer.gserviceaccount.com).

  Returns:
    Dataflow options.
  """
  options = {
      'project': project,
      'region': region,
      'temp_location': temp_dir,
      'runner': 'DataflowRunner',
      'experiment': 'use_runner_v2',
      'sdk_container_image': dataflow_container_image,
      'setup_file': _get_setup_file_path(),
      'max_num_workers': _MAX_DATAFLOW_WORKERS
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


def _generate_examples(
    before_image_path: str,
    after_image_path: str,
    example_patch_size: int,
    alignment_patch_size: int,
    labeling_patch_size: int,
    resolution: float,
    num_labeling_images: int,
    gdal_env: Dict[str, str],
    coordinates: beam.PCollection,
    stage_prefix: str) -> Tuple[beam.PCollection, beam.PCollection]:
  """Generates examples and labeling images from source images.

  Args:
    before_image_path: Before image path.
    after_image_path: After image path.
    example_patch_size: Size of patches to extract into examples. Typically 64.
    alignment_patch_size: Size of patches used for alignment. Setting this to a
      larger value will result in more context being considered during
      before/after image alignment, and may improve the alignment result.
    labeling_patch_size: Size in pixels of before and after image patches for
      labeling.
    resolution: Desired resolution of image patches.
    num_labeling_images: Number of labeling images to generate, or 0 to disable.
    gdal_env: GDAL environment configuration.
    coordinates: Collection of coordinates (longitude, latitude, label) to
      extract examples for.
    stage_prefix: Beam stage name prefix.

  Returns:
    PCollection of examples and PCollection of labeling images.
  """

  generate_labeling_images = (num_labeling_images > 0)
  results = (
      coordinates
      | stage_prefix + '_generate_examples' >> beam.ParDo(
          GenerateExamplesFn(
              before_image_path, after_image_path, generate_labeling_images,
              example_patch_size, alignment_patch_size, labeling_patch_size,
              resolution, gdal_env)).with_outputs(_EXAMPLES, _LABELING_IMAGES))
  examples = results[_EXAMPLES]
  labeling_images = results[_LABELING_IMAGES]

  if generate_labeling_images:
    labeling_images = (
        labeling_images
        | stage_prefix + '_sample_labeling_images' >>
        beam.combiners.Sample.FixedSizeGlobally(num_labeling_images)
        | stage_prefix + '_flatten' >> beam.FlatMap(lambda x: x))

  return examples, labeling_images


def _write_import_file(output_dir: str,
                       labeling_images: beam.PCollection) -> None:
  """Creates a beam stage to write an import file for the labeling images.

  The import file contains the paths of all the generated labeling images, one
  file per line. It is needed to import the images into a Cloud dataset.

  See https://cloud.google.com/vertex-ai/docs/datasets/prepare-image.

  Args:
    output_dir: Output directory. The import file will be written here.
    labeling_images: PCollection of images. Assumes that each record is a tuple
      whose first element is the name of the file relative to the output_dir.

  """
  import_file_path = os.path.join(output_dir, 'import_file')
  _ = (
      labeling_images
      | 'get_path_name' >> beam.Map(
          lambda record: os.path.join(output_dir, record[0]))
      | 'write_import_file' >> beam.io.textio.WriteToText(
          import_file_path,
          file_name_suffix='.csv',
          num_shards=1,
          shard_name_template=''))


def generate_examples_pipeline(
    before_image_path: str,
    after_image_path: str,
    example_patch_size: int,
    alignment_patch_size: int,
    labeling_patch_size: int,
    resolution: float,
    output_dir: str,
    num_output_shards: int,
    unlabeled_coordinates: List[Tuple[float, float]],
    labeled_coordinates: List[Tuple[float, float, float]],
    use_dataflow: bool,
    num_labeling_images: int,
    gdal_env: Dict[str, str],
    dataflow_container_image: Optional[str],
    cloud_project: Optional[str],
    cloud_region: Optional[str],
    worker_service_account: Optional[str]) -> None:
  """Runs example generation pipeline.

  Args:
    before_image_path: Before image path.
    after_image_path: After image path.
    example_patch_size: Size of patches to extract into examples. Typically 64.
    alignment_patch_size: Size of patches used for alignment. Setting this to a
      larger value will result in more context being considered during
      before/after image alignment, and may improve the alignment result.
    labeling_patch_size: Size in pixels of before and after image patches for
      labeling.
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
    dataflow_container_image: Container image to use when running Dataflow.
    cloud_project: Cloud project name.
    cloud_region: Cloud region, e.g. us-central1.
    worker_service_account: Email of service account that will launch workers.
  """

  temp_dir = os.path.join(output_dir, 'temp')
  if use_dataflow:
    if cloud_project is None or cloud_region is None:
      raise ValueError(
          'cloud_project and cloud_region must be specified when using '
          'Dataflow.')
    pipeline_options = _get_dataflow_pipeline_options(cloud_project,
                                                      cloud_region, temp_dir,
                                                      dataflow_container_image,
                                                      worker_service_account)
  else:
    pipeline_options = _get_local_pipeline_options()

  with beam.Pipeline(options=pipeline_options) as pipeline:
    if unlabeled_coordinates:
      unlabeled_coordinates_pcollection = (
          pipeline
          | 'create_unlabeled_coordinates' >> beam.Create([
              _Coordinate(lng, lat, -1.0) for lng, lat in unlabeled_coordinates
          ]))

      unlabeled_examples, labeling_images = _generate_examples(
          before_image_path, after_image_path, example_patch_size,
          alignment_patch_size, labeling_patch_size, resolution,
          num_labeling_images, gdal_env, unlabeled_coordinates_pcollection,
          'unlabeled')

      unlabeled_examples_output_prefix = (
          os.path.join(output_dir, 'examples', 'unlabeled', 'unlabeled'))

      _ = (
          unlabeled_examples
          | 'write_unlabeled_examples' >> beam.io.tfrecordio.WriteToTFRecord(
              unlabeled_examples_output_prefix,
              file_name_suffix='.tfrecord',
              num_shards=num_output_shards))

      if num_labeling_images > 0:
        labeling_images_dir = (
            os.path.join(output_dir, 'examples', 'labeling_images'))
        beam_utils.write_records_as_files(labeling_images, labeling_images_dir,
                                          temp_dir, 'write_labeling_images')
        _write_import_file(labeling_images_dir, labeling_images)

    if labeled_coordinates:
      labeled_coordinates_pcollection = (
          pipeline
          | 'create_labeled_coordinates' >> beam.Create([
              _Coordinate(lng, lat, label)
              for lng, lat, label in labeled_coordinates
          ]))

      labeled_examples, _ = _generate_examples(
          before_image_path, after_image_path, example_patch_size,
          alignment_patch_size, labeling_patch_size, resolution, 0,
          gdal_env, labeled_coordinates_pcollection, 'labeled')

      labeled_examples_output_prefix = (
          os.path.join(output_dir, 'examples', 'labeled', 'labeled'))

      _ = (
          labeled_examples
          | 'write_labeled_examples' >> beam.io.tfrecordio.WriteToTFRecord(
              labeled_examples_output_prefix,
              file_name_suffix='.tfrecord',
              num_shards=num_output_shards))
