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
"""Detects buildings in tiles and return image masks as tensors.

This pipeline is agnostic to which model is used. However, it should be at least
a tensorflow model in SavedModel format. The moedl should take a HxWxC image and
output a semantic sementation of the image.

"""

import functools
import os
import shutil
import tempfile
import time
from typing import Any, Iterable, Iterator, NamedTuple

import affine
import apache_beam as beam
from apache_beam import typehints
from apache_beam.utils import multi_process_shared

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
import shapely.geometry
import shapely.wkt
from skai import detect_buildings_constants
from skai import extract_tiles_constants
from skai import utils
import skai.buildings
import tensorflow as tf
import tensorflow_addons.image as tfa_image

_ = tfa_image.connected_components(tf.ones((10, 10), tf.uint8))

Example = tf.train.Example
Metrics = beam.metrics.Metrics
PCollection = beam.pvalue.PCollection
SparseTensor = tf.sparse.SparseTensor

AffineTuple = tuple[float, float, float, float, float, float]

_OVERLAP_THRESHOLD = 0.5
_BUILDING_PIXEL_THRESHOLD = 10
_SEGMENTATION_IMAGE_MULTIPLE = 64

_BUILDINGS_TO_DEDUP = 'buildings_to_dedup'
_PASSTHROUGH_BUILDINGS = 'passthrough_buildings'

_MASK_BINARIZATION_THRESHOLD = 0.4
_MASK_TENSOR_AS_LOGITS = False


def _recursively_copy_directory(
    src_dir: str, dest_dir: str, overwrite: bool = False
) -> None:
  """Copies a directory and all files in it.

  Args:
    src_dir: Path to source directory.
    dest_dir: Path to destination directory.
    overwrite: Overwrite files.
  """
  if not tf.io.gfile.isdir(dest_dir):
    tf.io.gfile.mkdir(dest_dir)

  # For reflecting the subdir structure of the source dir.
  for src_dir_name, src_subdirs, src_leaf_files in tf.io.gfile.walk(src_dir):
    dest_dir_current_path = os.path.join(dest_dir,
                                         os.path.relpath(src_dir_name, src_dir))
    # Make the subdirectories
    for sub_dir in src_subdirs:
      tf.io.gfile.mkdir(os.path.join(dest_dir, sub_dir))

    for leaf_file in src_leaf_files:
      tf.io.gfile.copy(
          os.path.join(src_dir_name, leaf_file),
          os.path.join(dest_dir_current_path, leaf_file), overwrite)


def _load_tf_model(model_path: str) -> tf.Module:
  """Loads tensorflow model from checkpoint.

  Args:
    model_path: Path of model checkpoint. Can be directory (for SavedModel) or a
      file (for .hd5).

  Returns:
    TF model.
  """
  # If path is a directory, we can assume the model is in SavedModel format.
  if tf.io.gfile.isdir(model_path):
    local_temp_name = tempfile.mkdtemp()
    _recursively_copy_directory(model_path, local_temp_name, overwrite=True)
    model = tf.saved_model.load(local_temp_name)
    tf.io.gfile.rmtree(local_temp_name)
  else:
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as local_temp:
      local_temp_name = local_temp.name
      with tf.io.gfile.GFile(model_path, 'rb') as f:
        shutil.copyfileobj(f, local_temp)
    model = tf.saved_model.load(local_temp_name)
  return model


def _pad_image(image: np.ndarray,
               pad_to: int,
               padding_method: str | None = 'constant',
               padding_constant: int | None = 0) -> np.ndarray:
  """Pads a batch of images.

  If the difference between the original image size and the padded size is odd,
  pad_image will add the extra pixel of padding to the right and bottom of the
  image.

  Args:
    image: [width x height x channel] image to be padded.
    pad_to: Image size to pad to.
    padding_method: Method for padding image. Defaults to 'constant'.
    padding_constant: Constant value to pad with. Defaults to black.

  Returns:
    The padded image.
  """

  padding = []
  for size in image.shape[:2]:
    pad_diff = (pad_to - size) // 2
    remainder = (pad_to - size) % 2
    padding.append((pad_diff, pad_diff + remainder))
  padding.append((0, 0))

  # Default to pad with black.
  return np.pad(
      image, padding, padding_method, constant_values=padding_constant)


def _recrop_mask(
    examples: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
  """Crops a batch of masks (N, X, X, C) to a given size.

  If the difference between the input image and the desired crop size is odd,
  the crop will not be centered.The bottom / right cropped out borders will be
  one pixel wider/taller than the left / top borders.

  Args:
    examples: 4D tensor representing a batch of building masks.
    new_width: The desired width of the tile.
    new_height: The desired height of the tile.

  Raises:
    ValueError: If crop size is larger than original.

  Returns:
    Tensor of cropped images.
  """
  if examples.shape[1] < new_width or examples.shape[2] < new_height:
    raise ValueError("Crop size can't be larger than original size")

  width_offset = (examples.shape[1] - new_width) // 2
  height_offset = (examples.shape[2] - new_height) // 2
  return examples[:, width_offset:width_offset + new_width,
                  height_offset:height_offset + new_height, :]


def _pad_to_square_multiple_of(
    image: np.ndarray, multiple_of: int) -> np.ndarray:
  """Pads an image to the nearest multiple of the specified constant.

  Output will be a square image.

  Args:
    image: [width x height x channels] sized image to be padded.
    multiple_of: Desired constant for the output to be a multiple of.

  Returns:
    The padded image.
  """
  if (image.shape[0] % multiple_of == 0) and (image.shape[0] == image.shape[1]):
    return image

  max_size = max(image.shape[0], image.shape[1])
  quotient = (max_size + multiple_of - 1) // multiple_of
  target_size = multiple_of * quotient
  return _pad_image(image, target_size, 'constant')


def ss_to_is_connected_components(
    segmentation_mask: tf.Tensor,
    confidence_mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Converts semantic segmentation features to instance segmentation.

  Each instance has a mask, score and label.

  Instances are generated by finding connected components in an undirected graph
  that is represented using the "segmentation_mask" feature. Vertices are pixels
  with nonzero label and their neighbors are all touching horizontally or
  vertically pixels (up to 4) that have the same label in the
  "segmentation_mask".

  Scores of instances are calculated using an average of overlapping values in
  the confidence mask for the matching label and the instance mask.

  Args:
    segmentation_mask: A tensor of shape [1, height, width, 1], where each value
      represents an assigned label for that pixel.
    confidence_mask: A tensor of shape [1, height, width, number of classes],
      where each value represents probability of a pixel having a specific
      label.

  Returns:
    A three-tuple of tensors containing, in this order:
      "instance_masks": A tensor of shape [1, num instances, height, width, 1]
        with instance masks.
      "instance_scores: A tensor of shape [1, num instances] with scores.
      "instance_labels": A tensor of shape [1, num instances] with labels.
  """
  tf.ensure_shape(segmentation_mask, [1, None, None, 1])
  seg_shape = tf.shape(segmentation_mask)
  tf.ensure_shape(confidence_mask, [1, seg_shape[1], seg_shape[2], None])

  segmentation_mask = tf.squeeze(segmentation_mask, [0, 3])
  num_instances, merged_instance_mask = cv2.connectedComponents(
      segmentation_mask.numpy().astype(np.int8)
  )
  num_instances -= 1

  masks = tf.cast(
      tf.expand_dims(merged_instance_mask, 0) == tf.reshape(
          tf.range(1, num_instances + 1), [num_instances, 1, 1]), tf.int32)
  labels = tf.math.reduce_max(segmentation_mask * masks, axis=[1, 2])
  # Scores are calculated by taking the mean of the confidence mask pixels (with
  # the matching label) that overlap the instance masks.
  confidence_mask_t = tf.transpose(tf.squeeze(confidence_mask, 0), [2, 0, 1])
  scores_sum = tf.math.reduce_sum(
      tf.gather(confidence_mask_t, labels) * tf.cast(masks, tf.float32),
      axis=[1, 2])
  scores = scores_sum / tf.cast(
      tf.math.reduce_sum(masks, axis=[1, 2]), tf.float32)

  masks = tf.cast(masks, tf.int8)
  return (tf.expand_dims(tf.expand_dims(masks, 0),
                         -1), tf.expand_dims(scores,
                                             0), tf.expand_dims(labels, 0))


def _pixel_xy_to_long_lat(
    x: Iterable[int],
    y: Iterable[int],
    crs: str,
    affine_transform: AffineTuple
) -> list[tuple[float, float]]:
  """Convert row, column offsets in pixel space to longitude, latitude.

  Args:
    x: Pixel x (column) offsets.
    y: Pixel y (row) offsets.
    crs: Coordinate reference system of image.
    affine_transform: Affine transform of the image.

  Returns:
    list of longitude, latitude tuples.
  """
  tx, ty = rasterio.transform.xy(affine.Affine(*affine_transform), y, x)
  if crs.lower() != 'epsg:4326':
    transformer = pyproj.Transformer.from_crs(
        crs.lower(), 'epsg:4326', always_xy=True)
    tx, ty = transformer.transform(tx, ty, errcheck=True)
  return list(zip(tx, ty))


def _get_mask_polygon(
    mask: np.ndarray,
    x_offset: int,
    y_offset: int,
    crs: str,
    affine_transform: AffineTuple) -> shapely.geometry.Polygon | None:
  """Returns a polygon around the mask."""
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
  x = contours[0][:, 0, 0] + x_offset
  y = contours[0][:, 0, 1] + y_offset
  coords = _pixel_xy_to_long_lat(x, y, crs, affine_transform)
  try:
    return shapely.geometry.Polygon(coords)
  except ValueError:
    return None


def _get_mask_bounds(sparse_mask) -> tuple[int, int, int, int]:
  xs = sparse_mask.indices[:, 1].numpy()
  ys = sparse_mask.indices[:, 0].numpy()
  return min(xs), min(ys), max(xs), max(ys)


def _construct_output_examples(
    instance_masks: tf.Tensor,
    instance_scores: tf.Tensor,
    input_example: Example,
) -> Iterable[Example]:
  """Packages instance data into final output example.

  Args:
    instance_masks: A tensor of shape [num instances, height, width, 1].
    instance_scores: A tensor of shape [num instances].
    input_example: The example passed to the stage containing information that
      we want to preserve in the output.

  Yields:
    An example for each building instance.
  """
  num_detections = instance_masks.shape[0]
  if num_detections == 0:
    return

  image_path = utils.get_bytes_feature(
      input_example, extract_tiles_constants.IMAGE_PATH
  )[0].decode()
  pixel_x_offset = utils.get_int64_feature(
      input_example, extract_tiles_constants.X_OFFSET
  )[0]
  pixel_y_offset = utils.get_int64_feature(
      input_example, extract_tiles_constants.Y_OFFSET
  )[0]
  margin_size = utils.get_int64_feature(
      input_example, extract_tiles_constants.MARGIN_SIZE
  )[0]
  tile_row = utils.get_int64_feature(
      input_example, extract_tiles_constants.TILE_ROW
  )[0]
  tile_col = utils.get_int64_feature(
      input_example, extract_tiles_constants.TILE_COL
  )[0]
  crs = utils.get_bytes_feature(
      input_example, extract_tiles_constants.CRS
  )[0].decode()
  affine_transform = input_example.features.feature[
      extract_tiles_constants.AFFINE_TRANSFORM
  ].float_list.value
  for i in range(num_detections):
    # ss_to_connected_components returns masks with shape
    # [1, num instances, h, w, 1]
    # Here we collapse the final dimensions of the instance mask to
    # [h, w]
    instance_mask = tf.squeeze(instance_masks[i, :, :, 0])
    # Collapse the instance score to a single element.
    instance_score = tf.squeeze(instance_scores[i])
    output_example = tf.train.Example()
    sparse_instance_mask = tf.sparse.from_dense(instance_mask)

    mask_image = tf.expand_dims(tf.cast(instance_mask, tf.uint8), -1) * 255
    mask_png = tf.io.encode_png(mask_image).numpy()

    mask_polygon = _get_mask_polygon(
        instance_mask.numpy().astype(np.uint8),
        pixel_x_offset,
        pixel_y_offset,
        crs,
        affine_transform,
    )
    if not mask_polygon:
      Metrics.counter('skai', 'invalid_mask_polygon').inc()
      continue
    mask_poly_wkt = shapely.wkt.dumps(mask_polygon)

    # Compute mask centroid.
    min_x, min_y, max_x, max_y = _get_mask_bounds(sparse_instance_mask)
    centroid_x = (min_x + max_x) // 2
    centroid_y = (min_y + max_y) // 2
    longitude, latitude = _pixel_xy_to_long_lat(
        [pixel_x_offset + centroid_x],
        [pixel_y_offset + centroid_y],
        crs,
        affine_transform,
    )[0]

    # Check if this mask is touching the edge of the image.
    on_edge = int(
        min_x == 0
        or min_y == 0
        or max_x == mask_image.shape[1] - 1
        or max_y == mask_image.shape[0] - 1
    )
    utils.add_int64_feature(
        detect_buildings_constants.ON_EDGE, on_edge, output_example
    )

    # Compute "area".
    area = len(sparse_instance_mask.indices)
    if area < _BUILDING_PIXEL_THRESHOLD:
      Metrics.counter('skai', 'num_buildings_below_pixel_threshold').inc()
      continue

    # serialize_sparse returns a 1-D tensor with 3 elements representing the
    # serialized indices, values, and shape, respectively.
    for sparse_data in tf.io.serialize_sparse(sparse_instance_mask):
      utils.add_bytes_feature(detect_buildings_constants.MASK,
                              sparse_data.numpy(), output_example)
    utils.add_bytes_feature('image', mask_png, output_example)
    utils.add_float_feature(detect_buildings_constants.CONFIDENCE,
                            instance_score, output_example)
    utils.add_float_feature(detect_buildings_constants.LONGITUDE,
                            longitude, output_example)
    utils.add_float_feature(detect_buildings_constants.LATITUDE,
                            latitude, output_example)
    utils.add_float_feature(detect_buildings_constants.AREA,
                            area, output_example)
    utils.add_bytes_feature('mask_wkt', mask_poly_wkt.encode(), output_example)

    # Mask boundary info for easier debugging.
    utils.add_int64_feature('min_x', min_x, output_example)
    utils.add_int64_feature('min_y', min_y, output_example)
    utils.add_int64_feature('max_x', max_x, output_example)
    utils.add_int64_feature('max_y', max_y, output_example)
    utils.add_int64_feature('centroid_x', centroid_x, output_example)
    utils.add_int64_feature('centroid_y', centroid_y, output_example)

    # Pass through tile data.
    utils.add_bytes_feature(
        detect_buildings_constants.IMAGE_PATH,
        image_path.encode(),
        output_example,
    )
    utils.add_int64_feature(detect_buildings_constants.TILE_PIXEL_COL,
                            pixel_x_offset, output_example)
    utils.add_int64_feature(detect_buildings_constants.TILE_PIXEL_ROW,
                            pixel_y_offset, output_example)
    utils.add_int64_feature(detect_buildings_constants.TILE_COL, tile_col,
                            output_example)
    utils.add_int64_feature(detect_buildings_constants.TILE_ROW, tile_row,
                            output_example)
    utils.add_int64_feature(detect_buildings_constants.MARGIN_SIZE,
                            margin_size, output_example)
    utils.add_bytes_feature(
        detect_buildings_constants.CRS, crs.encode(), output_example
    )
    utils.add_float_list_feature(detect_buildings_constants.AFFINE_TRANSFORM,
                                 affine_transform, output_example)

    yield output_example


def _extract_confidence_masks(
    image_height: int,
    image_width: int,
    input_tuple: tuple[tf.Tensor, tf.Tensor],
) -> tf.Tensor:
  """Extracts instance confidence masks.

  The raw instance confidence masks represent cropped out to bounding box and
  then scaled masks, therefore we need to reverse that transformation to get
  masks that span the whole image.

  Args:
    image_height: The input image height.
    image_width: The input image width.
    input_tuple: A tuple of unnormalized bounding box and raw instance
      confidence masks.

  Returns:
    Properly scaled instance confidence masks.
  """
  bbox, raw_confidence_mask = input_tuple
  bbox = tf.cast(tf.math.round(bbox), tf.int32)
  ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
  raw_confidence_mask = tf.expand_dims(raw_confidence_mask, axis=-1)
  cropped_confidence_mask = tf.image.resize(
      raw_confidence_mask, [ymax - ymin + 1, xmax - xmin + 1]
  )
  confidence_mask = tf.pad(
      cropped_confidence_mask,
      [[ymin, image_height - 1 - ymax], [xmin, image_width - 1 - xmax], [0, 0]],
  )
  return tf.image.resize(confidence_mask, (image_height, image_width))


def _extract_masks_and_scores(
    model_output: dict[str, tf.Tensor],
    image_height: int,
    image_width: int,
    detection_confidence_threshold: float,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Extracts instance confidence masks and scores from the model output.

  Args:
    model_output: The output of the building detection model.
    image_height: Width of the input image.
    image_width: Height of the input image.
    detection_confidence_threshold: All instances below this detection score
      will be dropped.

  Returns:
      A tuple of instance confidence masks and scores.
  """
  num_detections = tf.cast(model_output['num_detections'][0], tf.int32)

  raw_bboxes = model_output['detection_outer_boxes'][0, :num_detections]

  # Extract instance confidence masks.
  raw_confidence_masks = model_output['detection_masks'][0, :num_detections]
  if _MASK_TENSOR_AS_LOGITS:
    raw_confidence_masks = tf.math.sigmoid(raw_confidence_masks)

  extract_fn = functools.partial(
      _extract_confidence_masks,
      image_height,
      image_width,
  )
  if num_detections > 0:
    confidence_masks = tf.map_fn(
        extract_fn,
        (raw_bboxes, raw_confidence_masks),
        fn_output_signature=tf.float32,
    )
  else:
    # confidence_masks after map_fn with 0 detections had wrong shape
    confidence_masks = tf.zeros([0, image_height, image_width, 1])

  scores = model_output['detection_scores'][0, :num_detections]

  # Binarize confidence masks.
  masks = tf.cast(confidence_masks >= _MASK_BINARIZATION_THRESHOLD, tf.int32)
  # Filter out low confidence detections.
  is_high_confidence = scores >= detection_confidence_threshold
  masks = masks[is_high_confidence]
  scores = scores[is_high_confidence]

  return masks, scores


@typehints.with_output_types(tf.train.Example)
class DetectBuildingsFn(beam.DoFn):
  """Detects buildings within the given tiles and returns as discrete confidence masks."""

  def __init__(
      self, model_path: str, detection_confidence_threshold: float
  ) -> None:
    self._model_path = model_path
    self._detection_confidence_threshold = detection_confidence_threshold
    self._num_detected_buildings = Metrics.counter('skai',
                                                   'num_detected_buildings')

  def _make_dummy_input(self):
    image = np.zeros(
        (_SEGMENTATION_IMAGE_MULTIPLE, _SEGMENTATION_IMAGE_MULTIPLE, 3),
        dtype=np.uint8
    )
    example = tf.train.Example()
    example.features.feature[
        extract_tiles_constants.IMAGE_ENCODED
    ].bytes_list.value.append(tf.io.encode_png(image).numpy())
    return tf.constant([example.SerializeToString()])

  def setup(self) -> None:
    # Use a shared handle so that the model is only loaded once per worker and
    # shared by all processing threads. For more details, see
    #
    # https://medium.com/google-cloud/cache-reuse-across-dofns-in-beam-a34a926db848
    def load():
      start_time = time.time()
      Metrics.counter('skai', 'load_model_start').inc()
      if self._model_path:
        model = _load_tf_model(self._model_path)
        _ = model(self._make_dummy_input())
      else:
        model = None
      Metrics.counter('skai', 'load_model_end').inc()
      elapsed_time = time.time() - start_time
      Metrics.distribution('skai', 'model_load_time').update(elapsed_time)
      return model

    Metrics.counter('skai', 'setup_start').inc()
    self._model = multi_process_shared.MultiProcessShared(
        load, self._model_path
    ).acquire()
    Metrics.counter('skai', 'setup_end').inc()

  def process(self, example: Example) -> Iterator[Example]:
    """Runs building detection model on a single tile, encoded as a tf.Example.

    Args:
      example: Image tile stored as a tf.Example.

    Yields:
      Example containing the following features:
        - The building instances represented as a
          (tile width, tile height, number of buildings) tensor, serialized as a
          bytes.
        - The building instance mask, represented as a serialized sparse tensor.
        - The x,y coordinates of the input tile from the original grid.
        - The mapping from pixel to coordinates represented as a affine
          transformation matrix.
    """
    Metrics.counter('skai', 'process_example').inc()
    if self._model:
      image = tf.io.decode_image(
          utils.get_bytes_feature(
              example, extract_tiles_constants.IMAGE_ENCODED
          )[0],
          dtype=tf.uint8,
      ).numpy()

      if (image.shape[0] % _SEGMENTATION_IMAGE_MULTIPLE != 0) or (
          image.shape[1] % _SEGMENTATION_IMAGE_MULTIPLE != 0
      ):
        # Current segmentation model expects images that are a multiple of 64.
        image = _pad_to_square_multiple_of(image, _SEGMENTATION_IMAGE_MULTIPLE)
        example.features.feature[
            extract_tiles_constants.IMAGE_ENCODED
        ].bytes_list.value[0] = tf.io.encode_png(image).numpy()

      serialized_example = tf.constant([example.SerializeToString()])

      model_output = self._model(serialized_example)

      instance_masks, instance_scores = _extract_masks_and_scores(
          model_output,
          image.shape[0],
          image.shape[1],
          self._detection_confidence_threshold,
      )
      self._num_detected_buildings.inc(len(instance_scores))
    else:
      instance_masks = tf.constant([], shape=(0, 0, 0, 1))
      instance_scores = tf.constant([], shape=(0,))

    Metrics.counter('skai', 'examples_processed').inc()
    yield from _construct_output_examples(
        instance_masks, instance_scores, example
    )


def _get_regions_overlapped(mask: SparseTensor, margin_size: int) -> list[int]:
  """Computes which tile regions a building mask overlaps.

  Args:
    mask: Building mask as a SparseTensor.
    margin_size: Size of the margin in pixels.

  Returns:
    A list of 9 integers, where the kth element is the number of mask pixels in
    that region. 0 pixels means no overlap.
  """
  tile_height = mask.dense_shape[0].numpy()
  tile_width = mask.dense_shape[1].numpy()

  # Region size is the margin size * 2 because the region includes the margin of
  # the current tile AND the margin of the adjacent tile. For example, let's say
  # there are two tiles, X and Y, and X is directly on top of Y. This diagram
  # explains how X and Y overlap.
  #
  #     +-->   -----------------  Top edge of tile Y
  #     |
  #     |      Y's top margin
  # Overlap
  #  region    =================  Where the central regions of X and Y touch
  #     |
  #     |      X's bottom margin
  #     |
  #     +-->   -----------------  Bottom edge of tile X
  region_size = margin_size * 2

  row_starts = [
      0,
      region_size,
      tile_height - region_size,
  ]
  col_starts = [
      0,
      region_size,
      tile_width - region_size,
  ]
  row_sizes = [
      region_size,
      tile_height - 2 * region_size,
      region_size,
  ]
  col_sizes = [
      region_size,
      tile_width - 2 * region_size,
      region_size,
  ]
  output = []
  for region in range(9):
    row = int(region // 3)
    col = int(region % 3)
    start = (row_starts[row], col_starts[col])
    size = (row_sizes[row], col_sizes[col])
    output.append(len(tf.sparse.slice(mask, start=start, size=size).indices))
  return output


def _encode_sparse_tensor(sparse_tensor: SparseTensor, example: Example,
                          feature_name: str) -> None:
  """Encodes a SparseTensor into a TF Example.

  Args:
    sparse_tensor: SparseTensor to encode.
    example: TF Example to encode into.
    feature_name: Feature name to encode example in.
  """
  for sparse_data in tf.io.serialize_sparse(sparse_tensor):
    utils.add_bytes_feature(feature_name, sparse_data.numpy(), example)


def _decode_sparse_tensor(
    example: Example, feature_name: str, dtype: tf.DType
) -> SparseTensor:
  """Decodes a SparseTensor stored in a TF Example.

  Args:
    example: TF Example to decode.
    feature_name: Feature name of bytes feature encoding the SparseTensor.
    dtype: Data type of the SparseTensor to decode.

  Returns:
    Decoded SparseTensor.
  """
  t = tf.io.deserialize_many_sparse(
      [example.features.feature[feature_name].bytes_list.value], dtype
  )
  # This introduces an extra batch dimension to the encoded SparseTensor. Get
  # rid of it.
  u = tf.sparse.reshape(t, t.dense_shape[1:])
  return u


def augment_overlap_region(building: Example) -> Example:
  """Identifies the tile region(s) that the building touches.

  For the purposes of parallelizing building deduplication, each tile is divided
  into 9 regions as follows:

  +---------+
  |0|  1  |2|
  |-+-----+-|
  | |     | |
  |3|  4  |5|
  | |     | |
  |-+-----+-|
  |6|  7  |8|
  +---------+

  The width of 3 and 5 and the height of 1 and 7 equal "margin_size".  A
  building in the tile will touch one or more regions. We group the regions into
  "stages" as follows:

    Stage 0       Stage 1       Stage 2       Stage 3
  +---------+   +---------+   +---------+   +---------+
  |X|     |X|   | |     | |   | |  X  | |   | |     | |
  |-+-----+-|   |-+-----+-|   |-+-----+-|   |-+-----+-|
  | |     | |   | |     | |   | |     | |   | |     | |
  | |     | |   |X|     |X|   | |     | |   | |  X  | |
  | |     | |   | |     | |   | |     | |   | |     | |
  |-+-----+-|   |-+-----+-|   |-+-----+-|   |-+-----+-|
  |X|     |X|   | |     | |   | |  X  | |   | |     | |
  +---------+   +---------+   +---------+   +---------+

  This way, no building can touch more than one region in each stage, assuming
  that no building is larger than "margin_size" pixels.

  A tile is identified by the tuple (row index, column index). For tile (i, j),
  each region will be named as follows:

  0: (i-0.5, j-0.5)
  1: (i-0.5, j)
  2: (i-0.5, j+0.5)
  3: (i,     j-0.5)
  4: (i,     j)
  5: (i,     j+0.5)
  6: (i+0.5, j-0.5)
  7: (i+0.5, j)
  8: (i+0.5, j+0.5)

  This way, region 3 for tile (i, j) will share a name with region 5 of tile (i,
  j-1). This is a desirable property because the buildings in those regions need
  to be compared to find duplicates.

  This function adds "dedup_stage_n_region" features to the input building
  Example, where n = 0, 1, 2, 3.

  Args:
    building: TF Example containing building mask and tile index.

  Returns:
    Identical TF Example but with "dedup_stage_n_region" features.
  """
  stage_to_regions = {0: [0, 2, 6, 8], 1: [3, 5], 2: [1, 7], 3: [4]}

  tile_row = utils.get_int64_feature(
      building, detect_buildings_constants.TILE_ROW
  )[0]
  tile_col = utils.get_int64_feature(
      building, detect_buildings_constants.TILE_COL
  )[0]

  region_coords = {
      0: (tile_row - 0.5, tile_col - 0.5),
      1: (tile_row - 0.5, tile_col),
      2: (tile_row - 0.5, tile_col + 0.5),
      3: (tile_row, tile_col - 0.5),
      4: (tile_row, tile_col),
      5: (tile_row, tile_col + 0.5),
      6: (tile_row + 0.5, tile_col - 0.5),
      7: (tile_row + 0.5, tile_col),
      8: (tile_row + 0.5, tile_col + 0.5),
  }

  margin_size = utils.get_int64_feature(
      building, detect_buildings_constants.MARGIN_SIZE
  )[0]
  mask = _decode_sparse_tensor(
      building, detect_buildings_constants.MASK, dtype=tf.int32
  )
  overlaps = _get_regions_overlapped(mask, margin_size)
  if not any(overlaps):
    raise ValueError('Mask does not overlap any regions.')

  augmented = tf.train.Example()
  augmented.CopyFrom(building)
  for stage in range(4):
    regions_touched = [r for r in stage_to_regions[stage] if overlaps[r]]
    if not regions_touched:
      continue
    if len(regions_touched) > 1:
      regions_touched.sort(key=lambda r: overlaps[r], reverse=True)
      Metrics.counter('skai', 'dedup_mask_touches_multiple_regions').inc()
    feature = f'dedup_stage_{stage}_region'
    augmented.features.feature[feature].float_list.value[:] = region_coords[
        regions_touched[0]
    ]
  return augmented


class _ExtractBuildingsForStage(beam.DoFn):
  """Extracts the buildings that should be deduped for a particular stage.

  This is a multi-output DoFn. Outputs the buildings for the target stage as a
  tuple (overlap region, building example) in the "buildings_to_dedup"
  output. All other buildings are simply passed through to the
  "passthrough_buildings" output.
  """

  def __init__(self, stage: str):
    """Creates DoFn.

    Args:
      stage: Target dedup stage.
    """

    self._stage = stage
    self._region_feature = f'dedup_stage_{self._stage}_region'

  def process(self, building: Example):
    """Extracts the buildings that should be deduped for a particular stage.

    Args:
      building: TF Example for the building. Should have a "overlap_region"
        float feature.

    Yields:
      (overlap region, building) if building should be deduped at target stage,
      otherwise just the building itself.
    """
    region = building.features.feature[self._region_feature].float_list.value[:]
    if region:
      yield beam.pvalue.TaggedOutput(_BUILDINGS_TO_DEDUP, (region, building))
    else:
      yield beam.pvalue.TaggedOutput(_PASSTHROUGH_BUILDINGS, building)


def _extract_buildings_for_stage(buildings: PCollection,
                                 stage: int) -> tuple[PCollection, PCollection]:
  """Extracts the buildings that should be deduped for a particular stage.

  Args:
    buildings: PCollection of TF Examples representing buildings to dedup.
    stage: Stage number.

  Returns:
    A tuple of PCollections where the first item is the buildings to dedup at
    this stage, and the second PCollection is a list of buildings that should be
    passed on untouched.
  """
  results = (
      buildings
      | f'extract_buildings_stage_{stage}' >> beam.ParDo(
          _ExtractBuildingsForStage(stage)).with_outputs(
              _BUILDINGS_TO_DEDUP, _PASSTHROUGH_BUILDINGS))
  return results[_BUILDINGS_TO_DEDUP], results[_PASSTHROUGH_BUILDINGS]


def _indices_to_set(sparse_tensor: SparseTensor) -> set[tuple[int, int]]:
  return set(tuple(i) for i in sparse_tensor.indices.numpy())


def _get_global_mask(building: Example) -> set[tuple[int, int]]:
  """Extracts the global building mask from the input example.

  The global pixel coordinates are obtained by adding the tile's row and column
  pixel offsets to the indices of the mask sparse tensor.

  Args:
    building: TF Example representing the building.

  Returns:
    Building mask as a set of global (row, col) tuples.
  """
  mask_tensor = _decode_sparse_tensor(
      building, detect_buildings_constants.MASK, dtype=tf.int32
  )
  indices = mask_tensor.indices.numpy()
  indices[:, 0] += utils.get_int64_feature(
      building, detect_buildings_constants.TILE_PIXEL_ROW
  )[0]
  indices[:, 1] += utils.get_int64_feature(
      building, detect_buildings_constants.TILE_PIXEL_COL
  )[0]
  return set(tuple(i) for i in indices)


def _masks_overlap(mask1, mask2) -> bool:
  num_intersecting_pixels = len(mask1 & mask2)
  return ((num_intersecting_pixels / len(mask1) >= _OVERLAP_THRESHOLD) or
          (num_intersecting_pixels / len(mask2) >= _OVERLAP_THRESHOLD))


def _nms_score(building: Example) -> float:
  """Calculates score for ranking in non-max suppression.
  """
  confidence = utils.get_float_feature(
      building, detect_buildings_constants.CONFIDENCE
  )[0]
  edge_penalty = (
      utils.get_int64_feature(building, detect_buildings_constants.ON_EDGE)[0]
      * -100
  )

  return confidence + edge_penalty


def non_max_suppression(
    region_key: Any, region_buildings: Iterable[Example]) -> Iterable[Example]:
  """Deduplicate buildings using NMS.

  Args:
    region_key: Region key. Unused.
    region_buildings: Sequence of building Examples in region.

  Yields:
    Deduplicated buildings.
  """
  del region_key  # Unused.

  duplicate_buildings = Metrics.counter('skai', 'duplicate_buildings_removed')
  # Create a list of sets, where each set contains the pixel coordinates for one
  # building mask. Then we can use (set intersection / set size) to compute
  # amount of overlap.
  buildings = list(region_buildings)
  masks = [_get_global_mask(b) for b in buildings]
  scores = [_nms_score(b) for b in buildings]
  indexes = list(np.argsort(scores))
  while indexes:
    best = indexes[-1]
    yield buildings[best]
    new_indexes = [
        i for i in indexes[:-1] if not _masks_overlap(masks[best], masks[i])
    ]
    num_duplicates = len(indexes) - len(new_indexes) - 1
    if num_duplicates > 0:
      duplicate_buildings.inc(num_duplicates)
    indexes = new_indexes


def _process_stage(buildings: PCollection, stage: int) -> PCollection:
  """Runs one stage of building deduplication.

  Args:
    buildings: PCollection of buildings to be deduplicated.
    stage: Stage number.

  Returns:
    PCollection of buildings where buildings assigned to current stage have been
    deduplicated.
  """
  buildings_to_dedup, passthrough_buildings = _extract_buildings_for_stage(
      buildings, stage)
  deduped_buildings = (
      buildings_to_dedup
      | f'group_by_region_stage_{stage}' >> beam.GroupByKey()
      | f'deduplicate_stage_{stage}' >> beam.FlatMapTuple(non_max_suppression))
  merged = ((deduped_buildings, passthrough_buildings)
            | f'merge_buildings_stage_{stage}' >> beam.Flatten())
  return merged


def deduplicate_buildings(buildings: PCollection) -> PCollection:
  """Deduplicates buildings.

  Args:
    buildings: PCollection of buildings to be deduplicated.

  Returns:
    PCollection of deduplicated buildings.
  """
  buildings = (
      buildings
      | 'AugmentOverlapRegion' >> beam.Map(augment_overlap_region))

  for stage in (0, 1, 2, 3):
    buildings = _process_stage(buildings, stage)
  return buildings


def deduplicate_buildings_simple(buildings: PCollection) -> PCollection:
  """Deduplicates buildings.

  Args:
    buildings: PCollection of buildings to be deduplicated.

  Returns:
    PCollection of deduplicated buildings.
  """
  buildings = (
      buildings
      | 'AugmentOverlapRegion' >> beam.Map(augment_overlap_region))

  deduped_buildings = (
      buildings
      | 'add_key' >> beam.Map(lambda x: (None, x))
      | 'group_all_buildings' >> beam.GroupByKey()
      | 'non_max_suppression' >> beam.FlatMapTuple(non_max_suppression))
  return deduped_buildings


def write_tfrecords(
    examples: PCollection,
    output_prefix: str,
    num_shards: int,
    stage_prefix: str,
) -> None:
  """Writes examples as sharded TFRecords.

  Args:
    examples: PCollection of Tensorflow Examples.
    output_prefix: Output path prefix.
    num_shards: Number of shards to write.
    stage_prefix: Beam stage name prefix.
  """
  _ = (
      examples
      | stage_prefix + 'Serialize' >> beam.Map(lambda e: e.SerializeToString())
      | stage_prefix + 'WriteOutput' >> beam.io.tfrecordio.WriteToTFRecord(
          output_prefix,
          file_name_suffix='.tfrecord',
          num_shards=num_shards))


class BuildingRow(NamedTuple):
  longitude: float
  latitude: float
  confidence: float
  area: float
  wkt: str
  image_path: str


def _write_files(rows: list[BuildingRow], output_prefix: str) -> None:
  """Writes combined building rows to CSV, GeoPackage, and GeoParquet formats.

  Args:
    rows: List of building rows.
    output_prefix: Output files prefix.
  """
  df = pd.DataFrame(rows)
  with tf.io.gfile.GFile(f'{output_prefix}.csv', 'w') as f:
    df.to_csv(f, index=False)

  gdf = gpd.GeoDataFrame(
      df.drop(columns=['wkt']),
      geometry=df['wkt'].apply(shapely.wkt.loads),
      crs='EPSG:4326',
  )
  with tf.io.gfile.GFile(f'{output_prefix}.gpkg', 'wb') as f:
    gdf.to_file(f, driver='GPKG', engine='fiona')
  skai.buildings.write_buildings_file(gdf, f'{output_prefix}.parquet')


def write_examples_to_files(buildings: PCollection, output_prefix: str) -> None:
  _ = (
      buildings
      | 'to_rows'
      >> beam.Map(
          lambda example: BuildingRow(
              longitude=utils.get_float_feature(
                  example, detect_buildings_constants.LONGITUDE
              )[0],
              latitude=utils.get_float_feature(
                  example, detect_buildings_constants.LATITUDE
              )[0],
              confidence=utils.get_float_feature(
                  example, detect_buildings_constants.CONFIDENCE
              )[0],
              area=utils.get_float_feature(
                  example, detect_buildings_constants.AREA
              )[0],
              wkt=utils.get_bytes_feature(example, 'mask_wkt')[0].decode(),
              image_path=utils.get_bytes_feature(
                  example, detect_buildings_constants.IMAGE_PATH
              )[0].decode(),
          )
      )
      | 'combine_rows' >> beam.transforms.combiners.ToList()
      | 'write_files' >> beam.Map(_write_files, output_prefix=output_prefix)
  )
