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
"""Tests functions in detect_buildings module."""

import os

from typing import List, Tuple

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
import numpy as np
import pandas as pd
import PIL.Image
from skai import detect_buildings
from skai import detect_buildings_constants
from skai import extract_tiles_constants
from skai import utils
import tensorflow as tf2
import tensorflow.compat.v1 as tf

Example = tf2.train.Example
SparseTensor = tf2.sparse.SparseTensor
gfile = tf2.io.gfile


def _create_building_mask(tile_height: int, tile_width: int,
                          indices: List[Tuple[int, int]]) -> SparseTensor:
  values = np.ones((len(indices),), dtype=np.int32)
  return SparseTensor(
      indices=indices, values=values, dense_shape=[tile_height, tile_width])


def _create_building_example(tile_row: int, tile_col: int, tile_pixel_row: int,
                             tile_pixel_col: int, tile_height: int,
                             tile_width: int, margin_size: int,
                             confidence: float,
                             indices: List[Tuple[int, int]]) -> Example:
  e = Example()
  e.features.feature[
      detect_buildings_constants.TILE_ROW].int64_list.value.append(tile_row)
  e.features.feature[
      detect_buildings_constants.TILE_COL].int64_list.value.append(tile_col)
  e.features.feature[detect_buildings_constants
                     .TILE_PIXEL_ROW].int64_list.value.append(tile_pixel_row)
  e.features.feature[detect_buildings_constants
                     .TILE_PIXEL_COL].int64_list.value.append(tile_pixel_col)

  e.features.feature[detect_buildings_constants
                     .MARGIN_SIZE].int64_list.value.append(margin_size)
  e.features.feature[
      detect_buildings_constants.CONFIDENCE].float_list.value.append(confidence)

  e.features.feature[
      detect_buildings_constants.CRS].bytes_list.value.append(b'epsg:4326')
  e.features.feature[
      detect_buildings_constants.AFFINE_TRANSFORM].float_list.value.extend(
          [0.0] * 6)
  e.features.feature[
      detect_buildings_constants.ON_EDGE
  ].int64_list.value.append(0)
  e.features.feature[
      detect_buildings_constants.IMAGE_PATH
  ].bytes_list.value.append(b'image_path')

  mask_tensor = _create_building_mask(tile_height, tile_width, indices)
  detect_buildings._encode_sparse_tensor(mask_tensor, e,
                                         detect_buildings_constants.MASK)
  return e


def _get_float_feature(example: Example, feature_name: str) -> List[float]:
  return list(example.features.feature[feature_name].float_list.value)


def _create_fake_module(empty_detection: bool = False) -> tf.Module:
  """Creates a mock segmentation model.

  Args:
    empty_detection: If true, the model will return an empty detection;
      otherwise it will return detections equivalent to the mmeka model.

  Returns:
    A simple model that will always return a tensor with the same size as the
    input image, but with two channels for the building, background classes.
  """

  def _call_func_empty_detection(_):
    return {
        'num_detections': tf.zeros([1]),  # Zero detections.
        'detection_outer_boxes': tf.zeros([1, 100, 4]),
        'detection_masks': tf.zeros([1, 100, 128, 128]),
        'detection_scores': tf.zeros([1, 100]),
    }

  def _call_func(serialized_example):
    # Add 3 channels to first dimension to represent 3 building detections.
    # So output should have a shape of [3, H, W, 2] where 2 is the number of
    # classes (building, background).

    # Here we create a building in the center of the image.
    example = tf.io.parse_single_example(
        serialized_example[0], extract_tiles_constants.FEATURES
    )
    image = tf.io.decode_image(
        example[extract_tiles_constants.IMAGE_ENCODED], dtype=tf.float32
    )
    tile_shape = tf2.shape(image)

    building = tf2.constant(0.75, shape=(10, 10))
    background = tf2.constant(0.25, shape=(10, 10))
    confidence_mask = tf2.stack([background, building], axis=-1)
    pad_diff = (tile_shape[1] - tf2.shape(building)[0]) // 2
    remainder = (tile_shape[1] - tf2.shape(building)[0]) % 2
    padding = [
        [pad_diff, pad_diff + remainder],
        [pad_diff, pad_diff + remainder],
        [0, 0],
    ]

    building_mask = tf2.pad(
        confidence_mask, padding, 'constant', constant_values=0
    )
    building_mask = tf.math.sigmoid(building_mask)

    bboxes = tf2.constant([[
        [5, 5, 70, 70],
    ]])

    saved_model_output = {
        'num_detections': tf.constant([1]),
        'detection_outer_boxes': bboxes,
        'detection_masks': tf.expand_dims(building_mask, 0),
        'detection_scores': tf.ones([1, 100]),
    }
    return saved_model_output

  module = tf.Module()
  if empty_detection:
    module.__call__ = tf.function(_call_func_empty_detection)
  else:
    module.__call__ = tf.function(_call_func)
  module.__call__.get_concrete_function(
      tf.TensorSpec(shape=[1], dtype=tf.string)
  )
  return module


def _create_test_model_checkpoint(
    test_dir: str, empty_detection: bool = False
) -> str:
  """Creates a dummy model for testing.

  Args:
    test_dir: Temporary test specific directory.
    empty_detection: If true, the model will return an empty detection;
      otherwise it will return detections equivalent to the mmeka model.

  Returns:
    Path to serialized model on local disk.
  """
  module = _create_fake_module(empty_detection)
  model_path = test_dir + '/model/'

  tf2.saved_model.save(module, model_path)
  return model_path


def _create_fake_tile_example() -> tf.train.Example:
  example = tf2.train.Example()
  # Use non multiple of 64 to engage padding behavior.
  image = np.ones([126, 126, 3], dtype=np.uint8)
  image_data = PIL.Image.fromarray(image)

  utils.add_int64_feature(extract_tiles_constants.IMAGE_WIDTH, 126, example)
  utils.add_int64_feature(extract_tiles_constants.IMAGE_HEIGHT, 126, example)
  utils.add_int64_feature(extract_tiles_constants.X_OFFSET, 0, example)
  utils.add_int64_feature(extract_tiles_constants.Y_OFFSET, 0, example)
  utils.add_int64_feature(extract_tiles_constants.MARGIN_SIZE, 0, example)
  utils.add_int64_feature(extract_tiles_constants.TILE_ROW, 0, example)
  utils.add_int64_feature(extract_tiles_constants.TILE_COL, 0, example)
  utils.add_bytes_feature(extract_tiles_constants.IMAGE_FORMAT, b'png', example)
  utils.add_bytes_feature(
      extract_tiles_constants.IMAGE_ENCODED,
      tf.io.encode_png(image_data).numpy(),
      example,
  )
  utils.add_bytes_feature(extract_tiles_constants.CRS, b'epsg:4326', example)
  utils.add_float_list_feature(extract_tiles_constants.AFFINE_TRANSFORM,
                               [1., 0., 0., 0., 1., 0.], example)
  utils.add_bytes_feature(
      extract_tiles_constants.IMAGE_PATH, b'image_path', example
  )
  return example


class DetectBuildingsTest(tf.test.TestCase, parameterized.TestCase):

  def test_building_detection_empty_model_res(self):
    """Tests the Detect Buildings stage outputs zero building instances because the model returns empty detection."""

    example_tiles = [_create_fake_tile_example() for x in range(3)]

    def _check_results(results):
      self.assertEmpty(results)

    with test_pipeline.TestPipeline() as pipeline:
      result = (
          pipeline
          | 'CreateInput' >> beam.Create(example_tiles)
          | 'Inference'
          >> beam.ParDo(
              detect_buildings.DetectBuildingsFn(
                  _create_test_model_checkpoint(
                      self.create_tempdir().full_path, empty_detection=True
                  ),
                  detection_confidence_threshold=0.2,
              )
          )
      )

      util.assert_that(result, _check_results)

  def test_building_detection_empty_high_confidence_threshold(self):
    """Tests the Detect Buildings stage outputs zero building instances because the confidence threshold is too high."""

    example_tiles = [_create_fake_tile_example() for x in range(3)]

    def _check_results(results):
      self.assertEmpty(results)

    with test_pipeline.TestPipeline() as pipeline:
      result = (
          pipeline
          | 'CreateInput' >> beam.Create(example_tiles)
          | 'Inference'
          >> beam.ParDo(
              detect_buildings.DetectBuildingsFn(
                  _create_test_model_checkpoint(
                      self.create_tempdir().full_path, empty_detection=False
                  ),
                  detection_confidence_threshold=1000.0,
              )
          )
      )

      util.assert_that(result, _check_results)

  def test_building_detection(self):
    """Tests the Detect Buildings stage outputs correct building instances."""

    example_tiles = [_create_fake_tile_example() for x in range(3)]

    def _check_results(results):
      self.assertNotEmpty(results)
      for instance in results:
        self.assertCountEqual(
            instance.features.feature.keys(),
            [
                'affine_transform',
                'area',
                'centroid_x',
                'centroid_y',
                'confidence',
                'crs',
                'image',
                'image_path',
                'latitude',
                'longitude',
                'margin_size',
                'mask',
                'mask_wkt',
                'max_x',
                'max_y',
                'min_x',
                'min_y',
                'on_edge',
                'tile_col',
                'tile_pixel_col',
                'tile_pixel_row',
                'tile_row',
            ],
        )
        # Average building confidence must be > .5 or otherwise it would have
        # been classified as background.
        self.assertGreater(
            instance.features.feature[
                detect_buildings_constants.CONFIDENCE
            ].float_list.value[0],
            0.5,
        )
        sparse_mask = tf.deserialize_many_sparse(
            [
                instance.features.feature[
                    detect_buildings_constants.MASK
                ].bytes_list.value
            ],
            tf.int32,
        )
        dense_mask = tf.sparse.to_dense(sparse_mask)
        dense_mask = tf.squeeze(dense_mask)
        self.assertAllEqual(dense_mask.shape, [128, 128])

    with test_pipeline.TestPipeline() as pipeline:
      result = (
          pipeline
          | 'CreateInput' >> beam.Create(example_tiles)
          | 'Inference'
          >> beam.ParDo(
              detect_buildings.DetectBuildingsFn(
                  _create_test_model_checkpoint(
                      self.create_tempdir().full_path, empty_detection=False
                  ),
                  detection_confidence_threshold=0.2,
              )
          )
      )

      util.assert_that(result, _check_results)

  def test_write_csv(self):
    """Tests the Detect Buildings stage outputs correct building instances."""

    example_tiles = [_create_fake_tile_example() for x in range(3)]
    temp_dir = self.create_tempdir().full_path
    csv_path = os.path.join(temp_dir, 'dedup_buildings.csv')
    with test_pipeline.TestPipeline() as pipeline:
      buildings = (
          pipeline
          | 'CreateInput' >> beam.Create(example_tiles)
          | 'Inference'
          >> beam.ParDo(
              detect_buildings.DetectBuildingsFn(
                  _create_test_model_checkpoint(
                      temp_dir, empty_detection=False
                  ),
                  detection_confidence_threshold=0.2,
              )
          )
      )
      detect_buildings.write_csv(buildings, csv_path)

    df = pd.read_csv(f'{csv_path}-00000-of-00001')
    self.assertNotEmpty(df)
    self.assertCountEqual(
        df.columns,
        [
            'area',
            'confidence',
            'image_path',
            'latitude',
            'longitude',
            'wkt',
        ],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='region_0',
          pixel_coords=[(10, 10)],
          stage_0_region=[2.5, 6.5],
          stage_1_region=[],
          stage_2_region=[],
          stage_3_region=[],
      ),
      dict(
          testcase_name='region_1',
          pixel_coords=[(10, 60)],
          stage_0_region=[],
          stage_1_region=[],
          stage_2_region=[2.5, 7],
          stage_3_region=[],
      ),
      dict(
          testcase_name='region_2',
          pixel_coords=[(10, 360)],
          stage_0_region=[2.5, 7.5],
          stage_1_region=[],
          stage_2_region=[],
          stage_3_region=[],
      ),
      dict(
          testcase_name='region_3',
          pixel_coords=[(60, 10)],
          stage_0_region=[],
          stage_1_region=[3, 6.5],
          stage_2_region=[],
          stage_3_region=[],
      ),
      dict(
          testcase_name='region_4',
          pixel_coords=[(60, 60)],
          stage_0_region=[],
          stage_1_region=[],
          stage_2_region=[],
          stage_3_region=[3, 7],
      ),
      dict(
          testcase_name='region_5',
          pixel_coords=[(60, 360)],
          stage_0_region=[],
          stage_1_region=[3, 7.5],
          stage_2_region=[],
          stage_3_region=[],
      ),
      dict(
          testcase_name='region_6',
          pixel_coords=[(460, 10)],
          stage_0_region=[3.5, 6.5],
          stage_1_region=[],
          stage_2_region=[],
          stage_3_region=[],
      ),
      dict(
          testcase_name='region_7',
          pixel_coords=[(460, 60)],
          stage_0_region=[],
          stage_1_region=[],
          stage_2_region=[3.5, 7],
          stage_3_region=[],
      ),
      dict(
          testcase_name='region_8',
          pixel_coords=[(460, 360)],
          stage_0_region=[3.5, 7.5],
          stage_1_region=[],
          stage_2_region=[],
          stage_3_region=[],
      ),
  )
  def test_augment_overlap_single_region(
      self,
      pixel_coords: list[int],
      stage_0_region: list[float],
      stage_1_region: list[float],
      stage_2_region: list[float],
      stage_3_region: list[float],
  ):
    margin_size = 25
    tile_width = 400
    tile_height = 500

    building = _create_building_example(3, 7, 0, 0, tile_height, tile_width,
                                        margin_size, 0.5, pixel_coords)
    building = detect_buildings.augment_overlap_region(building)
    self.assertListEqual(
        _get_float_feature(building, 'dedup_stage_0_region'), stage_0_region)
    self.assertListEqual(
        _get_float_feature(building, 'dedup_stage_1_region'), stage_1_region)
    self.assertListEqual(
        _get_float_feature(building, 'dedup_stage_2_region'), stage_2_region)
    self.assertListEqual(
        _get_float_feature(building, 'dedup_stage_3_region'), stage_3_region)

  def test_augment_overlap_multi_regions(self):
    margin_size = 25
    tile_width = 400
    tile_height = 500

    # This building is on the corner of regions 0, 1, 3, and 4, and touches all
    # of them.
    pixel_coords1 = [(49, 49), (49, 50), (50, 49), (50, 50)]
    building1 = _create_building_example(3, 7, 0, 0, tile_height, tile_width,
                                         margin_size, 0.5, pixel_coords1)
    building1 = detect_buildings.augment_overlap_region(building1)
    self.assertListEqual(
        _get_float_feature(building1, 'dedup_stage_0_region'), [2.5, 6.5])
    self.assertListEqual(
        _get_float_feature(building1, 'dedup_stage_1_region'), [3, 6.5])
    self.assertListEqual(
        _get_float_feature(building1, 'dedup_stage_2_region'), [2.5, 7])
    self.assertListEqual(
        _get_float_feature(building1, 'dedup_stage_3_region'), [3, 7])

    # This building is on the corner of regions 4, 5, 7, and 8, and touches all
    # of them.
    pixel_coords2 = [(449, 349), (449, 350), (450, 349), (450, 350)]
    building2 = _create_building_example(3, 7, 0, 0, tile_height, tile_width,
                                         margin_size, 0.5, pixel_coords2)
    building2 = detect_buildings.augment_overlap_region(building2)
    self.assertListEqual(
        _get_float_feature(building2, 'dedup_stage_0_region'), [3.5, 7.5])
    self.assertListEqual(
        _get_float_feature(building2, 'dedup_stage_1_region'), [3, 7.5])
    self.assertListEqual(
        _get_float_feature(building2, 'dedup_stage_2_region'), [3.5, 7])
    self.assertListEqual(
        _get_float_feature(building2, 'dedup_stage_3_region'), [3, 7])

  def test_non_max_suppression_same_tile(self):
    """Tests that NMS works correctly on buildings from the same tile."""
    buildings = [
        _create_building_example(0, 0, 0, 0, 100, 100, 10, 1.0, [(0, 0), (0, 1),
                                                                 (1, 0)]),
        _create_building_example(0, 0, 0, 0, 100, 100, 10, 0.9, [(0, 0),
                                                                 (0, 1)]),
        _create_building_example(0, 0, 0, 0, 100, 100, 10, 0.8, [(0, 0)]),
        _create_building_example(0, 0, 0, 0, 100, 100, 10, 0.7, [(0, 1), (1, 1),
                                                                 (1, 2)]),
    ]
    deduped_buildings = list(detect_buildings.non_max_suppression(0, buildings))

    # Use confidences to identify buildings, since they are unique.
    deduped_confidences = [
        _get_float_feature(b, detect_buildings_constants.CONFIDENCE)[0]
        for b in deduped_buildings
    ]
    self.assertAllClose([1.0, 0.7], deduped_confidences)

  def test_non_max_suppression_different_tiles(self):
    """Tests that NMS works correctly on buildings from different tiles."""

    # The two tiles share an overlapping corner. The first building is in the
    # upper right corner of the first tile, while the other buildings are in the
    # lower left corner of the second tile.
    buildings = [
        _create_building_example(0, 0, 0, 0, 100, 100, 5, 1.0, [(90, 90),
                                                                (90, 91),
                                                                (91, 90)]),
        _create_building_example(1, 1, 90, 90, 100, 100, 5, 0.9, [(0, 0),
                                                                  (0, 1)]),
        _create_building_example(1, 1, 90, 90, 100, 100, 5, 0.8, [(0, 0)]),
        _create_building_example(1, 1, 90, 90, 100, 100, 5, 0.7,
                                 [(0, 1), (1, 1), (1, 2)]),
    ]
    deduped_buildings = list(detect_buildings.non_max_suppression(0, buildings))

    # Use confidences to identify buildings, since they are unique.
    deduped_confidences = [
        _get_float_feature(b, detect_buildings_constants.CONFIDENCE)[0]
        for b in deduped_buildings
    ]
    self.assertAllClose([1.0, 0.7], deduped_confidences)

  def test_deduplicate_buildings(self):
    buildings = [
        _create_building_example(0, 0, 0, 0, 100, 100, 5, 1.0, [(90, 90),
                                                                (90, 91),
                                                                (91, 90)]),
        _create_building_example(1, 1, 90, 90, 100, 100, 5, 0.9, [(0, 0),
                                                                  (0, 1)]),
        _create_building_example(1, 1, 90, 90, 100, 100, 5, 0.8, [(0, 0)]),
        _create_building_example(1, 1, 90, 90, 100, 100, 5, 0.7,
                                 [(0, 1), (1, 1), (1, 2)]),
    ]
    with test_pipeline.TestPipeline() as pipeline:
      buildings_pcollection = (
          pipeline
          | 'CreateInput' >> beam.Create(buildings))
      deduped_buildings = detect_buildings.deduplicate_buildings(
          buildings_pcollection)

    def _check_results(results):
      deduped_confidences = [
          _get_float_feature(b, detect_buildings_constants.CONFIDENCE)[0]
          for b in results
      ]
      self.assertAllClose([1.0, 0.7], deduped_confidences)

    util.assert_that(deduped_buildings, _check_results)

  def test_recursively_copy_directory(self):
    src_dir = self.create_tempdir()
    dest_dir = self.create_tempdir()
    src_subdir = src_dir.mkdir()
    # Make some files in the src directory.
    for file_prefix in ['a', 'b']:
      src_dir.create_file(file_path=file_prefix)
    for file_prefix in ['c', 'd']:
      src_subdir.create_file(file_path=file_prefix)
    detect_buildings._recursively_copy_directory(src_dir, dest_dir)
    for src, dest in zip(gfile.walk(src_dir), gfile.walk(dest_dir)):
      src_dir_name, src_subdir, src_leaf_files = src
      dest_dir_name, dest_subdir, dest_leaf_files = dest
      self.assertNotEqual(src_dir_name, dest_dir_name)
      self.assertEqual(src_subdir, dest_subdir)
      self.assertEqual(src_leaf_files, dest_leaf_files)

  def test_serialize_sparse_tensor(self):
    sparse_tensor = tf.sparse.SparseTensor(
        indices=[(0, 0), (0, 1), (1, 1)],
        values=np.array([1, 2, 3], dtype=np.int8),
        dense_shape=[2, 2])
    example = tf.train.Example()
    detect_buildings._encode_sparse_tensor(sparse_tensor, example, 'feature')
    decoded = detect_buildings._decode_sparse_tensor(
        example, 'feature', dtype=tf.int8
    )
    self.assertAllEqual(sparse_tensor.indices, decoded.indices)
    self.assertAllEqual(sparse_tensor.values, decoded.values)
    self.assertAllEqual(sparse_tensor.dense_shape, decoded.dense_shape)

  def test_image_padding(self):
    image = np.zeros((600, 500, 3))
    padded_image = detect_buildings._pad_to_square_multiple_of(image, 64)
    self.assertAllEqual(padded_image.shape, (640, 640, 3))

    image = np.zeros((1, 10, 3))
    padded_image = detect_buildings._pad_to_square_multiple_of(image, 64)
    self.assertAllEqual(padded_image.shape, (64, 64, 3))

    image = np.zeros((1, 64, 3))
    padded_image = detect_buildings._pad_to_square_multiple_of(image, 64)
    self.assertAllEqual(padded_image.shape, (64, 64, 3))

    image = np.zeros((1, 65, 3))
    padded_image = detect_buildings._pad_to_square_multiple_of(image, 64)
    self.assertAllEqual(padded_image.shape, (128, 128, 3))

    image = np.zeros((64, 64, 3))
    padded_image = detect_buildings._pad_to_square_multiple_of(image, 64)
    self.assertAllEqual(padded_image.shape, (64, 64, 3))

  def test_recrop_mask(self):
    image = np.ones((1, 2, 3)) * 17
    padded_image = detect_buildings._pad_to_square_multiple_of(image, 64)
    padded_image = np.expand_dims(padded_image, 0)
    recropped = detect_buildings._recrop_mask(
        padded_image, image.shape[0], image.shape[1])
    self.assertAllEqual(image, np.squeeze(recropped, 0))

  def test_get_mask_polygon(self):
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 12:38] = 1
    polygon = detect_buildings._get_mask_polygon(
        mask, 7, 13, 'EPSG:4326', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )
    self.assertEqual(
        list(polygon.exterior.coords),
        [(19.5, 23.5), (19.5, 52.5), (44.5, 52.5), (44.5, 23.5), (19.5, 23.5)],
    )


# `python -m skai.detect_buildings_test` will set __name__ to __main__, but
# `python -m unittest discover -s skai -p detect_buildings_test.py` will set
# __name__ to the filename.
if __name__ == '__main__' or __name__ == 'detect_buildings_test':
  tf.enable_eager_execution()
  tf.test.main()
