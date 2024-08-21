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

"""Tests for extract tiles pipeline.
"""

import pathlib

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that
import numpy as np
import rasterio
import shapely.geometry
from skai import extract_tiles
import tensorflow as tf

Tile = extract_tiles.Tile

TEST_IMAGE_PATH = 'test_data/blank.tif'


def _deserialize_image(serialized_image: bytes) -> np.ndarray:
  return tf.io.decode_image(serialized_image).numpy()


def _check_examples(expected_tiles: list[Tile]):
  """Validates examples generated from beam pipeline.

  Args:
    expected_tiles: List of tiles that examples should cover.

  Returns:
    Function for validating examples based on expected extents.
  """

  def _check_actual_examples(actual_examples):
    actual_tiles = set()
    for example in actual_examples:
      feature_names = set(example.features.feature.keys())
      assert feature_names == set([
          'image/width', 'image/height', 'image/format', 'image/encoded', 'x',
          'y', 'tile_column', 'tile_row', 'margin_size', 'crs',
          'affine_transform', 'image_path'
      ])

      assert (example.features.feature['image/format'].bytes_list.value[0] ==
              b'png')
      x = example.features.feature['x'].int64_list.value[0]
      y = example.features.feature['y'].int64_list.value[0]
      width = example.features.feature['image/width'].int64_list.value[0]
      height = example.features.feature['image/height'].int64_list.value[0]
      tile_column = example.features.feature['tile_column'].int64_list.value[0]
      tile_row = example.features.feature['tile_row'].int64_list.value[0]
      margin_size = example.features.feature['margin_size'].int64_list.value[0]
      image_path = (
          example.features.feature['image_path'].bytes_list.value[0].decode()
      )

      crs = example.features.feature['crs'].bytes_list.value[0].decode()
      assert crs == 'EPSG:3857', crs

      transform = tuple(
          example.features.feature['affine_transform'].float_list.value[:])
      assert np.allclose(
          transform,
          (0.5, 0.0, 19868555.0, 0.0, -0.5, -1878059.3724)), transform

      # Check that the encoded image has the same dimensions as the image/width
      # and image/height features.
      image = _deserialize_image(
          example.features.feature['image/encoded'].bytes_list.value[0])
      assert image.shape[0] == height
      assert image.shape[1] == width
      assert image.shape[2] == 3

      actual_tiles.add(
          Tile(
              image_path,
              x,
              y,
              width,
              height,
              margin_size,
              tile_column,
              tile_row,
          )
      )

    assert set(expected_tiles) == actual_tiles

  return _check_actual_examples


class ExtractTilesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    current_dir = pathlib.Path(__file__).parent
    self.test_image_path = str(current_dir / TEST_IMAGE_PATH)
    self.image = rasterio.open(self.test_image_path)

  def testGetTiles(self):
    """Tests that correct window extents are generated for a GeoTIFF."""
    tiles = set(
        extract_tiles.get_tiles(
            self.test_image_path,
            x_min=0,
            y_min=0,
            x_max=300,
            y_max=282,
            tile_size=100,
            margin=10,
        )
    )

    self.assertEqual(tiles, set([
        Tile(self.test_image_path, 0, 0, 120, 120, 10, 0, 0),
        Tile(self.test_image_path, 100, 0, 120, 120, 10, 1, 0),
        Tile(self.test_image_path, 200, 0, 120, 120, 10, 2, 0),
        Tile(self.test_image_path, 0, 100, 120, 120, 10, 0, 1),
        Tile(self.test_image_path, 100, 100, 120, 120, 10, 1, 1),
        Tile(self.test_image_path, 200, 100, 120, 120, 10, 2, 1),
        Tile(self.test_image_path, 0, 200, 120, 120, 10, 0, 2),
        Tile(self.test_image_path, 100, 200, 120, 120, 10, 1, 2),
        Tile(self.test_image_path, 200, 200, 120, 120, 10, 2, 2)
    ]))

  def testGetTilesForAOI(self):
    """Tests that correct window extents are generated for a subrectangle."""
    aoi = shapely.geometry.Polygon([
        (178.48236869631341506, -16.63234307247736155),
        (178.48236359544131346, -16.63313880852423665),
        (178.48353934645930963, -16.6331413589602839),
        (178.48351639253488088, -16.63235072378550328)])

    tiles = set(extract_tiles.get_tiles_for_aoi(
        self.test_image_path, aoi, tile_size=100, margin=10, gdal_env={}))

    self.assertEqual(tiles, set([
        Tile(self.test_image_path, 21, 19, 120, 120, 10, 0, 0),
        Tile(self.test_image_path, 21, 119, 120, 120, 10, 0, 1),
        Tile(self.test_image_path, 121, 19, 120, 120, 10, 1, 0),
        Tile(self.test_image_path, 121, 119, 120, 120, 10, 1, 1),
        Tile(self.test_image_path, 221, 19, 120, 120, 10, 2, 0),
        Tile(self.test_image_path, 221, 119, 120, 120, 10, 2, 1)
    ]))

  def testExtractTilesAsExamplesFn(self):
    """Tests that beam pipeline generates correct tensorflow examples."""

    # These tiles form a 3x3 grid that covers the test image.
    tiles = [
        Tile(self.test_image_path, 0, 0, 120, 120, 10, 0, 0),
        Tile(self.test_image_path, 100, 0, 120, 120, 10, 1, 0),
        Tile(self.test_image_path, 200, 0, 120, 120, 10, 2, 0),
        Tile(self.test_image_path, 0, 100, 120, 120, 10, 0, 1),
        Tile(self.test_image_path, 100, 100, 120, 120, 10, 1, 1),
        Tile(self.test_image_path, 200, 100, 120, 120, 10, 2, 1),
        Tile(self.test_image_path, 0, 200, 120, 120, 10, 0, 2),
        Tile(self.test_image_path, 100, 200, 120, 120, 10, 1, 2),
        Tile(self.test_image_path, 200, 200, 120, 120, 10, 2, 2)
    ]

    with test_pipeline.TestPipeline() as pipeline:
      result = (
          pipeline
          | beam.Create(tiles)
          | beam.ParDo(extract_tiles.ExtractTilesAsExamplesFn({}))
      )

      assert_that(result, _check_examples(tiles))


if __name__ == '__main__':
  absltest.main()
