# Copyright 2022 Google LLC
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
"""Tests for read_raster."""

import pathlib
import tempfile
from absl.testing import absltest

import affine
import apache_beam as beam
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as test_util
import geopandas as gpd
import rasterio

from skai import buildings
from skai import read_raster

TEST_IMAGE_PATH = 'test_data/blank.tif'

Window = read_raster._Window
WindowGroup = read_raster._WindowGroup


def _create_buildings_file(
    coordinates: list[tuple[float, float]], output_path: str
) -> gpd.GeoDataFrame:
  longitudes = [c[0] for c in coordinates]
  latitudes = [c[1] for c in coordinates]
  gdf = gpd.GeoDataFrame(
      geometry=gpd.points_from_xy(longitudes, latitudes), crs=4326
  )
  buildings.write_buildings_file(gdf, output_path)


class ReadRasterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    current_dir = pathlib.Path(__file__).parent
    self.test_image_path = str(current_dir / TEST_IMAGE_PATH)
    self.raster = rasterio.open(self.test_image_path)

  def test_get_windows(self):
    coordinates = [('a', 178.482925, -16.632893), ('b', 178.482283, -16.632279)]
    utm_crs = rasterio.crs.CRS.from_epsg(32760)  # UTM for above coordinates.
    windows = read_raster._get_windows(self.raster, 64, 0.5, coordinates)
    self.assertLen(windows, 2)
    for window in windows:
      self.assertEqual(window.source_crs, self.raster.crs)
      self.assertEqual(window.target_crs, utm_crs)
      if window.window_id == 'a':
        self.assertEqual(window.column, 113)
        self.assertEqual(window.row, 113)
        self.assertEqual(window.width, 66)
        self.assertEqual(window.height, 68)
        self.assertTrue(
            window.source_transform.almost_equals(
                affine.Affine(0.5, 0.0, 19868611.754, 0.0, -0.5, -1878116.215),
                precision=1e-3,
            )
        )
        self.assertTrue(
            window.target_transform.almost_equals(
                affine.Affine(0.5, 0.0, 658150.285, 0.0, -0.5, 8160485.707),
                precision=1e-3,
            )
        )
      elif window.window_id == 'b':
        self.assertEqual(window.column, -30)
        self.assertEqual(window.row, -29)
        self.assertEqual(window.width, 66)
        self.assertEqual(window.height, 67)
        self.assertTrue(
            window.source_transform.almost_equals(
                affine.Affine(0.5, 0.0, 19868540.288, 0.0, -0.5, -1878044.880),
                precision=1e-3,
            )
        )
        self.assertTrue(
            window.target_transform.almost_equals(
                affine.Affine(0.5, 0.0, 658082.301, 0.0, -0.5, 8160554.155),
                precision=1e-3,
            )
        )

  def test_get_windows_out_of_bounds(self):
    coordinates = [('a', 178.482925, -16.632893), ('b', 160, -10)]
    windows = read_raster._get_windows(self.raster, 64, 0.5, coordinates)
    self.assertLen(windows, 1)
    self.assertEqual(windows[0].window_id, 'a')

  def test_group_windows(self):
    windows = [Window('w1', 0, 0, 256, 256),
               Window('w2', 1, 1, 256, 256),
               Window('w3', 1000, 0, 256, 256)]
    groups = list(read_raster._group_windows(windows))
    self.assertLen(groups, 2)
    self.assertLen(windows, sum(len(g.members) for g in groups))

  def test_read_window_group(self):
    windows = [
        read_raster._compute_window(self.raster, 'w1', 0, 0, 64, 0.5),
        read_raster._compute_window(self.raster, 'w2', 0.0002, 0.0002, 64, 0.5),
        read_raster._compute_window(self.raster, 'w3', 20, 20, 64, 0.5),
    ]
    group1 = WindowGroup(windows[0])
    group1.add_window(windows[1])
    group2 = WindowGroup(windows[2])
    with test_pipeline.TestPipeline() as pipeline:
      patches = (
          pipeline
          | beam.Create(
              [(self.test_image_path, group1), (self.test_image_path, group2)]
          )
          | beam.ParDo(read_raster.ReadRasterWindowGroupFn(64, {}))
      )

      expected_image_path = self.test_image_path
      def _check_output_patches(patches):
        assert len(patches) == 3, len(patches)
        assert patches[0][0] == 'w1', patches[0][0]
        assert patches[0][1][0] == expected_image_path
        assert patches[0][1][1].shape == (64, 64, 3), patches[0][1][1].shape
        assert patches[1][0] == 'w2', patches[1][0]
        assert patches[1][1][0] == expected_image_path
        assert patches[1][1][1].shape == (64, 64, 3), patches[1][1][1]
        assert patches[2][0] == 'w3', patches[2][0]
        assert patches[2][1][0] == expected_image_path
        assert patches[2][1][1].shape == (64, 64, 3), patches[2][1][1]

      test_util.assert_that(patches, _check_output_patches)

  def test_buildings_to_groups(self):
    coordinates = [
        (178.482925, -16.632893),
        (178.482283, -16.632279),
        (178.482284, -16.632279)]

    with tempfile.NamedTemporaryFile(dir=absltest.TEST_TMPDIR.value) as f:
      buildings_path = f.name
      _create_buildings_file(coordinates, buildings_path)
      groups = list(read_raster._buildings_to_groups(
          self.test_image_path, buildings_path, 32, 0.5, {}))
      self.assertLen(groups, 2)

  def test_get_raster_bounds(self):
    bounds = read_raster.get_raster_bounds(self.test_image_path, {})
    self.assertSequenceAlmostEqual(
        bounds.bounds, [178.4822663, -16.6334717, 178.4836138, -16.6322581])


if __name__ == '__main__':
  absltest.main()
