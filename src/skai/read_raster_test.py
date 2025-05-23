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

import os
import pathlib
import tempfile
from absl.testing import absltest
import affine
import apache_beam as beam
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as test_util
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from skai import buildings
from skai import read_raster

TEST_IMAGE_PATH = 'test_data/blank.tif'

_RasterBin = read_raster._RasterBin
_RasterPoint = read_raster._RasterPoint
_Window = read_raster._Window
_WindowGroup = read_raster._WindowGroup


def _create_test_image_tiff_file_with_position_size(
    x: float,
    y: float,
    width: int,
    height: int,
    num_channels: int,
    crs: str,
    colorinterps: list[ColorInterp],
    tags: list[dict[str, str]],
    resolution: float = 0.1,
):
  image = np.random.randint(
      0, 256, (height, width, num_channels), dtype=np.uint8
  )

  profile = {
      'driver': 'GTiff',
      'height': height,
      'width': width,
      'count': num_channels,
      'dtype': 'uint8',
      'crs': crs,
      'transform': rasterio.transform.from_origin(
          x, y, resolution, resolution
      ),
  }
  _, image_path = tempfile.mkstemp(
      dir=absltest.TEST_TMPDIR.value, suffix='.tiff'
  )

  with rasterio.open(image_path, 'w', **profile) as dst:
    for band, band_tags in enumerate(tags):
      dst.update_tags(band + 1, **band_tags)

    for i in range(num_channels):
      dst.write(image[..., i], i + 1)
    dst.colorinterp = colorinterps
  return image_path


def _create_test_image_tiff_file(colorinterps: list[ColorInterp]):
  return _create_test_image_tiff_file_with_position_size(
      0, 0, 100, 100, len(colorinterps), '+proj=latlong', colorinterps, {}
  )


def _create_test_image_at_path(
    path: str,
    x: float,
    y: float,
    width: int,
    height: int,
    num_channels: int,
    crs: str,
    resolution: float,
):
  image = np.random.randint(
      0, 256, (height, width, num_channels), dtype=np.uint8
  )

  profile = {
      'driver': 'GTiff',
      'height': height,
      'width': width,
      'count': num_channels,
      'dtype': 'uint8',
      'crs': crs,
      'transform': rasterio.transform.from_origin(
          x, y, resolution, resolution
      ),
  }
  with rasterio.open(path, 'w', **profile) as dst:
    for i in range(num_channels):
      dst.write(image[..., i], i + 1)


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
    self.bad_alpha_test_image_path = str(
        current_dir / 'test_data/bad_alpha_channel.tif'
    )

  def test_generate_raster_points(self):
    coordinates = [
        (178.482925, -16.632893),
        (178.482283, -16.632279),
        (178.482925, -14),
        (180, -16.632893),
        (180, -14),
    ]

    with tempfile.NamedTemporaryFile(dir=absltest.TEST_TMPDIR.value) as f:
      buildings_path = f.name
      _create_buildings_file(coordinates, buildings_path)
      with test_pipeline.TestPipeline() as pipeline:
        result = (
            pipeline
            | beam.Create([self.test_image_path])
            | beam.FlatMap(
                read_raster._generate_raster_points, buildings_path
            )
        )
        def _check_raster_points(raster_points):
          self.assertLen(raster_points, 2)
          self.assertCountEqual(raster_points, [
              _RasterPoint(self.test_image_path, 178.482925, -16.632893),
              _RasterPoint(self.test_image_path, 178.482283, -16.632279),
          ])
        test_util.assert_that(result, _check_raster_points)

  def test_get_windows(self):
    raster_points = [
        _RasterPoint(self.test_image_path, 178.482925, -16.632893),
        _RasterPoint(self.test_image_path, 178.482283, -16.632279),
    ]

    with test_pipeline.TestPipeline() as pipeline:
      result = (
          pipeline
          | beam.Create(raster_points)
          | beam.ParDo(read_raster.MakeWindow(64, 0.5, {}))
      )

      def _check_windows(windows):
        self.assertLen(windows, 2)
        for key, window in windows:
          self.assertEqual(key, _RasterBin(self.test_image_path, 17848, -1664))
          self.assertEqual(window.source_crs, rasterio.crs.CRS.from_epsg(3857))
          self.assertEqual(window.target_crs, rasterio.crs.CRS.from_epsg(32760))
          if window.window_id == 'A17B32432A1085C1':
            self.assertEqual(window.column, 103)
            self.assertEqual(window.row, 103)
            self.assertEqual(window.width, 87)
            self.assertEqual(window.height, 88)
            self.assertTrue(
                window.source_transform.almost_equals(
                    affine.Affine(
                        0.5,
                        0.0,
                        19868606.5762,
                        0.0,
                        -0.5,
                        -1878110.9265,
                    ),
                    precision=1e-3,
                )
            )
            self.assertTrue(
                window.target_transform.almost_equals(
                    affine.Affine(
                        0.5, 0.0, 658145.2851, 0.0, -0.5, 8160490.7810
                    ),
                    precision=1e-3,
                )
            )
          elif window.window_id == '777B3243E80E85C1':
            self.assertEqual(window.column, -40)
            self.assertEqual(window.row, -40)
            self.assertEqual(window.width, 87)
            self.assertEqual(window.height, 89)
            self.assertTrue(
                window.source_transform.almost_equals(
                    affine.Affine(
                        0.5, 0.0, 19868535.1091, 0.0, -0.5, -1878039.5919
                    ),
                    precision=1e-3,
                )
            )
            self.assertTrue(
                window.target_transform.almost_equals(
                    affine.Affine(
                        0.5, 0.0, 658077.3008, 0.0, -0.5, 8160559.2287
                    ),
                    precision=1e-3,
                )
            )
          else:
            raise AssertionError(f'Unexpected window id "{window.window_id}"')

      test_util.assert_that(result, _check_windows)

  def test_group_windows(self):
    windows = [_Window('w1', 0, 0, 256, 256),
               _Window('w2', 1, 1, 256, 256),
               _Window('w3', 1000, 0, 256, 256)]
    with test_pipeline.TestPipeline() as pipeline:
      result = (
          pipeline
          | beam.Create([(_RasterBin(self.test_image_path, 0, 0), windows)])
          | beam.FlatMap(read_raster._group_windows)
      )

      def _check_groups(groups):
        self.assertLen(groups, 2)
        self.assertEqual(self.test_image_path, groups[0][0])
        self.assertEqual(self.test_image_path, groups[1][0])
        window_ids0 = [w.window_id for w in groups[0][1].members]
        window_ids1 = [w.window_id for w in groups[1][1].members]
        self.assertCountEqual(
            [window_ids0, window_ids1], [['w1', 'w2'], ['w3']]
        )

      test_util.assert_that(result, _check_groups)

  def test_read_window_group(self):
    raster = rasterio.open(self.test_image_path)
    windows = [
        read_raster._compute_window(raster, 'w1', 0, 0, 64, 0.5),
        read_raster._compute_window(raster, 'w2', 0.0002, 0.0002, 64, 0.5),
        read_raster._compute_window(raster, 'w3', 20, 20, 64, 0.5),
    ]
    group1 = _WindowGroup(windows[0])
    group1.add_window(windows[1])
    group2 = _WindowGroup(windows[2])
    with test_pipeline.TestPipeline() as pipeline:
      result = (
          pipeline
          | beam.Create(
              [(self.test_image_path, group1), (self.test_image_path, group2)]
          )
          | beam.ParDo(read_raster.ReadRasterWindowGroupFn([], {}))
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

      test_util.assert_that(result, _check_output_patches)

  def test_get_raster_bounds(self):
    bounds = read_raster.get_raster_bounds(self.test_image_path, {})
    self.assertSequenceAlmostEqual(
        bounds.bounds, [178.4822663, -16.6334717, 178.4836138, -16.6322581])

  def test_get_rgb_indices_grb_image(self):
    image_path = _create_test_image_tiff_file(
        [ColorInterp.green, ColorInterp.red, ColorInterp.blue]
    )
    dataset = rasterio.open(image_path)
    indices = read_raster.get_rgb_indices(dataset)
    self.assertSequenceEqual(indices, [2, 1, 3])

  def test_get_rgb_indices_bgr_image(self):
    image_path = _create_test_image_tiff_file(
        [ColorInterp.blue, ColorInterp.green, ColorInterp.red]
    )
    dataset = rasterio.open(image_path)
    indices = read_raster.get_rgb_indices(dataset)
    self.assertSequenceEqual(indices, [3, 2, 1])

  def test_get_rgb_indices_argb_image(self):
    image_path = _create_test_image_tiff_file([
        ColorInterp.alpha,
        ColorInterp.red,
        ColorInterp.green,
        ColorInterp.blue,
    ])
    dataset = rasterio.open(image_path)
    indices = read_raster.get_rgb_indices(dataset)
    self.assertSequenceEqual(indices, [2, 3, 4])

  def test_get_rgb_indices_missing_red(self):
    image_path = _create_test_image_tiff_file([
        ColorInterp.green,
        ColorInterp.blue,
    ])
    dataset = rasterio.open(image_path)
    with self.assertRaisesRegex(
        ValueError, 'Raster does not have a red channel.'
    ):
      read_raster.get_rgb_indices(dataset)

  def test_get_rgb_indices_missing_green(self):
    image_path = _create_test_image_tiff_file([
        ColorInterp.red,
        ColorInterp.blue,
    ])
    dataset = rasterio.open(image_path)
    with self.assertRaisesRegex(
        ValueError, 'Raster does not have a green channel.'
    ):
      read_raster.get_rgb_indices(dataset)

  def test_get_rgb_indices_missing_blue(self):
    image_path = _create_test_image_tiff_file([
        ColorInterp.red,
        ColorInterp.green,
    ])
    dataset = rasterio.open(image_path)
    with self.assertRaisesRegex(
        ValueError, 'Raster does not have a blue channel.'
    ):
      read_raster.get_rgb_indices(dataset)

  def test_get_rgb_indices_band_name_tags(self):
    image_path = _create_test_image_tiff_file_with_position_size(
        x=10,
        y=20,
        width=100,
        height=100,
        num_channels=4,
        crs='EPSG:4326',
        colorinterps=[
            ColorInterp.undefined,
            ColorInterp.undefined,
            ColorInterp.undefined,
            ColorInterp.undefined,
        ],
        tags=[
            {'BandName': 'Alpha'},
            {'BandName': 'Red'},
            {'BandName': 'Green'},
            {'BandName': 'Blue'},
        ],
    )
    dataset = rasterio.open(image_path)
    self.assertSequenceEqual(read_raster.get_rgb_indices(dataset), (2, 3, 4))

  def test_get_rgb_indices_bad_alpha_channel(self):
    raster = rasterio.open(self.bad_alpha_test_image_path)
    self.assertSequenceEqual(read_raster.get_rgb_indices(raster), (1, 2, 3))

  def test_convert_image_to_uint8(self):
    band = np.diag([4095, 2047, 1023, 511]).astype(np.uint16)
    image = np.stack([band, band, band], axis=2)
    self.assertSequenceEqual(image.shape, (4, 4, 3))
    converted = read_raster._convert_image_to_uint8(image, 12)
    self.assertEqual(converted.dtype, np.uint8)
    self.assertSequenceEqual(converted.shape, [4, 4, 3])
    np.testing.assert_equal(converted[0, 0, :], [255, 255, 255])
    np.testing.assert_equal(converted[1, 1, :], [127, 127, 127])
    np.testing.assert_equal(converted[2, 2, :], [63, 63, 63])
    np.testing.assert_equal(converted[3, 3, :], [31, 31, 31])

  def test_build_mosaic_vrt(self):
    image1_path = _create_test_image_tiff_file_with_position_size(
        x=10,
        y=20,
        width=100,
        height=100,
        num_channels=3,
        crs='EPSG:26910',
        colorinterps=[ColorInterp.red, ColorInterp.green, ColorInterp.blue],
        tags={},
    )
    image2_path = _create_test_image_tiff_file_with_position_size(
        x=20,
        y=20,
        width=100,
        height=100,
        num_channels=3,
        crs='EPSG:26910',
        colorinterps=[ColorInterp.red, ColorInterp.green, ColorInterp.blue],
        tags={},
    )
    vrt_path = read_raster._build_mosaic_vrt(
        [image1_path, image2_path],
        pathlib.Path(absltest.TEST_TMPDIR.value) / 'image',
        0.5,
    )
    vrt_raster = rasterio.open(vrt_path)
    vrt_image = vrt_raster.read()
    self.assertEqual(3, vrt_raster.count)
    self.assertEqual((0.5, 0.5), vrt_raster.res)
    # resolution lowered by 5x, so pixel sizes also lowered by 5x.
    self.assertEqual((3, 20, 40), vrt_image.shape)

  def test_build_individual_vrts(self):
    image1_path = _create_test_image_tiff_file_with_position_size(
        x=581370,
        y=4141960,
        width=200,
        height=100,
        num_channels=3,
        crs='EPSG:32610',
        colorinterps=[ColorInterp.red, ColorInterp.green, ColorInterp.blue],
        tags={},
    )
    image2_path = _create_test_image_tiff_file_with_position_size(
        x=581375,
        y=4141970,
        width=200,
        height=100,
        num_channels=3,
        crs='EPSG:32610',
        colorinterps=[ColorInterp.red, ColorInterp.green, ColorInterp.blue],
        tags={},
    )
    vrt_paths = read_raster._build_warped_vrts(
        [image1_path, image2_path],
        pathlib.Path(absltest.TEST_TMPDIR.value) / 'image',
        0.5,
        {},
    )
    self.assertLen(vrt_paths, 2)
    for vrt_path in vrt_paths:
      vrt_raster = rasterio.open(vrt_path)
      vrt_image = vrt_raster.read()
      self.assertEqual(3, vrt_raster.count)
      self.assertEqual((0.5, 0.5), vrt_raster.res)
      # Image size is divided by 5 because resolution lowered from 0.1 to 0.5.
      self.assertEqual((3, 41, 51), vrt_image.shape)
      # All images should have the same bounds. The bounds is the union of the
      # bounds of the individual images.
      np.testing.assert_allclose(
          (581370, 4141949.5, 581395.5, 4141970), tuple(vrt_raster.bounds)
      )

  def test_build_individual_vrts_diff_crs(self):
    image1_path = _create_test_image_tiff_file_with_position_size(
        x=581370,
        y=4141960,
        width=200,
        height=100,
        num_channels=3,
        crs='EPSG:32610',
        colorinterps=[ColorInterp.red, ColorInterp.green, ColorInterp.blue],
        tags={},
    )
    image2_path = _create_test_image_tiff_file_with_position_size(
        x=-122.08034382,
        y=37.42096571,
        width=200,
        height=100,
        num_channels=3,
        crs='EPSG:4326',
        colorinterps=[ColorInterp.red, ColorInterp.green, ColorInterp.blue],
        tags={},
        resolution=0.0000010198723355,  # About 0.1m
    )
    vrt_paths = read_raster._build_warped_vrts(
        [image1_path, image2_path],
        pathlib.Path(absltest.TEST_TMPDIR.value) / 'image',
        0.5,
        {},
    )
    self.assertLen(vrt_paths, 2)
    for vrt_path in vrt_paths:
      vrt_raster = rasterio.open(vrt_path)
      vrt_image = vrt_raster.read()
      self.assertEqual(3, vrt_raster.count)
      self.assertEqual((0.5, 0.5), vrt_raster.res)
      # Image size is divided by 5 because resolution lowered from 0.1 to 0.5.
      self.assertEqual((3, 41, 47), vrt_image.shape)
      # All images should have the same bounds. The bounds is the union of the
      # bounds of the individual images.
      np.testing.assert_allclose(
          (581370, 4141949.675, 581393.5, 4141970.175), tuple(vrt_raster.bounds)
      )

  def test_prepare_building_detection_input_images(self):
    images_dir = self.create_tempdir()
    _create_test_image_at_path(
        path=os.path.join(images_dir, 'image1.tif'),
        x=581370,
        y=4141960,
        width=200,
        height=100,
        num_channels=3,
        crs='EPSG:26910',
        resolution=1.0,
    )
    _create_test_image_at_path(
        path=os.path.join(images_dir, 'image2.tif'),
        x=581380,
        y=4141970,
        width=200,
        height=100,
        num_channels=3,
        crs='EPSG:26910',
        resolution=0.5,
    )
    _create_test_image_at_path(
        path=os.path.join(images_dir, 'standalone_image.tif'),
        x=-122.08034382,
        y=37.42096571,
        width=500,
        height=500,
        num_channels=3,
        crs='EPSG:4326',
        resolution=0.000006,
    )
    image_paths = [
        os.path.join(images_dir, 'image*.tif'),
        os.path.join(images_dir, 'standalone_image.tif'),
    ]
    vrt_dir = self.create_tempdir()
    vrt_paths = read_raster.prepare_building_detection_input_images(
        image_paths, vrt_dir, {}
    )
    self.assertLen(vrt_paths, 2)
    vrt1 = rasterio.open(vrt_paths[0])
    self.assertEqual(vrt1.crs.to_epsg(), 32610)
    self.assertEqual(vrt1.res, (0.5, 0.5))
    vrt2 = rasterio.open(vrt_paths[1])
    self.assertEqual(vrt2.crs.to_epsg(), 32610)
    self.assertEqual(vrt2.res, (0.5, 0.5))
    self.assertEqual(vrt1.width, vrt2.width)
    self.assertEqual(vrt1.height, vrt2.height)
    self.assertEqual(vrt1.transform, vrt2.transform)

    self.assertCountEqual(
        os.listdir(vrt_dir),
        [
            'mosaics',
            'input-00000-of-00002.vrt',
            'input-00001-of-00002.vrt',
        ],
    )
    self.assertCountEqual(
        os.listdir(os.path.join(vrt_dir, 'mosaics')),
        [
            'mosaic-00000.vrt',
        ],
    )


if __name__ == '__main__':
  absltest.main()
