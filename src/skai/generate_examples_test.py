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

import glob
import os
import pathlib
import tempfile
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as test_util
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from skai import buildings
from skai import generate_examples
from skai import read_raster
from skai import utils
import tensorflow as tf


TEST_IMAGE_PATH = 'test_data/blank.tif'
TEST_CONFIG_PATH = 'test_data/config.json'
TEST_MISSING_DATASET_NAME_CONFIG_PATH = (
    'test_data/missing_dataset_name_config.json'
)
TEST_MISSING_OUPTUT_DIR_CONFIG_PATH = 'test_data/missing_output_dir_config.json'
TEST_CONFIG_WITH_IMAGE_INFO_PATH = 'test_data/config_with_image_info.json'

RasterInfo = read_raster.RasterInfo


def _get_test_file_path(file_name: str) -> str:
  return str(pathlib.Path(__file__).parent / file_name)


def _get_before_image_id(example):
  return example.features.feature['pre_image_id'].bytes_list.value[0].decode()


def _deserialize_image(serialized_image: bytes) -> np.ndarray:
  return tf.io.decode_image(serialized_image).numpy()


def _unordered_all_close(list1: list[Any], list2: list[Any]) -> bool:
  """Return that two lists of coordinates are close to each other."""
  if len(list1) != len(list2):
    return False

  sorted_list1 = sorted(list1)
  sorted_list2 = sorted(list2)
  return np.allclose(sorted_list1, sorted_list2)


def _create_buildings_file(
    coordinates: list[tuple[float, float]], output_path: str
) -> None:
  longitudes = [c[0] for c in coordinates]
  latitudes = [c[1] for c in coordinates]
  gdf = gpd.GeoDataFrame(
      {
          'area_in_meters': [0.0] * len(coordinates),
      },
      geometry=gpd.points_from_xy(longitudes, latitudes),
      crs=4326,
  )
  buildings.write_buildings_file(gdf, output_path)


def _create_labeled_buildings_file(
    coordinates: list[tuple[float, float, float, str]], output_path: str
) -> None:
  longitudes = [c[0] for c in coordinates]
  latitudes = [c[1] for c in coordinates]
  labels = [c[2] for c in coordinates]
  string_labels = [c[3] for c in coordinates]
  area_in_meters = [0.0] * len(coordinates)
  gdf = gpd.GeoDataFrame(
      {
          'label': labels,
          'string_label': string_labels,
          'area_in_meters': area_in_meters,
      },
      geometry=gpd.points_from_xy(longitudes, latitudes),
      crs=4326,
  )
  buildings.write_buildings_file(gdf, output_path)


def _create_buildings_file_with_plus_code(
    coordinates: list[tuple[float, float]], output_path: str
) -> None:
  longitudes = [c[0] for c in coordinates]
  latitudes = [c[1] for c in coordinates]
  gdf = gpd.GeoDataFrame(
      {
          'area_in_meters': [0.0] * len(coordinates),
          'full_plus_code': ['abc'] * len(coordinates),
      },
      geometry=gpd.points_from_xy(longitudes, latitudes),
      crs=4326,
  )
  buildings.write_buildings_file(gdf, output_path)


def _create_labeled_geojson(
    coordinates: list[tuple[float, float, str]], label_property: str
) -> None:
  longitudes = [c[0] for c in coordinates]
  latitudes = [c[1] for c in coordinates]
  string_labels = [c[2] for c in coordinates]
  gdf = gpd.GeoDataFrame(
      {
          label_property: string_labels,
      },
      geometry=gpd.points_from_xy(longitudes, latitudes),
      crs=4326,
  )
  _, path = tempfile.mkstemp(suffix='.geojson', dir=absltest.TEST_TMPDIR.value)
  gdf.to_file(path, driver='GeoJSON')
  return path


def _create_rectangle_at_point(
    point: tuple[float, float],
) -> shapely.Polygon:
  half_length = 0.0000001  # In degrees, approximately equal to 1 meter.
  return shapely.Polygon([
      (point[0] - half_length, point[1] - half_length),
      (point[0] + half_length, point[1] - half_length),
      (point[0] + half_length, point[1] + half_length),
      (point[0] - half_length, point[1] + half_length),
  ])


def _create_buildings_file_with_footprints(
    coordinates: list[tuple[float, float]],
    output_path: str,
) -> None:
  longitudes = [c[0] for c in coordinates]
  latitudes = [c[1] for c in coordinates]
  footprints = [
      _create_rectangle_at_point((x, y)) for x, y in zip(longitudes, latitudes)
  ]
  gdf = gpd.GeoDataFrame(
      {
          'area_in_meters': [0.0] * len(coordinates),
      },
      geometry=footprints,
      crs=4326,
  )
  buildings.write_buildings_file(gdf, output_path)


def _create_example(
    example_id: str,
    int64_id: int = 0) -> tf.train.Example:
  example = tf.train.Example()
  utils.add_bytes_feature('example_id', example_id.encode(), example)
  utils.add_int64_feature('int64_id', int64_id, example)
  utils.add_float_list_feature('coordinates', (90, 90), example)
  utils.add_bytes_feature(
      'encoded_coordinates', b'encoded_coordinates', example
  )
  utils.add_bytes_feature('pre_image_id', b'pre_image_id', example)
  utils.add_bytes_feature('post_image_id', b'post_image_id', example)
  utils.add_bytes_feature('plus_code', b'plus_code', example)
  utils.add_float_feature('label', 0.0, example)
  utils.add_float_feature('post_footprint_x_shift_meters', 0.0, example)
  utils.add_float_feature('post_footprint_y_shift_meters', 0.0, example)
  utils.add_float_feature('post_footprint_match_score', 0.0, example)
  utils.add_bytes_feature('building_image_id', b'building_image_id', example)
  utils.add_bytes_feature('pre_image_png', b'', example)
  utils.add_bytes_feature('post_image_png', b'', example)
  utils.add_bytes_feature('pre_image_png_large', b'', example)
  utils.add_bytes_feature('post_image_png_large', b'', example)
  return example


def _check_examples(
    before_image_id: str,
    after_image_id: str,
    small_patch_size: int,
    large_patch_size: int,
    expected_coordinates: list[tuple[float, float, float]],
    expected_string_labels: list[str],
    expected_plus_codes: list[str],
    expect_blank_before: bool,
    expect_large_patch: bool,
    expect_footprint: bool,
):
  """Validates examples generated from beam pipeline.

  Args:
    before_image_id: Expected before image id.
    after_image_id: Expected after image id.
    small_patch_size: The expected size of small patches.
    large_patch_size: The expected size of large patches.
    expected_coordinates: List of coordinates that examples should have.
    expected_string_labels: List of string labels that examples should have.
    expected_plus_codes: List of plus codes that examples should have.
    expect_blank_before: If true, the before image should be all zeros.
    expect_large_patch: If true, the examples should contain large patches.
    expect_footprint: If true, the examples should contain footprints.

  Returns:
    Function for validating examples.
  """

  def _check_examples_internal(actual_examples):
    actual_coordinates = set()
    actual_string_labels = []
    actual_plus_codes = []
    expected_small_shape = (small_patch_size, small_patch_size, 3)
    expected_large_shape = (large_patch_size, large_patch_size, 3)

    expected_feature_names = set([
        'pre_image_png',
        'pre_image_id',
        'post_image_png',
        'post_image_id',
        'coordinates',
        'encoded_coordinates',
        'label',
        'example_id',
        'int64_id',
        'plus_code',
        'string_label',
        'area_in_meters',
        'post_footprint_x_shift_meters',
        'post_footprint_y_shift_meters',
        'post_footprint_match_score',
        'building_image_id',
    ])
    if expect_large_patch:
      expected_feature_names.update(
          ['pre_image_png_large', 'post_image_png_large']
      )
    if expect_footprint:
      expected_feature_names.update(['footprint_wkb'])
      expected_feature_names.update(['post_footprint_wkb'])

    for example in actual_examples:
      feature_names = set(example.features.feature.keys())

      if feature_names != expected_feature_names:
        extra_features = feature_names - expected_feature_names
        missing_features = expected_feature_names - feature_names
        raise ValueError(
            'Feature set does not match.\nExpected:' +
            f' {" ".join(expected_feature_names)}.\n' +
            f'Got: {" ".join(feature_names)}.\n' +
            f'Extra features: {" ".join(extra_features)}.\n' +
            f'Missing features: {" ".join(missing_features)}'
        )

      actual_before_id = (
          example.features.feature['pre_image_id'].bytes_list.value[0].decode()
      )
      assert actual_before_id == before_image_id, actual_before_id
      actual_after_id = example.features.feature[
          'post_image_id'].bytes_list.value[0].decode()
      assert actual_after_id == after_image_id, actual_after_id

      longitude, latitude = (
          example.features.feature['coordinates'].float_list.value)
      label = example.features.feature['label'].float_list.value[0]

      before_image = _deserialize_image(
          example.features.feature['pre_image_png'].bytes_list.value[0])
      assert before_image.shape == expected_small_shape, (
          f'Expected before image shape = {expected_small_shape}, '
          f'actual = {before_image.shape}')

      if expect_blank_before:
        assert not before_image.any(), 'Expected blank before image.'
      else:
        assert before_image.any(), 'Expected non-blank before image.'

      after_image = _deserialize_image(
          example.features.feature['post_image_png'].bytes_list.value[0])
      assert after_image.shape == expected_small_shape, (
          f'Expected after image shape = {expected_small_shape}, '
          f'actual = {after_image.shape}')

      if expect_large_patch:
        large_before_image = _deserialize_image(
            example.features.feature['pre_image_png_large'].bytes_list.value[0])
        assert large_before_image.shape == expected_large_shape, (
            f'Expected large before image shape = {expected_large_shape}, '
            f'actual = {large_before_image.shape}')
        if expect_blank_before:
          assert not large_before_image.any(), (
              'Expected blank large before image.')
        else:
          assert large_before_image.any(), (
              'Expected non-blank large before image.')

        large_after_image = _deserialize_image(
            example.features.feature['post_image_png_large'].bytes_list.value[0]
        )
        assert large_after_image.shape == expected_large_shape, (
            f'Expected large after image shape = {expected_large_shape}, '
            f'actual = {large_after_image.shape}')

      actual_coordinates.add((longitude, latitude, label))
      actual_string_labels.append(
          example.features.feature['string_label'].bytes_list.value[0].decode())
      actual_plus_codes.append(
          example.features.feature['plus_code'].bytes_list.value[0].decode()
      )

    assert _unordered_all_close(expected_coordinates, actual_coordinates)
    assert set(expected_string_labels) == set(actual_string_labels)
    assert len(expected_plus_codes) == len(actual_plus_codes)

    expected_plus_codes_sorted = sorted(expected_plus_codes)
    actual_plus_codes_sorted = sorted(actual_plus_codes)
    for expected_plus_code, actual_plus_code in zip(
        expected_plus_codes_sorted, actual_plus_codes_sorted
    ):
      assert expected_plus_code == actual_plus_code

  return _check_examples_internal


class GenerateExamplesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image_path = _get_test_file_path(TEST_IMAGE_PATH)
    self.buildings_path = _get_test_file_path('buildings')
    self.test_image_path_patterns = _get_test_file_path(
        'test_data/country_*.tif'
    )
    self.test_config_path = _get_test_file_path(TEST_CONFIG_PATH)
    self.test_missing_dataset_name_config_path = _get_test_file_path(
        TEST_MISSING_DATASET_NAME_CONFIG_PATH
    )
    self.test_missing_output_dir_config_path = _get_test_file_path(
        TEST_MISSING_OUPTUT_DIR_CONFIG_PATH
    )

  def test_generate_examples_fn(self):
    """Tests GenerateExamplesFn class."""

    _create_buildings_file_with_footprints(
        [(178.482925, -16.632893), (178.482283, -16.632279)],
        self.buildings_path,
    )

    before_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    after_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    with test_pipeline.TestPipeline() as pipeline:
      examples = generate_examples._generate_examples(
          pipeline,
          before_image_info,
          after_image_info,
          self.buildings_path,
          62,
          32,
          0.5,
          {},
      )

      # Example at second input coordinate should be dropped because its patch
      # falls mostly outside the before and after image bounds.
      test_util.assert_that(
          examples,
          _check_examples(
              self.test_image_path,
              self.test_image_path,
              32,
              62,
              [(178.482925, -16.632893, -1.0)],
              [''],
              ['5VMW9F8M+R5V8F4'],
              False,
              True,
              True,
          ),
          label='assert_examples',
      )

  def test_generate_examples_fn_labeled(self):
    """Tests GenerateExamplesFn class."""

    _create_labeled_buildings_file(
        [
            (178.482925, -16.632893, 0.0, 'no_damage'),
            (178.482924, -16.632894, 1.0, 'destroyed'),
        ],
        self.buildings_path,
    )
    before_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    after_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]

    with test_pipeline.TestPipeline() as pipeline:
      examples = generate_examples._generate_examples(
          pipeline,
          before_image_info,
          after_image_info,
          self.buildings_path,
          62,
          32,
          0.5,
          {},
      )

      test_util.assert_that(
          examples,
          _check_examples(
              self.test_image_path,
              self.test_image_path,
              32,
              62,
              [(178.482925, -16.632893, 0.0), (178.482924, -16.632894, 1.0)],
              ['no_damage', 'destroyed'],
              ['5VMW9F8M+R5V8F4', '5VMW9F8M+R5V872'],
              False,
              True,
              False,
          ),
          label='assert_examples',
      )

  def test_generate_examples_fn_with_plus_codes(self):
    """Tests GenerateExamplesFn class."""

    _create_buildings_file_with_plus_code(
        [(178.482925, -16.632893), (178.482283, -16.632279)],
        self.buildings_path,
    )
    before_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    after_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    with test_pipeline.TestPipeline() as pipeline:
      examples = generate_examples._generate_examples(
          pipeline,
          before_image_info,
          after_image_info,
          self.buildings_path,
          62,
          32,
          0.5,
          {},
      )

      # Example at second input coordinate should be dropped because its patch
      # falls mostly outside the before and after image bounds.
      test_util.assert_that(
          examples,
          _check_examples(
              self.test_image_path,
              self.test_image_path,
              32,
              62,
              [(178.482925, -16.632893, -1.0)],
              [''],
              ['abc'],
              False,
              True,
              False,
          ),
          label='assert_examples',
      )

  def test_generate_examples_fn_no_before(self):
    """Tests GenerateExamplesFn class without before image."""

    _create_buildings_file(
        [(178.482925, -16.632893), (178.482283, -16.632279)],
        self.buildings_path,
    )
    after_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    with test_pipeline.TestPipeline() as pipeline:
      examples = generate_examples._generate_examples(
          pipeline, [], after_image_info, self.buildings_path, 62, 32, 0.5, {}
      )

      # Example at second input coordinate should be dropped because its patch
      # falls mostly outside the before and after image bounds.
      test_util.assert_that(
          examples,
          _check_examples(
              '',
              self.test_image_path,
              32,
              62,
              [(178.482925, -16.632893, -1.0)],
              [''],
              ['5VMW9F8M+R5V8F4'],
              True,
              True,
              False,
          ),
          label='assert_examples',
      )

  def test_generate_examples_pipeline(self):
    output_dir = self.create_tempdir().full_path
    _create_buildings_file(
        [(178.482925, -16.632893), (178.482283, -16.632279)],
        self.buildings_path,
    )
    before_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    after_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    generate_examples._generate_examples_pipeline(
        before_image_info=before_image_info,
        after_image_info=after_image_info,
        large_patch_size=32,
        example_patch_size=32,
        resolution=0.5,
        output_dir=output_dir,
        num_output_shards=1,
        buildings_path=self.buildings_path,
        buildings_labeled=False,
        use_dataflow=False,
        gdal_env={},
        dataflow_job_name='test',
        cloud_project=None,
        cloud_region=None,
        worker_service_account=None,
        min_workers=0,
        max_workers=0,
        wait_for_dataflow_job=True,
        cloud_detector_model_path=None,
        output_metadata_file=False,
        output_parquet=False,
        output_images=False,
    )

    tfrecords = os.listdir(
        os.path.join(output_dir, 'examples', 'unlabeled-large')
    )
    self.assertSameElements(['unlabeled-00000-of-00001.tfrecord'], tfrecords)

  def test_generate_examples_pipeline_labeled(self):
    output_dir = self.create_tempdir().full_path
    _create_labeled_buildings_file(
        [
            (178.482925, -16.632893, 0.0, 'no_damage'),
            (178.482924, -16.632894, 1.0, 'destroyed'),
        ],
        self.buildings_path,
    )
    before_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    after_image_info = [
        RasterInfo(self.test_image_path, None, None)
    ]
    generate_examples._generate_examples_pipeline(
        before_image_info=before_image_info,
        after_image_info=after_image_info,
        large_patch_size=32,
        example_patch_size=32,
        resolution=0.5,
        output_dir=output_dir,
        num_output_shards=1,
        buildings_path=self.buildings_path,
        buildings_labeled=True,
        use_dataflow=False,
        gdal_env={},
        dataflow_job_name='test',
        cloud_project=None,
        cloud_region=None,
        worker_service_account=None,
        min_workers=0,
        max_workers=0,
        wait_for_dataflow_job=True,
        cloud_detector_model_path=None,
        output_metadata_file=True,
        output_parquet=True,
        output_images=False,
    )

    tfrecords = os.listdir(
        os.path.join(output_dir, 'examples', 'labeled-large')
    )
    self.assertSameElements(['labeled-00000-of-00001.tfrecord'], tfrecords)
    metadata_pattern = os.path.join(
        output_dir, 'examples', 'metadata', 'metadata.csv-*-of-*'
    )
    metadata = pd.concat([
        pd.read_csv(p, dtype={'string_label': str})
        for p in glob.glob(metadata_pattern)
    ])
    self.assertLen(metadata, 2)
    self.assertCountEqual(
        metadata.columns,
        [
            'example_id',
            'int64_id',
            'encoded_coordinates',
            'longitude',
            'latitude',
            'post_image_id',
            'pre_image_id',
            'plus_code',
            'string_label',
            'label',
            'post_footprint_x_shift_meters',
            'post_footprint_y_shift_meters',
            'post_footprint_match_score',
            'building_image_id',
        ],
    )

    # No assert for example_id as each example_id depends on the image path
    # which varies with platforms where this test is run
    np.testing.assert_equal(
        metadata.encoded_coordinates.values,
        ['A17B32432A1085C1', 'A17B32432B1085C1'],
    )
    np.testing.assert_almost_equal(
        metadata.latitude.values, [-16.632893, -16.632894], decimal=6
    )
    np.testing.assert_almost_equal(
        metadata.longitude.values, [178.482925, 178.482924], decimal=6
    )
    np.testing.assert_equal(
        metadata.pre_image_id.values,
        [self.test_image_path, self.test_image_path],
    )
    np.testing.assert_equal(
        metadata.post_image_id.values,
        [self.test_image_path, self.test_image_path],
    )
    np.testing.assert_equal(
        metadata['plus_code'].values, ['5VMW9F8M+R5V8F4', '5VMW9F8M+R5V872']
    )
    np.testing.assert_equal(
        metadata['string_label'].values, ['no_damage', 'destroyed']
    )
    np.testing.assert_equal(metadata['label'].values, [0.0, 1.0])

    parquet_files = os.listdir(
        os.path.join(output_dir, 'examples', 'labeled-parquet')
    )
    self.assertSameElements(parquet_files, ['examples-00000-of-00001.parquet'])

  def test_config_loaded_correctly_from_json_file(self):
    config = generate_examples.ExamplesGenerationConfig.init_from_json_path(
        self.test_config_path
    )
    # Passed values.
    self.assertEqual(config.dataset_name, 'dummy dataset name')
    self.assertEqual(config.output_dir, 'path/to/output_dir')
    self.assertEqual(config.before_image_config, 'path/to/before_image_config')
    self.assertEqual(config.after_image_config, 'path/to/after_image_config')
    self.assertEqual(config.buildings_method, 'file')
    self.assertEqual(config.buildings_file, 'path/to/building_file')
    self.assertEqual(config.resolution, 0.3)
    self.assertEqual(config.use_dataflow, True)
    self.assertEqual(config.cloud_project, 'project name')
    self.assertEqual(config.cloud_region, 'region')
    self.assertEqual(
        config.worker_service_account,
        'account',
    )
    self.assertEqual(config.max_dataflow_workers, 100)
    self.assertEqual(config.resolution, 0.3)

    # Default values.
    self.assertEqual(config.before_image_patterns, [])
    self.assertEqual(config.after_image_patterns, [])
    self.assertIsNone(config.aoi_path)
    self.assertEqual(config.example_patch_size, 64)
    self.assertEqual(config.large_patch_size, 256)
    self.assertEqual(config.output_shards, 20)
    self.assertEqual(
        config.overpass_url, 'https://lz4.overpass-api.de/api/interpreter'
    )
    self.assertEqual(
        config.open_buildings_feature_collection,
        'GOOGLE/Research/open-buildings/v3/polygons',
    )
    self.assertEqual(config.earth_engine_service_account, '')
    self.assertIsNone(config.earth_engine_private_key)
    self.assertIsNone(config.labels_file)
    self.assertIsNone(config.label_property)
    self.assertIsNone(config.labels_to_classes)
    self.assertIsNone(config.num_keep_labeled_examples)
    self.assertIsNone(config.configuration_path)

  def test_config_raise_error_on_missing_dataset_name(self):
    with self.assertRaisesRegex(KeyError, ''):
      generate_examples.ExamplesGenerationConfig.init_from_json_path(
          self.test_missing_dataset_name_config_path
      )

  def testConfigRaiseErrorOnMissingOutputDir(self):
    with self.assertRaisesRegex(KeyError, ''):
      generate_examples.ExamplesGenerationConfig.init_from_json_path(
          self.test_missing_output_dir_config_path
      )

  def testConfigWithImageInfo(self):
    config = generate_examples.ExamplesGenerationConfig.init_from_json_path(
        _get_test_file_path(TEST_CONFIG_WITH_IMAGE_INFO_PATH)
    )
    # Passed values.
    self.assertEqual(config.dataset_name, 'dummy dataset name')
    self.assertEqual(config.output_dir, 'path/to/output_dir')
    self.assertSameElements(
        config.before_image_info,
        [
            RasterInfo('before/image/1.tif', [4, 3, 2], 12),
            RasterInfo('before/image/2.tif', [1, 2, 3], 8),
            RasterInfo('before/image/3.tif', None, 8),
            RasterInfo('before/image/4.tif', [3, 2, 1], None),
        ],
    )
    self.assertSameElements(
        config.after_image_info,
        [
            RasterInfo('after/image/1.tif', [4, 3, 2], 12),
            RasterInfo('after/image/2.tif', [1, 2, 3], 8),
            RasterInfo('after/image/3.tif', None, None),
        ],
    )

  def test_read_labels_file(self):
    label_property = 'damage'
    labels_file = _create_labeled_geojson(
        [
            (37, -122, 'damaged'),
            (38, -123, 'undamaged'),
        ],
        label_property,
    )
    labels_to_classes = ['damaged=1', 'undamaged=0']
    _, output_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
    generate_examples.read_labels_file(
        labels_file,
        label_property,
        labels_to_classes,
        max_points=0,
        output_path=output_path,
    )
    gdf = gpd.read_parquet(output_path)
    self.assertLen(gdf, 2)
    np.testing.assert_array_equal(gdf['longitude'].values, [37, 38])
    np.testing.assert_array_equal(gdf['latitude'].values, [-122, -123])
    np.testing.assert_array_equal(
        gdf['string_label'].values, ['damaged', 'undamaged']
    )
    np.testing.assert_array_equal(gdf['label'], [1.0, 0.0])
    np.testing.assert_array_equal(
        gdf.geometry.values,
        [
            shapely.geometry.Point(37.0, -122.0),
            shapely.geometry.Point(38.0, -123.0),
        ],
    )

  def test_get_image_infos_from_image_patterns(self):
    config = generate_examples.ExamplesGenerationConfig(
        dataset_name='test',
        output_dir='test',
        before_image_patterns=[self.test_image_path],
        after_image_patterns=[self.test_image_path],
    )
    before_image_info, after_image_info = (
        generate_examples._get_image_infos_from_config(config, {})
    )
    self.assertSameElements(
        before_image_info,
        [read_raster.RasterInfo(self.test_image_path, (1, 2, 3), 8)],
    )
    self.assertSameElements(
        after_image_info,
        [read_raster.RasterInfo(self.test_image_path, (1, 2, 3), 8)],
    )

  def test_get_image_infos_from_image_configs(self):
    temp_dir = pathlib.Path(tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value))
    before_image_config = str(temp_dir / 'before_image_config.txt')
    with open(before_image_config, 'w') as f:
      f.write(f'{self.test_image_path}\n')
    after_image_config = str(temp_dir / 'after_image_config.txt')
    with open(after_image_config, 'w') as f:
      f.write(f'{self.test_image_path}\n')
    config = generate_examples.ExamplesGenerationConfig(
        dataset_name='test',
        output_dir='test',
        before_image_config=before_image_config,
        after_image_config=after_image_config,
    )
    before_image_info, after_image_info = (
        generate_examples._get_image_infos_from_config(config, {})
    )
    self.assertSameElements(
        before_image_info,
        [read_raster.RasterInfo(self.test_image_path, (1, 2, 3), 8)],
    )
    self.assertSameElements(
        after_image_info,
        [read_raster.RasterInfo(self.test_image_path, (1, 2, 3), 8)],
    )

  def test_get_image_infos_from_image_infos(self):
    config = generate_examples.ExamplesGenerationConfig(
        dataset_name='test',
        output_dir='test',
        before_image_info=[
            read_raster.RasterInfo('a.tif', (1, 2, 3), 8),
            read_raster.RasterInfo('b.tif', (2, 3, 4), 12),
        ],
        after_image_info=[
            read_raster.RasterInfo('c.tif', (1, 2, 3), 8),
            read_raster.RasterInfo('d.tif', (2, 3, 4), 12),
        ],
    )
    before_image_info, after_image_info = (
        generate_examples._get_image_infos_from_config(config, {})
    )
    self.assertSameElements(
        before_image_info,
        [
            read_raster.RasterInfo('a.tif', (1, 2, 3), 8),
            read_raster.RasterInfo('b.tif', (2, 3, 4), 12),
        ],
    )
    self.assertSameElements(
        after_image_info,
        [
            read_raster.RasterInfo('c.tif', (1, 2, 3), 8),
            read_raster.RasterInfo('d.tif', (2, 3, 4), 12),
        ],
    )

  def test_align_images(self):
    after_image = np.zeros((128, 128, 3), dtype=np.uint8)
    before_image_size = 64

    # These are the offsets if the before image is aligned exactly at the center
    # of the after image.
    centered_i = (after_image.shape[0] - before_image_size) // 2
    centered_j = (after_image.shape[1] - before_image_size) // 2

    # These are the offsets for the best alignment.
    i = 13
    j = 17

    after_image[i:i + before_image_size, j:j + before_image_size, :] = 255
    after_image[i + before_image_size // 2, j + before_image_size // 2, :] = 0
    before_image = after_image[
        i : i + before_image_size, j : j + before_image_size, :
    ]
    row_shift, col_shift, match_score = generate_examples._align_images(
        before_image, after_image
    )
    self.assertEqual(row_shift, i - centered_i)
    self.assertEqual(col_shift, j - centered_j)
    self.assertEqual(match_score, 1.0)

  def test_align_image_pairs(self):
    blank_image = np.zeros((128, 128, 3), dtype=np.uint8)
    features = [
        generate_examples._FeatureUnion(
            before_image=('before1', blank_image)
        ),
        generate_examples._FeatureUnion(
            before_image=('before2', blank_image)
        ),
        generate_examples._FeatureUnion(
            after_image=('after1', blank_image)
        ),
        generate_examples._FeatureUnion(
            after_image=('after2', blank_image)
        ),
    ]
    longitude = 123
    latitude = 78
    encoded_coordinates = utils.encode_coordinates(longitude, latitude)
    alignments = list(
        generate_examples._align_image_pairs((encoded_coordinates, features))
    )
    self.assertLen(alignments, 4)
    for (before_image_id, after_image_id), alignment in alignments:
      self.assertIn(before_image_id, ['before1', 'before2'])
      self.assertIn(after_image_id, ['after1', 'after2'])
      self.assertAlmostEqual(alignment.longitude, longitude)
      self.assertAlmostEqual(alignment.latitude, latitude)

  def test_fix_alignments(self):
    alignments = [
        generate_examples._Alignment(
            encoded_coordinates='aaa',
            longitude=123,
            latitude=78,
            row_shift=7,
            col_shift=9,
            match_score=0.01,
        ),
        generate_examples._Alignment(
            encoded_coordinates='bbb',
            longitude=123.0000001,
            latitude=78.0000001,
            row_shift=11,
            col_shift=13,
            match_score=0.5,
        ),
        generate_examples._Alignment(
            encoded_coordinates='ccc',
            longitude=124,
            latitude=79,
            row_shift=1,
            col_shift=2,
            match_score=0.7,
        ),
    ]
    fixed_alignments = list(
        generate_examples._fix_alignments((('before1', 'after1'), alignments))
    )
    self.assertCountEqual(
        fixed_alignments,
        [
            (
                b'`\x0f\x9b2\xf8\xbcY\xa0c\x83\x18o\xb3\xdb\xc4\xfa',
                generate_examples._Alignment(
                    encoded_coordinates='aaa',
                    longitude=123,
                    latitude=78,
                    row_shift=11,
                    col_shift=13,
                    match_score=0.01,
                ),
            ),
            (
                b'\xcb\xa7\xf0\xe0\xd6\xab\xf4\xa7\x19>\x07V\x9dp\xba1',
                generate_examples._Alignment(
                    encoded_coordinates='bbb',
                    longitude=123.0000001,
                    latitude=78.0000001,
                    row_shift=11,
                    col_shift=13,
                    match_score=0.5,
                ),
            ),
            (
                b'\xbb\xfa\xe4Ig\xe4\xe9\xad\xfd1m[zrq\xd6',
                generate_examples._Alignment(
                    encoded_coordinates='ccc',
                    longitude=124,
                    latitude=79,
                    row_shift=1,
                    col_shift=2,
                    match_score=0.7,
                ),
            ),
        ],
    )

  def test_fix_alignments_no_good_matches(self):
    alignments = [
        generate_examples._Alignment(
            encoded_coordinates='aaa',
            longitude=123,
            latitude=78,
            row_shift=7,
            col_shift=9,
            match_score=0.01,
        ),
        generate_examples._Alignment(
            encoded_coordinates='bbb',
            longitude=123.0000001,
            latitude=78.0000001,
            row_shift=11,
            col_shift=13,
            match_score=0.01,
        ),
    ]
    fixed_alignments = list(
        generate_examples._fix_alignments((('before1', 'after1'), alignments))
    )
    self.assertCountEqual(
        fixed_alignments,
        [
            (
                b'`\x0f\x9b2\xf8\xbcY\xa0c\x83\x18o\xb3\xdb\xc4\xfa',
                generate_examples._Alignment(
                    encoded_coordinates='aaa',
                    longitude=123,
                    latitude=78,
                    row_shift=7,
                    col_shift=9,
                    match_score=0.01,
                ),
            ),
            (
                b'\xcb\xa7\xf0\xe0\xd6\xab\xf4\xa7\x19>\x07V\x9dp\xba1',
                generate_examples._Alignment(
                    encoded_coordinates='bbb',
                    longitude=123.0000001,
                    latitude=78.0000001,
                    row_shift=11,
                    col_shift=13,
                    match_score=0.01,
                ),
            ),
        ],
    )

  def test_shift_footprint(self):
    footprint = shapely.geometry.Polygon(
        [(1, 1), (1, 2), (2, 2), (2, 1)],
    )
    footprint_wkb = shapely.to_wkb(footprint)
    shifted_footprint_wkb = generate_examples._shift_footprint(
        footprint_wkb, 111000, 222000
    )
    shifted_footprint = shapely.from_wkb(shifted_footprint_wkb)
    self.assertTrue(
        shifted_footprint.equals_exact(
            shapely.geometry.Polygon([(2, 3), (2, 4), (3, 4), (3, 3)]),
            0.1,
        )
    )

  def test_write_example_metadata(self):
    output_dir = self.create_tempdir().full_path
    examples = [
        _create_example('1'),
        _create_example('2'),
        _create_example('3'),
    ]
    with test_pipeline.TestPipeline() as pipeline:
      examples = pipeline | beam.Create(examples)
      generate_examples._write_example_metadata(examples, output_dir)

    metadata_files = os.listdir(
        os.path.join(output_dir, 'examples', 'metadata')
    )
    self.assertIn('metadata-00000-of-00001.parquet', metadata_files)
    self.assertIn('metadata.csv-00000-of-00001', metadata_files)

  @parameterized.named_parameters(
      dict(
          testcase_name='full',
          r1=0,
          r2=127,
          c1=0,
          c2=127,
      ),
      dict(
          testcase_name='center',
          r1=10,
          r2=14,
          c1=9,
          c2=15,
      ),
      dict(
          testcase_name='top_left',
          r1=0,
          r2=14,
          c1=0,
          c2=15,
      ),
      dict(
          testcase_name='bottom_right',
          r1=14,
          r2=127,
          c1=15,
          c2=127,
      ),
      dict(
          testcase_name='top_half',
          r1=0,
          r2=14,
          c1=0,
          c2=127,
      ),
      dict(
          testcase_name='bottom_half',
          r1=14,
          r2=127,
          c1=0,
          c2=127,
      ),
      dict(
          testcase_name='left_half',
          r1=0,
          r2=127,
          c1=0,
          c2=14,
      ),
      dict(
          testcase_name='right_half',
          r1=0,
          r2=127,
          c1=14,
          c2=127,
      ),
  )
  def test_find_blank_rows_cols(self, r1: int, r2: int, c1: int, c2: int):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[r1:r2+1, c1:c2+1, :] = 255
    y1, y2, x1, x2 = generate_examples._find_blank_rows_cols(image)
    self.assertEqual(y1, r1)
    self.assertEqual(y2, 127 - r2)
    self.assertEqual(x1, c1)
    self.assertEqual(x2, 127 - c2)

  def test_find_nonblank_bounds_empty(self):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    y1, y2, x1, x2 = generate_examples._find_blank_rows_cols(image)
    self.assertEqual(y1, -1)
    self.assertEqual(y2, -1)
    self.assertEqual(x1, -1)
    self.assertEqual(x2, -1)

  def test_write_examples_to_tfrecords(self):
    num_examples = 100
    examples = [_create_example(str(i), i) for i in range(num_examples)]
    output_dir = self.create_tempdir().full_path
    num_shards = 5
    with test_pipeline.TestPipeline() as pipeline:
      examples = pipeline | beam.Create(examples)
      generate_examples._write_examples_to_tfrecords(
          examples,
          output_dir,
          'examples',
          num_shards,
          'stage_name',
      )

    for i in range(num_shards):
      shard_path = os.path.join(
          output_dir, f'examples-{i:05d}-of-{num_shards:05d}.tfrecord'
      )
      self.assertTrue(os.path.exists(shard_path))
      shard_ids = [
          tf.train.Example.FromString(r)
          .features.feature['int64_id']
          .int64_list.value[0]
          for r in tf.data.TFRecordDataset([shard_path]).as_numpy_iterator()
      ]
      self.assertLen(shard_ids, num_examples // num_shards)
      for int64_id in shard_ids:
        self.assertEqual(int64_id % num_shards, i)


if __name__ == '__main__':
  absltest.main()
