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

"""Tests for generate_examples.py."""

import os
import pathlib
import tempfile
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as test_util
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from skai import buildings
from skai import generate_examples
from skai import read_raster
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


def _check_examples(
    before_image_id: str,
    after_image_id: str,
    small_patch_size: int,
    large_patch_size: int,
    expected_coordinates: list[tuple[float, float, float]],
    expected_string_labels: list[str],
    expected_plus_codes: list[str],
    expect_blank_before: bool,
    expect_large_patch: bool):
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

  Returns:
    Function for validating examples.
  """

  def _check_examples_internal(actual_examples):
    actual_coordinates = set()
    actual_string_labels = []
    actual_plus_codes = []
    expected_small_shape = (small_patch_size, small_patch_size, 3)
    expected_large_shape = (large_patch_size, large_patch_size, 3)

    for example in actual_examples:
      feature_names = set(example.features.feature.keys())
      # TODO(jzxu): Use constants for these feature name strings.
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
      ])
      if expect_large_patch:
        expected_feature_names.update(
            ['pre_image_png_large', 'post_image_png_large']
        )
      assert (
          feature_names == expected_feature_names
      ), f'Feature set does not match. Got: {" ".join(feature_names)}'

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

  def testGenerateExamplesFn(self):
    """Tests GenerateExamplesFn class."""

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
          'unlabeled',
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
          ),
          label='assert_examples',
      )

  def testGenerateExamplesFnLabeled(self):
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
          'labeled',
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
          ),
          label='assert_examples',
      )

  def testGenerateExamplesFnWithPlusCodes(self):
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
          'unlabeled',
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
          ),
          label='assert_examples',
      )

  def testGenerateExamplesFnNoBefore(self):
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
          pipeline, [], after_image_info, self.buildings_path, 62, 32,
          0.5, {}, 'unlabeled')

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
          ),
          label='assert_examples',
      )

  def testGenerateExamplesPipeline(self):
    output_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
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
        max_workers=0,
        wait_for_dataflow_job=True,
        cloud_detector_model_path=None,
        output_metadata_file=False)

    tfrecords = os.listdir(
        os.path.join(output_dir, 'examples', 'unlabeled-large')
    )
    self.assertSameElements(tfrecords, ['unlabeled-00000-of-00001.tfrecord'])

  def testGenerateExamplesWithOutputMetaDataFile(self):
    output_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
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
        max_workers=0,
        wait_for_dataflow_job=True,
        cloud_detector_model_path=None,
        output_metadata_file=True,
    )

    tfrecords = os.listdir(
        os.path.join(output_dir, 'examples', 'unlabeled-large')
    )
    df_metadata_contents = pd.read_csv(
        os.path.join(output_dir, 'examples', 'metadata_examples.csv')
    )

    # No assert for example_id as each example_id depends on the image path
    # which varies with platforms where this test is run
    self.assertEqual(
        df_metadata_contents.encoded_coordinates[0], 'A17B32432A1085C1'
    )
    self.assertAlmostEqual(
        df_metadata_contents.latitude[0], -16.632892608642578
    )
    self.assertAlmostEqual(
        df_metadata_contents.longitude[0], 178.48292541503906
    )
    self.assertEqual(df_metadata_contents.pre_image_id[0], self.test_image_path)
    self.assertEqual(
        df_metadata_contents.post_image_id[0], self.test_image_path
    )
    self.assertEqual(df_metadata_contents.plus_code[0], '5VMW9F8M+R5V8F4')
    self.assertSameElements(tfrecords, ['unlabeled-00000-of-00001.tfrecord'])

  def testConfigLoadedCorrectlyFromJsonFile(self):
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

  def testConfigRaiseErrorOnMissingDatasetName(self):
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


if __name__ == '__main__':
  absltest.main()
