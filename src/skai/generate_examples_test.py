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

import glob
import os
import pathlib
import tempfile
from typing import Any, List, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import numpy as np
from skai import generate_examples
from skai import utils
import tensorflow as tf


TEST_IMAGE_PATH = 'test_data/blank.tif'
TEST_CONFIG_PATH = 'test_data/config.json'
TEST_MISSING_DATASET_NAME_CONFIG_PATH = (
    'test_data/missing_dataset_name_config.json'
)
TEST_MISSING_OUPTUT_DIR_CONFIG_PATH = 'test_data/missing_output_dir_config.json'


def _get_before_image_id(example):
  return example.features.feature['pre_image_id'].bytes_list.value[0].decode()


def _deserialize_image(serialized_image: bytes) -> np.ndarray:
  return tf.io.decode_image(serialized_image).numpy()


def _unordered_all_close(list1: List[Any], list2: List[Any]) -> bool:
  """Return that two lists of coordinates are close to each other."""
  if len(list1) != len(list2):
    return False

  sorted_list1 = sorted(list1)
  sorted_list2 = sorted(list2)
  return np.allclose(sorted_list1, sorted_list2)


def _check_examples(
    before_image_id: str,
    after_image_id: str,
    small_patch_size: int,
    large_patch_size: int,
    expected_coordinates: List[Tuple[float, float, float]],
    expected_string_labels: List[str],
    expected_plus_codes: List[str],
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
          'string_label'
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
    current_dir = pathlib.Path(__file__).parent
    self.test_image_path = str(current_dir / TEST_IMAGE_PATH)
    self.coordinates_path = str(current_dir / 'coordinates')
    self.test_image_path_patterns = str(current_dir / 'test_data/country_*.tif')
    self.test_config_path = str(current_dir / TEST_CONFIG_PATH)
    self.test_missing_dataset_name_config_path = str(
        current_dir / TEST_MISSING_DATASET_NAME_CONFIG_PATH
    )
    self.test_missing_output_dir_config_path = str(
        current_dir / TEST_MISSING_OUPTUT_DIR_CONFIG_PATH
    )

  def testGenerateExamplesFn(self):
    """Tests GenerateExamplesFn class."""

    unlabeled_coordinates = [(178.482925, -16.632893, -1.0, ''),
                             (178.482283, -16.632279, -1.0, '')]
    utils.write_coordinates_file(unlabeled_coordinates, self.coordinates_path)

    with test_pipeline.TestPipeline() as pipeline:
      large_examples, small_examples = generate_examples._generate_examples(
          pipeline, [self.test_image_path], [self.test_image_path],
          self.coordinates_path, 62, 32, 0.5, {}, 'unlabeled')

      # Example at second input coordinate should be dropped because its patch
      # falls mostly outside the before and after image bounds.
      assert_that(
          large_examples,
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
          label='assert_large_examples',
      )
      assert_that(
          small_examples,
          _check_examples(
              self.test_image_path,
              self.test_image_path,
              32,
              62,
              [(178.482925, -16.632893, -1.0)],
              [''],
              ['5VMW9F8M+R5V8F4'],
              False,
              False,
          ),
          label='assert_small_examples',
      )

  def testGenerateExamplesFnLabeled(self):
    """Tests GenerateExamplesFn class."""

    labeled_coordinates = [(178.482925, -16.632893, 0, 'no_damage'),
                           (178.482924, -16.632894, 1, 'destroyed')]
    utils.write_coordinates_file(labeled_coordinates, self.coordinates_path)

    with test_pipeline.TestPipeline() as pipeline:
      large_examples, small_examples = generate_examples._generate_examples(
          pipeline, [self.test_image_path], [self.test_image_path],
          self.coordinates_path, 62, 32, 0.5, {}, 'labeled')

      assert_that(
          large_examples,
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
          label='assert_large_examples',
      )

      assert_that(
          small_examples,
          _check_examples(
              self.test_image_path,
              self.test_image_path,
              32,
              62,
              [(178.482925, -16.632893, 0.0), (178.482924, -16.632894, 1.0)],
              ['no_damage', 'destroyed'],
              ['5VMW9F8M+R5V8F4', '5VMW9F8M+R5V872'],
              False,
              False,
          ),
          label='assert_small_examples',
      )

  def testGenerateExamplesFnNoBefore(self):
    """Tests GenerateExamplesFn class without before image."""

    coordinates = [(178.482925, -16.632893, -1.0, ''),
                   (178.482283, -16.632279, -1.0, '')]
    utils.write_coordinates_file(coordinates, self.coordinates_path)

    with test_pipeline.TestPipeline() as pipeline:
      large_examples, small_examples = generate_examples._generate_examples(
          pipeline, [], [self.test_image_path], self.coordinates_path, 62, 32,
          0.5, {}, 'unlabeled')

      # Example at second input coordinate should be dropped because its patch
      # falls mostly outside the before and after image bounds.
      assert_that(
          large_examples,
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
          label='assert_large_examples',
      )

      assert_that(
          small_examples,
          _check_examples(
              '',
              self.test_image_path,
              32,
              62,
              [(178.482925, -16.632893, -1.0)],
              [''],
              ['5VMW9F8M+R5V8F4'],
              True,
              False,
          ),
          label='assert_small_examples',
      )

  def testGenerateExampleFnPathPattern(self):
    """Test GenerateExampleFn class with a path pattern."""
    coordinates = [(178.482925, -16.632893, -1.0, '')]
    utils.write_coordinates_file(coordinates, self.coordinates_path)

    expected_before_image_ids = glob.glob(self.test_image_path_patterns)

    with test_pipeline.TestPipeline() as pipeline:
      # The path patterns specify two before images.
      large_examples, small_examples = generate_examples._generate_examples(
          pipeline, [self.test_image_path_patterns],
          [self.test_image_path], self.coordinates_path, 62, 32, 0.5,
          {}, 'unlabeled')

      small_examples_before_ids = (
          small_examples | 'Map small examples to before image ids' >>
          beam.Map(_get_before_image_id))
      large_examples_before_ids = (
          large_examples | 'Map large examples to before image ids' >>
          beam.Map(_get_before_image_id))

      assert_that(small_examples_before_ids,
                  equal_to(expected_before_image_ids),
                  'check small examples before image ids')

      assert_that(large_examples_before_ids,
                  equal_to(expected_before_image_ids),
                  'check large examples before image ids')

  def testGenerateExamplesPipeline(self):
    output_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
    coordinates = [(178.482925, -16.632893), (178.482283, -16.632279)]
    generate_examples.generate_examples_pipeline(
        before_image_patterns=[self.test_image_path],
        after_image_patterns=[self.test_image_path],
        large_patch_size=32,
        example_patch_size=32,
        resolution=0.5,
        output_dir=output_dir,
        num_output_shards=1,
        unlabeled_coordinates=coordinates,
        labeled_coordinates=[],
        use_dataflow=False,
        gdal_env={},
        dataflow_job_name='test',
        cloud_project=None,
        cloud_region=None,
        worker_service_account=None,
        max_workers=0)

    tfrecords = os.listdir(os.path.join(output_dir, 'examples', 'unlabeled'))
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
        'GOOGLE/Research/open-buildings/v2/polygons',
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

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_before_and_after_image_patterns',
          before_image_patterns=[],
          after_image_patterns=[],
      ),
      dict(
          testcase_name='empty_after_image_patterns',
          before_image_patterns=['pattern_a', 'pattern_b'],
          after_image_patterns=[],
      ),
      dict(
          testcase_name='duplicate_after_image_patterns',
          before_image_patterns=['pattern_a', 'pattern_b'],
          after_image_patterns=[
              'pattern_c',
              'pattern_d',
              'pattern_c',
              'pattern_c',
              'pattern_d',
              'pattern_e'
          ],
      ),
      dict(
          testcase_name='duplicate_before_image_patterns',
          before_image_patterns=['pattern_a', 'pattern_b', 'pattern_a'],
          after_image_patterns=['pattern_c'],
      ),
      dict(
          testcase_name='duplicate_before_and_empty_after_image_patterns',
          before_image_patterns=['pattern_a', 'pattern_b', 'pattern_a'],
          after_image_patterns=[],
      ),
  )
  def testValidateImagePatternsRaises(
      self, before_image_patterns, after_image_patterns
  ):
    with self.assertRaises(ValueError):
      generate_examples.validate_image_patterns(before_image_patterns, False)
      generate_examples.validate_image_patterns(after_image_patterns, True)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_before_image_patterns',
          before_image_patterns=[],
          after_image_patterns=['pattern_a', 'pattern_b'],
      ),
      dict(
          testcase_name='no_empty_before_and_after_image_patterns',
          before_image_patterns=['pattern_a', 'pattern_b'],
          after_image_patterns=['pattern_c'],
      ),
  )
  def testValidateImagePatternsNoRaises(
      self, before_image_patterns, after_image_patterns
  ):
    generate_examples.validate_image_patterns(before_image_patterns, False)
    generate_examples.validate_image_patterns(after_image_patterns, True)


if __name__ == '__main__':
  absltest.main()
