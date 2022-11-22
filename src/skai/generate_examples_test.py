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
from typing import Any, List, Tuple
from absl.testing import absltest

from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that
import numpy as np
from skai import generate_examples
from skai import utils
import tensorflow as tf

TEST_IMAGE_PATH = 'test_data/blank.tif'


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
    expect_blank_before: bool,
    expect_large_patch: bool):
  """Validates examples generated from beam pipeline.

  Args:
    before_image_id: Expected before image id.
    after_image_id: Expected after image id.
    small_patch_size: The expected size of small patches.
    large_patch_size: The expected size of large patches.
    expected_coordinates: List of coordinates that examples should have.
    expect_blank_before: If true, the before image should be all zeros.
    expect_large_patch: If true, the examples should contain large patches.

  Returns:
    Function for validating examples.
  """

  def _check_examples_internal(actual_examples):
    actual_coordinates = set()
    expected_small_shape = (small_patch_size, small_patch_size, 3)
    expected_large_shape = (large_patch_size, large_patch_size, 3)

    for example in actual_examples:
      feature_names = set(example.features.feature.keys())
      # TODO(jzxu): Use constants for these feature name strings.
      expected_feature_names = [
          'pre_image_png', 'pre_image_id', 'post_image_png', 'post_image_id',
          'coordinates', 'encoded_coordinates', 'label'
      ]
      if expect_large_patch:
        expected_feature_names.extend(
            ['pre_image_png_large', 'post_image_png_large'])
      assert feature_names == set(
          expected_feature_names
      ), f'Feature set does not match. Got: {" ".join(feature_names)}'

      actual_before_id = example.features.feature[
          'pre_image_id'].bytes_list.value[0].decode()
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

    assert _unordered_all_close(expected_coordinates, actual_coordinates)

  return _check_examples_internal


class GenerateExamplesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    current_dir = pathlib.Path(__file__).parent
    self.test_image_path = str(current_dir / TEST_IMAGE_PATH)
    self.coordinates_path = str(current_dir / 'coordinates')

  def testGenerateExamplesFn(self):
    """Tests GenerateExamplesFn class."""

    unlabeled_coordinates = [(178.482925, -16.632893, -1.0),
                             (178.482283, -16.632279, -1.0)]
    utils.write_coordinates_file(unlabeled_coordinates, self.coordinates_path)

    with test_pipeline.TestPipeline() as pipeline:
      large_examples, small_examples = generate_examples._generate_examples(
          pipeline, [self.test_image_path], [self.test_image_path],
          self.coordinates_path, 62, 32, 0.5, {}, 'unlabeled')

      # Example at second input coordinate should be dropped because its patch
      # falls mostly outside the before and after image bounds.
      assert_that(
          large_examples,
          _check_examples(self.test_image_path, self.test_image_path, 32, 62,
                          [(178.482925, -16.632893, -1.0)], False, True),
          label='assert_large_examples')
      assert_that(
          small_examples,
          _check_examples(self.test_image_path, self.test_image_path, 32, 62,
                          [(178.482925, -16.632893, -1.0)], False, False),
          label='assert_small_examples')

  def testGenerateExamplesFnLabeled(self):
    """Tests GenerateExamplesFn class."""

    labeled_coordinates = [(178.482925, -16.632893, 0),
                           (178.482924, -16.632894, 1)]
    utils.write_coordinates_file(labeled_coordinates, self.coordinates_path)

    with test_pipeline.TestPipeline() as pipeline:
      large_examples, small_examples = generate_examples._generate_examples(
          pipeline, [self.test_image_path], [self.test_image_path],
          self.coordinates_path, 62, 32, 0.5, {}, 'labeled')

      assert_that(
          large_examples,
          _check_examples(self.test_image_path, self.test_image_path, 32, 62,
                          [(178.482925, -16.632893, 0.0),
                           (178.482924, -16.632894, 1.0)], False, True),
          label='assert_large_examples')

      assert_that(
          small_examples,
          _check_examples(self.test_image_path, self.test_image_path, 32, 62,
                          [(178.482925, -16.632893, 0.0),
                           (178.482924, -16.632894, 1.0)], False, False),
          label='assert_small_examples')

  def testGenerateExamplesFnNoBefore(self):
    """Tests GenerateExamplesFn class without before image."""

    coordinates = [(178.482925, -16.632893, -1.0),
                   (178.482283, -16.632279, -1.0)]
    utils.write_coordinates_file(coordinates, self.coordinates_path)

    with test_pipeline.TestPipeline() as pipeline:
      large_examples, small_examples = generate_examples._generate_examples(
          pipeline, [], [self.test_image_path], self.coordinates_path, 62, 32,
          0.5, {}, 'unlabeled')

      # Example at second input coordinate should be dropped because its patch
      # falls mostly outside the before and after image bounds.
      assert_that(
          large_examples,
          _check_examples('', self.test_image_path, 32, 62,
                          [(178.482925, -16.632893, -1.0)], True, True),
          label='assert_large_examples')

      assert_that(
          small_examples,
          _check_examples('', self.test_image_path, 32, 62,
                          [(178.482925, -16.632893, -1.0)], True, False),
          label='assert_small_examples')

  def testGenerateExamplesPipeline(self):
    output_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
    coordinates = [(178.482925, -16.632893), (178.482283, -16.632279)]
    generate_examples.generate_examples_pipeline(
        before_image_paths=[self.test_image_path],
        after_image_paths=[self.test_image_path],
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
        dataflow_container_image=None,
        cloud_project=None,
        cloud_region=None,
        worker_service_account=None,
        max_workers=0)

    tfrecords = os.listdir(os.path.join(output_dir, 'examples', 'unlabeled'))
    self.assertSameElements(tfrecords, ['unlabeled-00000-of-00001.tfrecord'])


if __name__ == '__main__':
  absltest.main()
