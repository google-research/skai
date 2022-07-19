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

import io
import pathlib
from typing import List, Tuple
from absl.testing import absltest

import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that
import numpy as np
import PIL.Image
from skai import generate_examples
from skai import utils
import tensorflow as tf

Image = PIL.Image.Image

TEST_IMAGE_PATH = 'test_data/blank.tif'


def _deserialize_image(serialized_image: bytes) -> Image:
  return PIL.Image.open(io.BytesIO(serialized_image), formats=['PNG'])


def _unordered_all_close(list1: List[Tuple[float, float]],
                         list2: List[Tuple[float, float]]) -> bool:
  """Return that two lists of coordinates are close to each other."""
  if len(list1) != len(list2):
    return False

  sorted_list1 = sorted(list1)
  sorted_list2 = sorted(list2)
  return np.allclose(sorted_list1, sorted_list2)


def _check_serialized_examples(expected_patch_size: int,
                               expected_coordinates: List[Tuple[float, float]]):
  """Validates examples generated from beam pipeline.

  Args:
    expected_patch_size: The expected size of encoded patches.
    expected_coordinates: List of coordinates that examples should have.

  Returns:
    Function for validating examples.
  """

  def _check_examples(actual_serialized_examples):
    actual_coordinates = set()
    for serialized in actual_serialized_examples:
      example = tf.train.Example()
      example.ParseFromString(serialized)
      feature_names = set(example.features.feature.keys())
      # TODO(jzxu): Use constants for these feature name strings.
      assert feature_names == set([
          'pre_image_png', 'post_image_png', 'coordinates',
          'encoded_coordinates', 'label'
      ])

      longitude, latitude = (
          example.features.feature['coordinates'].float_list.value)

      before_image = _deserialize_image(
          example.features.feature['pre_image_png'].bytes_list.value[0])
      assert before_image.width == expected_patch_size, (
          f'Expected before image width = {expected_patch_size}, '
          f'actual = {before_image.width}')
      assert before_image.height == expected_patch_size, (
          f'Expected before image height = {expected_patch_size}, '
          f'actual = {before_image.height}')

      after_image = _deserialize_image(
          example.features.feature['post_image_png'].bytes_list.value[0])
      assert after_image.width == expected_patch_size, (
          f'Expected after image width = {expected_patch_size}, '
          f'actual = {after_image.width}')
      assert after_image.height == expected_patch_size, (
          f'Expected after image height = {expected_patch_size}, '
          f'actual = {after_image.height}')

      actual_coordinates.add((longitude, latitude))

    assert _unordered_all_close(expected_coordinates, actual_coordinates)

  return _check_examples


def _check_serialized_images(expected_width: int,
                             expected_height: int,
                             expected_coordinates: List[Tuple[float, float]]):
  """Validates images generated from beam pipeline.

  Args:
    expected_width: The expected size of encoded patches.
    expected_height: The expected size of encoded patches.
    expected_coordinates: List of coordinates that examples should have.

  Returns:
    Function for validating examples.
  """

  def _check_images(actual_serialized_images):
    actual_coordinates = set()
    for name, serialized_image in actual_serialized_images:
      assert name.endswith('.png'), name
      encoded_coords = name[:-4]  # Remove ".png" suffix.
      longitude, latitude = utils.decode_coordinates(encoded_coords)
      actual_coordinates.add((longitude, latitude))
      image = utils.deserialize_image(serialized_image, 'png')
      assert image.width == expected_width
      assert image.height == expected_height

    assert _unordered_all_close(expected_coordinates, actual_coordinates)

  return _check_images


class GenerateExamplesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    current_dir = pathlib.Path(__file__).parent
    self.test_image_path = str(current_dir / TEST_IMAGE_PATH)

  def testGenerateExamplesFn(self):
    """Tests GenerateExamplesFn class."""

    coordinates = [
        generate_examples._Coordinate(178.482925, -16.632893, -1),
        generate_examples._Coordinate(178.482283, -16.632279, -1)]

    with test_pipeline.TestPipeline() as pipeline:
      coordinates_pcollection = (
          pipeline
          | beam.Create(coordinates))

      examples, labeling_images = generate_examples._generate_examples(
          self.test_image_path, self.test_image_path, 32, 64, 62, 0.5, 1, {},
          coordinates_pcollection, 'unlabeled')

      # Example at second input coordinate should be dropped because its patch
      # falls mostly outside the before and after image bounds.
      assert_that(examples, _check_serialized_examples(
          32, [(178.482925, -16.632893)]), label='assert_examples')

      expected_width = 154   # 62 + 62 + 10 * 3
      expected_height = 114  # 62 + 2 * 10 + height for caption
      assert_that(
          labeling_images,
          _check_serialized_images(expected_width, expected_height,
                                   [(178.482925, -16.632893)]),
          label='assert_labeling_images')


if __name__ == '__main__':
  absltest.main()
