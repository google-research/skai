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

"""Tests for skai.semi_supervised.augment.ctaugment."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
from skai.semi_supervised.augment import ctaugment
import tensorflow.compat.v1 as tf

# See https://arxiv.org/pdf/2001.07685.pdf Table 13 for valid ranges of each
# transformation level
TRANSFORMATION_LEVEL = 0.8  # Degree of severity with which to apply
                             # transformation, used as `level` arg


class CTAugmentTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_policy(self):
    augmenter2 = ctaugment.CTAugment(depth=2)
    self.assertLen(augmenter2.policy(True), 2)
    self.assertLen(augmenter2.policy(False), 2)
    augmenter3 = ctaugment.CTAugment(depth=3)
    self.assertLen(augmenter3.policy(True), 3)
    self.assertLen(augmenter3.policy(False), 3)

  def test_apply_policy(self):
    augmenter = ctaugment.CTAugment(depth=2)
    policy = augmenter.policy(True)
    # TODO(jlee24): Make cutout size a flag to specify
    image = np.ones((64, 64, 6)) * 128
    augmented_image = ctaugment.apply(image, policy)
    self.assertEqual(augmented_image.shape, (64, 64, 6))
    self.assertNotAllEqual(image, augmented_image)

  def test_update_rates(self):
    augmenter = ctaugment.CTAugment(depth=2)
    policy = augmenter.policy(False)
    stats_before_update = augmenter.stats()
    augmenter.update_rates(policy, 0.1)
    stats_after_update = augmenter.stats()
    self.assertNotEqual(stats_before_update, stats_after_update)

  def test_cutout_numpy(self):
    image = np.ones((64, 64, 3))
    image[:, :] = (250, 0, 0)  # Set pixels to red
    transformed_image = ctaugment.cutout_numpy(image.astype('uint8'))
    self.assertNotAllEqual(image, transformed_image)
    self.assertAllEqual(image.shape, transformed_image.shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'autocontrast',
          'transformation': ctaugment.autocontrast,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'blur',
          'transformation': ctaugment.blur,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'brightness',
          'transformation': ctaugment.brightness,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'color',
          'transformation': ctaugment.color,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'contrast',
          'transformation': ctaugment.contrast,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'equalize',
          'transformation': ctaugment.equalize,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'invert',
          'transformation': ctaugment.invert,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'posterize',
          'transformation': ctaugment.posterize,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'rescale',
          'transformation': ctaugment.rescale,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'rotate',
          'transformation': ctaugment.rotate,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'sharpness',
          'transformation': ctaugment.sharpness,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'shear_x',
          'transformation': ctaugment.shear_x,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'shear_y',
          'transformation': ctaugment.shear_y,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'smooth',
          'transformation': ctaugment.smooth,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'solarize',
          'transformation': ctaugment.solarize,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'translate_x',
          'transformation': ctaugment.translate_x,
          'level': TRANSFORMATION_LEVEL
      }, {
          'testcase_name': 'translate_y',
          'transformation': ctaugment.translate_y,
          'level': TRANSFORMATION_LEVEL
      })
  def test_transformation(self, transformation, level):
    image = np.ones((64, 64, 3))
    image[:10, :] = 0  # Set some pixels to black
    image[10:30, :] = (250, 0, 0)  # Set some pixels to red
    image[:, 10:] = (0, 0, 200)  # Set some pixels to blue
    image = Image.fromarray(image.astype('uint8'))
    if transformation == ctaugment.rescale:  # requires a scaling method too
      transformed_image = transformation(image, level, TRANSFORMATION_LEVEL)
    else:
      transformed_image = transformation(image, level)
    self.assertNotEqual(image, transformed_image)
    self.assertAllEqual(image.size, transformed_image.size)

if __name__ == '__main__':
  absltest.main()
