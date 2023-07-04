"""Tests for randaugment."""

from absl.testing import parameterized
import numpy as np
from skai.model import randaugment
import tensorflow.compat.v1 as tf

from google3.testing.pybase import googletest


class RandaugmentTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    image_colored = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_white = np.ones((3, 224, 224, 3), dtype=np.uint8) * 255
    self.input_feature_colored = np.concatenate(
        [image_colored, image_colored], axis=-1
    )
    self.input_feature_white = np.concatenate(
        [image_white, image_white], axis=-1
    )

  def test_autocontrast(self):
    aug_image = randaugment.autocontrast(self.input_feature_colored)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_equalize(self):
    aug_image = randaugment.equalize(self.input_feature_colored)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_invert(self):
    aug_image = randaugment.invert(self.input_feature_colored)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_posterize(self):
    aug_image = randaugment.posterize(self.input_feature_colored, 2)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_solarize(self):
    aug_image = randaugment.solarize(self.input_feature_colored)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_solarize_add(self):
    aug_image = randaugment.solarize_add(self.input_feature_colored, 1)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_color(self):
    aug_image = randaugment.color(self.input_feature_colored, 0.5)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_contrast(self):
    aug_image = randaugment.contrast(self.input_feature_colored, 0.5)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_brightness(self):
    aug_image = randaugment.brightness(self.input_feature_colored, 0.5)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_sharpness(self):
    aug_image = randaugment.sharpness(self.input_feature_colored, 0.5)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_shear_x(self):
    aug_image = randaugment.shear_x(self.input_feature_colored, 0.5, [128] * 3)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_shear_y(self):
    aug_image = randaugment.shear_y(self.input_feature_colored, 0.5, [128] * 3)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_translate_x(self):
    aug_image = randaugment.translate_x(
        self.input_feature_colored, 128, [128] * 3
    )
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_translate_y(self):
    aug_image = randaugment.translate_y(
        self.input_feature_colored, 128, [128] * 3
    )
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_cutout(self):
    aug_image = randaugment.cutout(self.input_feature_colored, 4, [128] * 3)
    self.assertNotAllEqual(self.input_feature_colored, aug_image)
    self.assertNotAllEqual(aug_image[..., :3], aug_image[..., 3:])

  def test_rotate(self):
    aug_image = randaugment.rotate(self.input_feature_white, 90)
    self.assertAllEqual(self.input_feature_white, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

    aug_image = randaugment.rotate(self.input_feature_white, 180)
    self.assertAllEqual(self.input_feature_white, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

    aug_image = randaugment.rotate(self.input_feature_white, 270)
    self.assertAllEqual(self.input_feature_white, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

    aug_image = randaugment.rotate(self.input_feature_white, 360)
    self.assertAllEqual(self.input_feature_white, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])

    aug_image = randaugment.rotate(self.input_feature_white, 30)
    self.assertNotAllEqual(self.input_feature_white, aug_image)
    self.assertAllEqual(aug_image[..., :3], aug_image[..., 3:])


if __name__ == "__main__":
  googletest.main()
