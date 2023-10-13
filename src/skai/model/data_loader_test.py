"""Tests for data loaders."""

import os
import tempfile
from typing import List

from absl.testing import absltest
import numpy as np
from skai.model import data
import tensorflow as tf


RESNET_IMAGE_SIZE = 224


def _make_temp_dir() -> str:
  """Create a temporary directory and return its path.

    Returns:
        str: The path to the created temporary directory.
    """
  return tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))


def _make_serialized_image(size: int, pixel_value: int) -> bytes:
  """Create a serialized PNG image with a specified size and pixel value.

    Args:
        size (int): The size (width and height) of the square image.
        pixel_value (int): The pixel value to fill the image with.

    Returns:
        bytes: The serialized PNG image.
    """
  image = np.ones((size, size, 3), dtype=np.uint8) * pixel_value
  return tf.io.encode_png(image).numpy()


def _make_example(
    example_id: str,
    longitude: float,
    latitude: float,
    encoded_coordinates: str,
    label: float,
    string_label: float,
    patch_size: int,
    large_patch_size: int,
    before_pixel_value: int,
    after_pixel_value: int,
) -> tf.train.Example:
  """Create a TensorFlow Example from provided attributes.

    Args:
        example_id (str): Identifier for the example.
        longitude (float): Longitude coordinate.
        latitude (float): Latitude coordinate.
        encoded_coordinates (str): Encoded coordinates.
        label (float): Numeric label.
        string_label (float): String label.
        patch_size (int): Size of the image patches.
        large_patch_size (int): Size of large image patches.
        before_pixel_value (int): Pixel value for "before" images.
        after_pixel_value (int): Pixel value for "after" images.

    Returns:
        tf.train.Example: A TensorFlow Example containing the provided attributes.
    """
  example = tf.train.Example()
  example.features.feature['example_id'].bytes_list.value.append(
      example_id.encode()
  )
  example.features.feature['coordinates'].float_list.value.extend(
      (longitude, latitude)
  )
  example.features.feature['encoded_coordinates'].bytes_list.value.append(
      encoded_coordinates.encode()
  )
  example.features.feature['label'].float_list.value.append(label)
  example.features.feature['string_label'].bytes_list.value.append(
      string_label.encode()
  )
  example.features.feature['pre_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size, before_pixel_value)
  )
  example.features.feature['post_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size, after_pixel_value)
  )
  example.features.feature['pre_image_png_large'].bytes_list.value.append(
      _make_serialized_image(large_patch_size, before_pixel_value)
  )
  example.features.feature['post_image_png_large'].bytes_list.value.append(
      _make_serialized_image(large_patch_size, after_pixel_value)
  )
  return example


def _write_tfrecord(examples: List[tf.train.Example], path: str) -> None:
  """Write a list of TensorFlow Examples to a TFRecord file.

    Args:
        examples (List[tf.train.Example]): List of TensorFlow Examples to write.
        path (str): The path to the output TFRecord file.

    Returns:
        None
    """
  with tf.io.TFRecordWriter(path) as file_writer:
    for example in examples:
      file_writer.write(example.SerializeToString())


def _create_test_data():
  """Create test data and save it to TFRecord files.

    Returns:
        Tuple[str, str, str]: A tuple containing the paths to the labeled training,
        labeled test, and unlabeled TFRecord files.
    """
  examples_dir = _make_temp_dir()
  labeled_train_path = os.path.join(
      examples_dir, 'train_labeled_examples.tfrecord')
  labeled_test_path = os.path.join(
      examples_dir, 'test_labeled_examples.tfrecord')
  unlabeled_path = os.path.join(
      examples_dir, 'unlabeled_examples.tfrecord')

  _write_tfrecord([
      _make_example('1st', 0, 0, 'A0', 0, 'no_damage', 64, 256, 0, 255),
      _make_example('2nd', 0, 1, 'A1', 0, 'no_damage', 64, 256, 0, 255),
      _make_example('3rd', 0, 2, 'A2', 1, 'major_damage', 64, 256, 0, 255),
  ], labeled_train_path)

  _write_tfrecord([
      _make_example('4th', 1, 0, 'B0', 0, 'no_damage', 64, 256, 0, 255),
  ], labeled_test_path)

  _write_tfrecord([
      _make_example('5th', 2, 0, 'C0', -1, 'bad_example', 64, 256, 0, 255),
      _make_example('6th', 2, 1, 'C1', -1, 'bad_example', 64, 256, 0, 255),
      _make_example('7th', 2, 2, 'C2', -1, 'bad_example', 64, 256, 0, 255),
      _make_example('8th', 2, 3, 'C3', -1, 'bad_example', 64, 256, 0, 255),
  ], unlabeled_path)

  return labeled_train_path, labeled_test_path, unlabeled_path


class DataLoaderTest(absltest.TestCase):
  def setUp(self):
    """Set up the test environment.

        This method is called before each test case to prepare the necessary data.
        """
    super().setUp()

    labeled_train_path, labeled_test_path, unlabeled_path = _create_test_data()
    self.labeled_train_path = labeled_train_path
    self.labeled_test_path = labeled_test_path
    self.unlabeled_path = unlabeled_path

  def test_get_skai_dataset_post_only(self):
    """Test loading the SKAI dataset with post-disaster images only.

        This test checks if the SKAI dataset is loaded correctly with post-disaster
        images only and validates the shape and dtype of the loaded images.
        """
    dataset_builder = data.get_dataset('skai')

    kwargs = {
        'labeled_train_pattern': self.labeled_train_path,
        'unlabeled_train_pattern': self.unlabeled_path,
        'validation_pattern': self.labeled_test_path,
        'use_post_disaster_only': True,
        'data_dir': _make_temp_dir(),
    }

    dataloader = dataset_builder(
        1,
        initial_sample_proportion=1,
        subgroup_ids=(),
        subgroup_proportions=(),
        **kwargs
    )
    ds = dataloader.train_ds
    features = next(ds.as_numpy_iterator())
    self.assertIn('input_feature', features)
    self.assertIn('large_image', features['input_feature'])
    input_feature = features['input_feature']['large_image']
    self.assertEqual(
        input_feature.shape,
        (RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3),
    )
    self.assertEqual(input_feature.dtype, np.float32)
    np.testing.assert_equal(input_feature, 1.0)

  def test_get_skai_dataset_pre_post(self):
    """Test loading the SKAI dataset with both pre and post-disaster images.

        This test checks if the SKAI dataset is loaded correctly with both
        pre and post-disaster images and validates the shape and dtype of the loaded
        images.
        """
    dataset_builder = data.get_dataset('skai')

    kwargs = {
        'labeled_train_pattern': self.labeled_train_path,
        'unlabeled_train_pattern': self.unlabeled_path,
        'validation_pattern': self.labeled_test_path,
        'use_post_disaster_only': False,
        'data_dir': _make_temp_dir(),
    }

    dataloader = dataset_builder(
        1,
        initial_sample_proportion=1,
        subgroup_ids=(),
        subgroup_proportions=(),
        **kwargs
    )
    ds = dataloader.train_ds
    features = next(ds.as_numpy_iterator())
    self.assertIn('input_feature', features)
    self.assertIn('large_image', features['input_feature'])
    input_feature = features['input_feature']['large_image']
    self.assertEqual(
        input_feature.shape, (RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 6)
    )
    self.assertEqual(input_feature.dtype, np.float32)
    np.testing.assert_equal(input_feature[:, :, :3], 0.0)
    np.testing.assert_equal(input_feature[:, :, 3:], 1.0)

  def test_get_skai_dataset_small_images(self):
    """Test loading the SKAI dataset with small images.

        This test checks if the SKAI dataset is loaded correctly with small images
        and validates the shape and dtype of the loaded images.
        """
    dataset_builder = data.get_dataset('skai')

    kwargs = {
        'labeled_train_pattern': self.labeled_train_path,
        'unlabeled_train_pattern': self.unlabeled_path,
        'validation_pattern': self.labeled_test_path,
        'use_post_disaster_only': False,
        'data_dir': _make_temp_dir(),
        'load_small_images': True,
    }

    dataloader = dataset_builder(
        1,
        initial_sample_proportion=1,
        subgroup_ids=(),
        subgroup_proportions=(),
        **kwargs
    )
    ds = dataloader.train_ds
    features = next(ds.as_numpy_iterator())
    self.assertIn('input_feature', features)
    self.assertIn('small_image', features['input_feature'])
    self.assertIn('large_image', features['input_feature'])
    small_image = features['input_feature']['small_image']
    large_image = features['input_feature']['large_image']
    self.assertEqual(
        small_image.shape, (RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 6)
    )
    self.assertEqual(small_image.dtype, np.float32)
    np.testing.assert_equal(small_image[:, :, :3], 0.0)
    np.testing.assert_equal(small_image[:, :, 3:], 1.0)

    self.assertEqual(
        large_image.shape, (RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 6)
    )
    self.assertEqual(small_image.dtype, np.float32)
    np.testing.assert_equal(large_image[:, :, :3], 0.0)
    np.testing.assert_equal(large_image[:, :, 3:], 1.0)

  def test_upsample_subgroup(self):
    """Test upsampling a subgroup in the dataset.

        This test checks if a subgroup in the dataset is upsampled correctly based
        on the specified lambda_value and validates the resulting dataset sizes.
        """
    dataset_builder = data.get_dataset('skai')

    kwargs = {
        'labeled_train_pattern': self.labeled_train_path,
        'unlabeled_train_pattern': self.unlabeled_path,
        'validation_pattern': self.labeled_test_path,
        'use_post_disaster_only': False,
        'data_dir': _make_temp_dir(),
    }

    dataloader = dataset_builder(
        1,
        initial_sample_proportion=1,
        subgroup_ids=(),
        subgroup_proportions=(),
        **kwargs)
    ds = dataloader.train_ds
    subgroup_sizes = data.get_subgroup_sizes(ds)
    self.assertEqual(subgroup_sizes['0'], 2)
    self.assertEqual(subgroup_sizes['1'], 1)
    lambda_value = 10
    upsampled_ds = data.upsample_subgroup(
        ds, lambda_value, 'subgroup_label', subgroup_sizes
    )
    self.assertLen(
        list(
            upsampled_ds.filter(
                lambda x: tf.math.equal(x['subgroup_label'], 0)
            ).as_numpy_iterator()
        ),
        2,
    )
    self.assertLen(
        list(
            upsampled_ds.filter(
                lambda x: tf.math.equal(x['subgroup_label'], 1)
            ).as_numpy_iterator()
        ),
        1 * lambda_value,
    )


if __name__ == '__main__':
  absltest.main()
