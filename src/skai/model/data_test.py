"""Tests for data sets."""

import os
import tempfile
from typing import List
from absl.testing import absltest
import numpy as np
from skai.model import data
import tensorflow as tf
import tensorflow_datasets as tfds


def _make_temp_dir() -> str:
  return tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))


def _make_serialized_image(size: int) -> bytes:
  image = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
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
) -> tf.train.Example:
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
      _make_serialized_image(patch_size)
  )
  example.features.feature['post_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size)
  )
  example.features.feature['pre_image_png_large'].bytes_list.value.append(
      _make_serialized_image(large_patch_size)
  )
  example.features.feature['post_image_png_large'].bytes_list.value.append(
      _make_serialized_image(large_patch_size)
  )
  return example


def _write_tfrecord(examples: List[tf.train.Example], path: str) -> None:
  with tf.io.TFRecordWriter(path) as file_writer:
    for example in examples:
      file_writer.write(example.SerializeToString())


def _create_test_data():
  examples_dir = _make_temp_dir()
  labeled_train_path = os.path.join(
      examples_dir, 'train_labeled_examples.tfrecord')
  labeled_test_path = os.path.join(
      examples_dir, 'test_labeled_examples.tfrecord')
  unlabeled_path = os.path.join(
      examples_dir, 'unlabeled_examples.tfrecord')

  _write_tfrecord([
      _make_example('1st', 0, 0, 'A0', 0, 'no_damage', 64, 256),
      _make_example('2nd', 0, 1, 'A1', 0, 'no_damage', 64, 256),
      _make_example('3rd', 0, 2, 'A2', 1, 'major_damage', 64, 256),
  ], labeled_train_path)

  _write_tfrecord([
      _make_example('4th', 1, 0, 'B0', 0, 'no_damage', 64, 256),
  ], labeled_test_path)

  _write_tfrecord([
      _make_example('5th', 2, 0, 'C0', -1, 'bad_example', 64, 256),
      _make_example('6th', 2, 1, 'C1', -1, 'bad_example', 64, 256),
      _make_example('7th', 2, 2, 'C2', -1, 'bad_example', 64, 256),
      _make_example('8th', 2, 3, 'C3', -1, 'bad_example', 64, 256),
  ], unlabeled_path)

  return labeled_train_path, labeled_test_path, unlabeled_path


class SkaiDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for Skai dataset."""

  DATASET_CLASS = data.SkaiDataset
  SPLITS = {
      'labeled_train': 3,
      'labeled_test': 1,
      'unlabeled': 4
  }
  EXAMPLE_DIR = _make_temp_dir()
  BUILDER_CONFIG_NAMES_TO_TEST = ['test_config']
  SKIP_TF1_GRAPH_MODE = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    labeled_train_path, labeled_test_path, unlabeled_path = _create_test_data()

    cls.DATASET_CLASS.BUILDER_CONFIGS = [
        data.SkaiDatasetConfig(
            name='test_config',
            labeled_train_pattern=labeled_train_path,
            labeled_test_pattern=labeled_test_path,
            unlabeled_pattern=unlabeled_path)
    ]


def _create_test_data_with_hex_strings():
  examples_dir = _make_temp_dir()
  labeled_train_path = os.path.join(
      examples_dir, 'train_labeled_examples.tfrecord')
  labeled_test_path = os.path.join(
      examples_dir, 'test_labeled_examples.tfrecord')
  unlabeled_path = os.path.join(
      examples_dir, 'unlabeled_examples.tfrecord')

  _write_tfrecord([
      _make_example('b0b947f423a1c77ac948c76f63fa8209', 0, 0, 'A0', 0, 'no_damage', 64, 256),
      _make_example('5fb3fc48db76805c169e8dc667c3f266', 0, 1, 'A1', 0, 'no_damage', 64, 256),
      _make_example('21bdfdb3f65974473d4a19f05871449d', 0, 2, 'A2', 1, 'major_damage', 64, 256),
  ], labeled_train_path)

  _write_tfrecord([
      _make_example('a564b943bdebd4936ce0fd135cc19fbf', 1, 0, 'B0', 0, 'no_damage', 64, 256),
  ], labeled_test_path)

  _write_tfrecord([
      _make_example('3a8e68680d3ec6d1013d11f492a2d7d5', 2, 0, 'C0', -1, 'bad_example', 64, 256),
      _make_example('1004dc994ff1888052aa3ff4be5e55cf', 2, 1, 'C1', -1, 'bad_example', 64, 256),
      _make_example('4b49276f4f10856b9e8a57fad78ee593', 2, 2, 'C2', -1, 'bad_example', 64, 256),
      _make_example('97a9600f1e418132af93ea03f4264ad2', 2, 3, 'C3', -1, 'bad_example', 64, 256),
  ], unlabeled_path)

  return labeled_train_path, labeled_test_path, unlabeled_path
class TestDataEncoder(absltest.TestCase):
  def setUp(self):
    self.data_encoder = data.DataEncoder()
    labeled_train_path, labeled_test_path, unlabeled_path = _create_test_data_with_hex_strings()
    self.labeled_train_path = labeled_train_path
    self.labeled_test_path = labeled_test_path
    self.unlabeled_path = unlabeled_path

    dataset_builder = data.get_dataset('skai')
    kwargs = {
        'labeled_train_pattern': self.labeled_train_path,
        'unlabeled_train_pattern': self.unlabeled_path,
        'validation_pattern': self.labeled_test_path,
        'use_post_disaster_only': False,
        'load_small_images': True,
        'data_dir': _make_temp_dir(),
    }

    dataloader = dataset_builder(
        1,
        initial_sample_proportion=1,
        subgroup_ids=(),
        subgroup_proportions=(),
        **kwargs
    )
    self.dataloader = data.apply_batch(dataloader, 2)

  def test_encode_example_ids_returns_dataloader(self):
    # Check if encode_example_id method correctly returns a dataloader
    encoded_dataloader = self.data_encoder.encode_example_ids(self.dataloader)
    self.assertIsInstance(encoded_dataloader, data.Dataloader)

  def test_encode_example_ids_encodes_strings_to_int(self):
    # Check if the example IDs are correctly encoded to ints
    encoded_dataloader = self.data_encoder.encode_example_ids(self.dataloader)
    dataset = encoded_dataloader.train_splits[0]
    encoded_example_ids = list(dataset.map(lambda x: x['example_id']
                                           ).as_numpy_iterator())[0]
    self.assertIsInstance(encoded_example_ids, np.ndarray)
    self.assertTrue(np.issubdtype(encoded_example_ids.dtype, np.integer))

  def test_encode_string_labels_returns_dataloader(self):
    # Check if encode_string_label method correctly returns a dataloader
    encoded_dataloader = self.data_encoder.encode_string_labels(self.dataloader)
    self.assertIsInstance(encoded_dataloader, data.Dataloader)

  def test_encode_string_labels_encodes_strings_to_int(self):
    # Check if encode_string_label method correctly returns a dataloader
    encoded_dataloader = self.data_encoder.encode_string_labels(self.dataloader)
    dataset = encoded_dataloader.train_splits[0] #pick one example and evaluate
    encoded_string_label = list(dataset.map(lambda x: x['string_label']
                                           ).as_numpy_iterator())[0]
    self.assertIsInstance(encoded_string_label, np.ndarray)
    self.assertTrue(np.issubdtype(encoded_string_label.dtype, np.integer))

  def test_decode_example_ids_returns_dataloader(self):
    encoded_dataloader = self.data_encoder.encode_example_ids(self.dataloader)
    decoded_data = self.data_encoder.decode_example_ids(encoded_dataloader)
    self.assertIsInstance(decoded_data, data.Dataloader)

  def test_decode_int_label_decodes_int_to_string(self):
    # Check if the example IDs are correctly encoded
    encoded_dataloader = self.data_encoder.encode_string_labels(self.dataloader)
    decoded_dataloader = self.data_encoder.decode_string_labels(encoded_dataloader)
    dataset = decoded_dataloader.train_splits[0]
    decoded_int_label = list(dataset.map(lambda x: x['string_label']
                                           ).as_numpy_iterator())[0]
    self.assertIsInstance(decoded_int_label, np.ndarray)
    self.assertTrue(np.issubdtype(decoded_int_label.dtype, np.str_) or
                    np.issubdtype(decoded_int_label.dtype, object))

  def test_decode_example_id_outputs_matches_inputs(self):
    all_example_ids = []
    dataset_true = self.dataloader.train_splits[0]
    true_id_list = list(dataset_true.map(lambda x: x['example_id']).as_numpy_iterator())
    for string_label in true_id_list:
      all_example_ids += string_label.tolist()

    encoded_dataloader = self.data_encoder.encode_example_ids(self.dataloader)
    decoded_dataloader = self.data_encoder.decode_example_ids(encoded_dataloader)

    all_decoded_ids = []
    dataset_decoded = decoded_dataloader.train_splits[0]
    decoded_id_list = list(dataset_decoded.map(lambda x: x['example_id']).as_numpy_iterator())
    for string_label in decoded_id_list:
      all_decoded_ids += string_label.tolist()
    self.assertItemsEqual(all_example_ids[:len(all_decoded_ids)],
                    all_decoded_ids)

  def test_decode_string_label_outputs_matches_inputs(self):
    all_string_labels = []
    dataset_true = self.dataloader.train_splits[0]
    true_labels_list = list(dataset_true.map(lambda x: x['string_label']).as_numpy_iterator())
    for string_label in true_labels_list:
      all_string_labels += string_label.tolist()

    encoded_dataloader = self.data_encoder.encode_string_labels(self.dataloader)
    decoded_dataloader = self.data_encoder.decode_string_labels(encoded_dataloader)

    all_decoded_labels = []
    dataset_decoded = decoded_dataloader.train_splits[0]
    decoded_labels_list = list(dataset_decoded.map(lambda x: x['string_label']).as_numpy_iterator())
    for string_label in decoded_labels_list:
      all_decoded_labels += string_label.tolist()
    self.assertItemsEqual(all_string_labels[:len(all_decoded_labels)],
                    all_decoded_labels)


if __name__ == '__main__':
  tfds.testing.test_main()
