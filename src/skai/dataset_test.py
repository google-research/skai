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

"""Tests for data loader."""

import os
import tempfile
from typing import List
import numpy as np
from skai import dataset
import tensorflow as tf
import tensorflow_datasets as tfds


def _make_serialized_image(size: int, pixel_value: int) -> bytes:
  image = np.ones((size, size, 3), dtype=np.uint8) * pixel_value
  return tf.io.encode_png(image).numpy()


def _make_example(
    longitude: float,
    latitude: float,
    encoded_coordinates: str,
    label: float,
    patch_size: int,
    before_pixel_value: int,
    after_pixel_value: int) -> tf.train.Example:
  example = tf.train.Example()
  example.features.feature['coordinates'].float_list.value.extend(
      (longitude, latitude))
  example.features.feature['encoded_coordinates'].bytes_list.value.append(
      encoded_coordinates.encode())
  example.features.feature['label'].float_list.value.append(label)
  example.features.feature['pre_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size, before_pixel_value))
  example.features.feature['post_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size, after_pixel_value))
  return example


def _write_tfrecord(examples: List[tf.train.Example], path: str) -> None:
  with tf.io.TFRecordWriter(path) as file_writer:
    for example in examples:
      file_writer.write(example.SerializeToString())


class SkaiDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for my_dataset dataset."""

  DATASET_CLASS = dataset.SkaiDataset
  SPLITS = {
      'labeled_train': 3,
      'labeled_test': 1,
      'unlabeled': 4
  }
  EXAMPLE_DIR = tempfile.mkdtemp()
  BUILDER_CONFIG_NAMES_TO_TEST = ['test_dataset']
  SKIP_TF1_GRAPH_MODE = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    examples_dir = tempfile.mkdtemp()
    labeled_train_path = os.path.join(
        examples_dir, 'train_labeled_examples.tfrecord')
    labeled_test_path = os.path.join(
        examples_dir, 'test_labeled_examples.tfrecord')
    unlabeled_path = os.path.join(
        examples_dir, 'unlabeled_examples.tfrecord')

    _write_tfrecord([
        _make_example(0, 0, 'A0', 0, 64, 0, 0),
        _make_example(0, 1, 'A1', 0, 64, 1, 1),
        _make_example(0, 2, 'A2', 1, 64, 2, 2),
    ], labeled_train_path)

    _write_tfrecord([
        _make_example(1, 0, 'B0', 0, 64, 3, 3),
    ], labeled_test_path)

    _write_tfrecord([
        _make_example(2, 0, 'C0', -1, 64, 4, 4),
        _make_example(2, 1, 'C1', -1, 64, 5, 5),
        _make_example(2, 2, 'C2', -1, 64, 6, 6),
        _make_example(2, 3, 'C3', -1, 64, 7, 7),
    ], unlabeled_path)

    cls.DATASET_CLASS.BUILDER_CONFIGS = [
        dataset.SkaiDatasetConfig(
            name='test_dataset',
            labeled_train_pattern=labeled_train_path,
            labeled_test_pattern=labeled_test_path,
            unlabeled_pattern=unlabeled_path)
    ]

if __name__ == '__main__':
  tfds.testing.test_main()
