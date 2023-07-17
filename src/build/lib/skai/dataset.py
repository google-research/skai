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

"""TFDS dataset for SKAI."""

import dataclasses
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class SkaiDatasetConfig(tfds.core.BuilderConfig):
  """Configuration for SKAI datasets.

  Any of the attributes can be left blank if they don't exist.

  Attributes:
    labeled_train_pattern: Pattern for labeled training examples tfrecords.
    labeled_test_pattern: Pattern for labeled test examples tfrecords.
    unlabeled_pattern: Pattern for unlabeled examples tfrecords.
  """
  labeled_train_pattern: str = ''
  labeled_test_pattern: str = ''
  unlabeled_pattern: str = ''


class SkaiDataset(tfds.core.GeneratorBasedBuilder):
  """TFDS dataset for SKAI.

  Example usage:
    import tensorflow_datasets as tfds
    from skai import dataset

    ds = tfds.load('skai_dataset', builder_kwargs={
      'config': SkaiDatasetConfig(
          name='example',
          labeled_train_pattern='gs://path/to/train_labeled_examples.tfrecord',
          labeled_test_pattern='gs://path/to/test_labeled_examples.tfrecord',
          unlabeled_pattern='gs://path/to/unlabeled_examples-*.tfrecord')
    })
    labeled_train_dataset = ds['labeled_train']
    labeled_test_dataset = ds['labeled_test']
    unlabeled_test_dataset = ds['unlabeled']
  """

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description='Skai',
        features=tfds.features.FeaturesDict({
            'coordinates':
                tfds.features.Tensor(shape=(2,), dtype=tf.float32),
            'encoded_coordinates':
                tfds.features.Tensor(shape=(), dtype=tf.string),
            'pre_image_png':
                tfds.features.Tensor(shape=(64, 64, 3), dtype=tf.uint8),
            'post_image_png':
                tfds.features.Tensor(shape=(64, 64, 3), dtype=tf.uint8),
            'label':
                tfds.features.Tensor(shape=(), dtype=tf.float32)
        }))

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    splits = {}
    if self.builder_config.labeled_train_pattern:
      splits['labeled_train'] = self._generate_examples(
          self.builder_config.labeled_train_pattern)
    if self.builder_config.labeled_test_pattern:
      splits['labeled_test'] = self._generate_examples(
          self.builder_config.labeled_test_pattern)
    if self.builder_config.unlabeled_pattern:
      splits['unlabeled'] = self._generate_examples(
          self.builder_config.unlabeled_pattern)
    return splits

  def _decode_record(self, record_bytes):
    features = tf.io.parse_single_example(
        record_bytes,
        {
            'coordinates': tf.io.FixedLenFeature([2], dtype=tf.float32),
            'encoded_coordinates': tf.io.FixedLenFeature([], dtype=tf.string),
            'pre_image_png': tf.io.FixedLenFeature([], dtype=tf.string),
            'post_image_png': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.float32),
        })
    example_id = features['encoded_coordinates']
    features['pre_image_png'] = tf.io.decode_image(features['pre_image_png'])
    features['post_image_png'] = tf.io.decode_image(features['post_image_png'])
    return example_id, features

  def _generate_examples(self, pattern: str):
    if not pattern:
      return
    paths = tf.io.gfile.glob(pattern)
    ds = tf.data.TFRecordDataset(paths).map(self._decode_record)
    return ds.as_numpy_iterator()
