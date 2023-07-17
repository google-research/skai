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
"""Tests for skai.semi_supervised.dataloader.prepare_ssl_data."""

import io
import os
import tempfile

from absl.testing import absltest
import numpy as np
import PIL.Image
from skai.semi_supervised.dataloader import prepare_ssl_data
import tensorflow.compat.v1 as tf

NUM_EXAMPLE_FILES = 10
LATITUDE, LONGITUDE = 0.1, 0.1
SHUFFLE_SEED = 0
NUM_CLASSES_FOR_MULTI_CLASS = 4


def _float_feature(value_list):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def _bytes_feature(bytes_value):
  """Returns a bytes_list."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_value]))


def _create_image_bytes():
  """Returns the bytes of a PNG image."""
  image = PIL.Image.new('RGB', (161, 161))
  buffer = io.BytesIO()
  image.save(buffer, format='PNG')
  return buffer.getvalue()


def _create_mask_bytes():
  # Creates black and white image which only has 1 channel.
  image = PIL.Image.new('L', (161, 161))
  buffer = io.BytesIO()
  image.save(buffer, format='PNG')
  return buffer.getvalue()


def _create_png_example(use_mask: bool,
                        use_pre_disaster_image: bool,
                        label: int = 0.0):
  before_png = _create_image_bytes()
  after_png = _create_image_bytes()
  features_config = {
      prepare_ssl_data.POST_IMAGE_PNG_KEY: _bytes_feature(after_png),
      'label': _float_feature([label]),
      'coordinates': _float_feature([LATITUDE, LONGITUDE]),
  }
  if use_pre_disaster_image:
    before_png = _create_image_bytes()
    features_config[prepare_ssl_data.PRE_IMAGE_PNG_KEY] = _bytes_feature(
        before_png)

    if use_mask:
      features_config[
          prepare_ssl_data.PRE_SEGMENTATION_PNG_KEY] = _bytes_feature(
              _create_mask_bytes())
      features_config[
          prepare_ssl_data.POST_SEGMENTATION_PNG_KEY] = _bytes_feature(
              _create_mask_bytes())

  example = tf.train.Example(
      features=tf.train.Features(feature=features_config))
  return example.SerializeToString()


class PrepareSslDataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp()
    # Create files for each dataset source
    for i in range(NUM_EXAMPLE_FILES):
      # One segmented dataset, one not.
      for use_mask in [True, False]:
        mask_file_suffix = 'with_masks' if use_mask else ''
        train_label_ex = os.path.join(
            self.tmp_dir,
            f'train_label_examples_{mask_file_suffix}-{i:05d}-of-{NUM_EXAMPLE_FILES:05d}.tfrecord'
        )
        train_unlabel_ex = os.path.join(
            self.tmp_dir,
            f'train_unlabel_examples_{mask_file_suffix}-{i:05d}-of-{NUM_EXAMPLE_FILES:05d}.tfrecord'
        )
        test_ex = os.path.join(
            self.tmp_dir,
            f'test_examples_{mask_file_suffix}-{i:05d}-of-{NUM_EXAMPLE_FILES:05d}.tfrecord'
        )
        # For each record file, write an example.
        for ex in [train_label_ex, train_unlabel_ex, test_ex]:
          with tf.io.TFRecordWriter(ex) as writer:
            writer.write(
                _create_png_example(
                    use_mask=use_mask,
                    use_pre_disaster_image=True,
                    label=i % 2))

      # Do the same for the datasets with only post-disaster images.
      train_label_ex_post_disaster_image_only = os.path.join(
          self.tmp_dir,
          f'train_label_examples_post_disaster_image_only-{i:05d}-of-{NUM_EXAMPLE_FILES:05d}.tfrecord'
      )
      train_unlabel_ex_post_disaster_image_only = os.path.join(
          self.tmp_dir,
          f'train_unlabel_examples_post_disaster_image_only-{i:05d}-of-{NUM_EXAMPLE_FILES:05d}.tfrecord'
      )
      test_ex_post_disaster_image_only = os.path.join(
          self.tmp_dir,
          f'test_examples_post_disaster_image_only-{i:05d}-of-{NUM_EXAMPLE_FILES:05d}.tfrecord'
      )
      for ex in [
          train_label_ex_post_disaster_image_only,
          train_unlabel_ex_post_disaster_image_only,
          test_ex_post_disaster_image_only
      ]:
        with tf.io.TFRecordWriter(ex) as writer:
          writer.write(
              _create_png_example(
                  use_mask=False,
                  use_pre_disaster_image=False,
                  label=i % 2))

      multi_class_train_label_ex = os.path.join(
          self.tmp_dir,
          f'multi_class_train_label_examples-{i:05d}-of-{NUM_EXAMPLE_FILES:05d}.tfrecord'
      )
      with tf.io.TFRecordWriter(multi_class_train_label_ex) as writer:
        writer.write(
            _create_png_example(
                use_mask=False,
                use_pre_disaster_image=True,
                label=i % NUM_CLASSES_FOR_MULTI_CLASS))

    self.train_label_ex = [
        os.path.join(self.tmp_dir, 'train_label_examples_-*.tfrecord')
    ]
    self.multi_class_train_label_ex = [
        os.path.join(self.tmp_dir,
                     'multi_class_train_label_examples-*.tfrecord')
    ]
    self.train_unlabel_ex = [
        os.path.join(self.tmp_dir, 'train_unlabel_examples_-*.tfrecord')
    ]
    self.test_ex = [os.path.join(self.tmp_dir, 'test_examples_-*.tfrecord')]

    # Set up a different dataset containing mask features.
    self.train_label_ex_masks = [
        os.path.join(self.tmp_dir, 'train_label_examples_with_masks*.tfrecord')
    ]
    self.train_unlabel_ex_masks = [
        os.path.join(self.tmp_dir,
                     'train_unlabel_examples_with_masks*.tfrecord')
    ]
    self.test_ex_masks = [
        os.path.join(self.tmp_dir, 'test_examples_with_masks*.tfrecord')
    ]

    # Set up a different dataset that uses only post-disaster images.
    self.train_label_ex_post_disaster_image_only = [
        os.path.join(self.tmp_dir,
                     'train_label_examples_post_disaster_image_only*.tfrecord')
    ]
    self.train_unlabel_ex_post_disaster_image_only = [
        os.path.join(
            self.tmp_dir,
            'train_unlabel_examples_post_disaster_image_only*.tfrecord')
    ]
    self.test_ex_post_disaster_image_only = [
        os.path.join(self.tmp_dir,
                     'test_examples_post_disaster_image_only*.tfrecord')
    ]

  def testGetExampleFiles(self):
    train_label = prepare_ssl_data.get_example_files(self.train_label_ex)
    self.assertNotEmpty(train_label)
    train_unlabel = prepare_ssl_data.get_example_files(self.train_unlabel_ex)
    self.assertNotEmpty(train_unlabel)
    test = prepare_ssl_data.get_example_files(self.test_ex)
    self.assertNotEmpty(test)

    train_label_masks = prepare_ssl_data.get_example_files(
        self.train_label_ex_masks)
    self.assertNotEmpty(train_label_masks)
    train_unlabel_masks = prepare_ssl_data.get_example_files(
        self.train_unlabel_ex_masks)
    self.assertNotEmpty(train_unlabel_masks)
    test_masks = prepare_ssl_data.get_example_files(self.test_ex_masks)
    self.assertNotEmpty(test_masks)

  def testGetNoneExampleFiles(self):
    with self.assertRaises(ValueError):
      _ = prepare_ssl_data.get_example_files([])

  def testGetNonexistentExampleFiles(self):
    with self.assertRaises(ValueError):
      _ = prepare_ssl_data.get_example_files(['nonexistent.tfrecord'])

  def testCreateDataset(self):
    with tf.Session() as sess:
      for use_mask, example_sets in zip(
          [False, True],
          [(self.train_label_ex, self.train_unlabel_ex, self.test_ex),
           (self.train_label_ex_masks, self.train_unlabel_ex_masks,
            self.test_ex_masks)]):
        ssl_data = prepare_ssl_data.create_dataset(
            name='test_dataset',
            train_label_filepatterns=example_sets[0],
            train_unlabel_filepatterns=example_sets[1],
            test_filepatterns=example_sets[2],
            num_classes=2,
            height=161,
            width=161,
            shuffle=False,
            num_labeled_examples=6,
            num_unlabeled_validation_examples=10,
            num_augmentations=2,
            do_memoize=False,
            use_mask=use_mask)

        train_labeled_examples = ssl_data.train_labeled.make_one_shot_iterator()
        example = sess.run(train_labeled_examples.get_next())
        self.assertEqual(example[prepare_ssl_data.IMAGE_KEY].shape,
                         (161, 161, 8 if use_mask else 6))

  def testCreateDatasetNoTrainingData(self):
    with self.assertRaises(ValueError):
      _ = prepare_ssl_data.create_dataset(
          name='test_dataset',
          train_label_filepatterns=[],  # empty but cannot be
          train_unlabel_filepatterns=self.train_unlabel_ex,
          test_filepatterns=self.test_ex,
          num_classes=2,
          height=161,
          width=161,
          shuffle=False,
          num_labeled_examples=6,
          num_unlabeled_validation_examples=10,
          num_augmentations=2,
          do_memoize=False)

  def testCreateDatasetEvalMode(self):
    with tf.Session() as sess:
      ssl_data = prepare_ssl_data.create_dataset(
          name='test_dataset',
          train_label_filepatterns=[],
          train_unlabel_filepatterns=[],
          test_filepatterns=self.test_ex,
          num_classes=2,
          height=161,
          width=161,
          inference_mode=True,
          shuffle=False,
          num_labeled_examples=6,
          num_unlabeled_validation_examples=10,
          num_augmentations=2,
          do_memoize=False)

      test_examples = ssl_data.test.make_one_shot_iterator()
      example = sess.run(test_examples.get_next())
      self.assertEqual(example[prepare_ssl_data.IMAGE_KEY].shape, (161, 161, 6))

  def testTakeBalanced(self):

    def input_data_generator():
      """Generates 10 images and labels for use as a dataset."""
      for i in range(10):
        yield {
            prepare_ssl_data.IMAGE_KEY: np.zeros((10, 10, 6)),
            prepare_ssl_data.LABEL_KEY: int((i < 5)),
            prepare_ssl_data.COORDS_KEY: np.zeros((2), dtype=float)
        }

    with tf.Session() as sess:
      ds = tf.data.Dataset.from_generator(
          input_data_generator,
          output_types={
              prepare_ssl_data.IMAGE_KEY: tf.float32,
              prepare_ssl_data.LABEL_KEY: tf.int64,
              prepare_ssl_data.COORDS_KEY: tf.float64
          },
          output_shapes={
              prepare_ssl_data.IMAGE_KEY: (10, 10, 6),
              prepare_ssl_data.LABEL_KEY: (),
              prepare_ssl_data.COORDS_KEY: (2)
          })
      ds = prepare_ssl_data.take_balanced(ds, 3, 3, 10)
      iterator = ds.make_one_shot_iterator()
      output_labels = []
      for _ in range(6):
        output_labels.append(
            sess.run(iterator.get_next())[prepare_ssl_data.LABEL_KEY])
      self.assertCountEqual([0, 0, 0, 1, 1, 1], output_labels)

  def testCreateDatasetWithTakeBalanced(self):
    with tf.Session() as sess:
      ssl_data = prepare_ssl_data.create_dataset(
          name='test_dataset',
          train_label_filepatterns=self.train_label_ex,
          train_unlabel_filepatterns=self.train_unlabel_ex,
          test_filepatterns=self.test_ex,
          num_classes=2,
          height=161,
          width=161,
          shuffle=False,
          num_labeled_examples=10,
          num_unlabeled_validation_examples=10,
          num_augmentations=2,
          do_memoize=False,
          num_labeled_positives=3,
          num_labeled_negatives=3)

      train_labeled_examples = ssl_data.train_labeled.make_one_shot_iterator()
      output_labels = []
      for _ in range(6):
        example = sess.run(train_labeled_examples.get_next())
        output_labels.append(example[prepare_ssl_data.LABEL_KEY])
      self.assertCountEqual(output_labels, [0, 0, 0, 1, 1, 1])

  def testCreateMultiClassDataset(self):
    with tf.Session() as sess:
      ssl_data = prepare_ssl_data.create_dataset(
          name='test_multi_class_dataset',
          train_label_filepatterns=self.multi_class_train_label_ex,
          train_unlabel_filepatterns=self.train_unlabel_ex,
          test_filepatterns=self.test_ex,
          num_classes=NUM_CLASSES_FOR_MULTI_CLASS,
          height=161,
          width=161,
          shuffle=False,
          num_labeled_examples=10,
          num_unlabeled_validation_examples=10,
          num_augmentations=2,
          do_memoize=False)

      train_labeled_examples = ssl_data.train_labeled.make_one_shot_iterator()
      output_labels = []
      for _ in range(6):
        example = sess.run(train_labeled_examples.get_next())
        output_labels.append(example[prepare_ssl_data.LABEL_KEY])
      self.assertCountEqual(output_labels, [0, 1, 2, 3, 0, 1])

  def testCreateDatasetWithPostDisasterImageOnly(self):
    with tf.Session() as sess:
      ssl_data = prepare_ssl_data.create_dataset(
          name='test_dataset_post_disaster_image_only',
          train_label_filepatterns=self.train_label_ex_post_disaster_image_only,
          train_unlabel_filepatterns=self
          .train_unlabel_ex_post_disaster_image_only,
          test_filepatterns=self.test_ex_post_disaster_image_only,
          num_classes=2,
          height=161,
          width=161,
          shuffle=False,
          num_labeled_examples=6,
          num_unlabeled_validation_examples=10,
          num_augmentations=2,
          do_memoize=False,
          use_mask=False,
          use_pre_disaster_image=False)

      train_labeled_examples = ssl_data.train_labeled.make_one_shot_iterator()
      example = sess.run(train_labeled_examples.get_next())
      self.assertEqual(example[prepare_ssl_data.IMAGE_KEY].shape,
                       (161, 161, 3))

if __name__ == '__main__':
  absltest.main()
