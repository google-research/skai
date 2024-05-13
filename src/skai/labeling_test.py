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

"""Tests for labeling."""

import os
import random
import tempfile

from absl.testing import absltest
import numpy as np
import pandas as pd
from skai import labeling
from skai import utils
import tensorflow as tf


Example = tf.train.Example


def _read_tfrecord(path: str) -> list[Example]:
  examples = []
  for record in tf.data.TFRecordDataset([path]):
    example = Example()
    example.ParseFromString(record.numpy())
    examples.append(example)
  return examples


def _write_example_to_tfrecord(
    example_id_lon_lat: tuple[str, [float, float]],
    tfrecord_output_path: str,
):
  with tf.io.TFRecordWriter(tfrecord_output_path) as writer:
    random.shuffle(example_id_lon_lat)
    for _, (example_id, (lon, lat)) in enumerate(example_id_lon_lat):
      example = Example()
      utils.add_bytes_feature(
          feature_name='example_id',
          value=example_id.encode(),
          example=example,
      )
      image = tf.io.encode_png(np.zeros((256, 256, 3), dtype=np.uint8)).numpy()
      utils.add_bytes_feature(
          feature_name='pre_image_png_large', value=image, example=example
      )
      utils.add_bytes_feature(
          feature_name='post_image_png_large', value=image, example=example
      )
      utils.add_float_list_feature(
          feature_name='coordinates', value=[lon, lat], example=example
      )
      utils.add_int64_feature(
          feature_name='int64_id', value=ord(example_id), example=example
      )

      writer.write(example.SerializeToString())


def _read_tfrecords(path: str) -> list[Example]:
  paths = [f'{path}/{tfrecord}' for tfrecord in os.listdir(f'{path}/')]
  examples = []
  for record in tf.data.TFRecordDataset([paths]):
    example = Example()
    example.ParseFromString(record.numpy())
    examples.append(example)
  return examples


class LabelingTest(absltest.TestCase):

  def test_create_buffered_tfrecords(self):
    """Tests create_buffered_tfrecords."""
    # Create 5 unlabeled examples in 3 tfrecords.
    with tempfile.TemporaryDirectory() as examples_dir:
      os.mkdir(os.path.join(examples_dir, 'examples'))
      os.mkdir(os.path.join(examples_dir, 'filtered'))
      os.mkdir(os.path.join(examples_dir, 'examples', 'unlabeled'))

      examples_pattern = os.path.join(
          examples_dir, 'examples', 'unlabeled', '*'
      )
      metadata_examples_path = os.path.join(
          examples_dir, 'examples', 'metadata_examples.csv'
      )
      filtered_tfrecords_output_dir = os.path.join(
          examples_dir, 'filtered',
      )

      # a is connected to d within 78 metres
      # c is connected to b within 78 metres
      # e is not connected to any of the other points within 78 metres
      df_metadata = pd.DataFrame(
          data=[
              ['a', 92.850449, 20.148951],
              ['b', 92.889694, 20.157515],
              ['c', 92.889740, 20.157454],
              ['d', 92.850479, 20.148664],
              ['e', 92.898537, 20.160021],
          ],
          columns=['example_id', 'longitude', 'latitude'],
      )
      df_metadata = df_metadata.sample(frac=1)
      df_metadata.to_csv(metadata_examples_path, index=False)

      example_id_lon_lat_create_tfrecords = {
          '001': [('a', [92.850449, 20.148951]), ('b', [92.889694, 20.157515])],
          '002': [('c', [92.889740, 20.157454])],
          '003': [
              ('d', [92.850479, 20.148664]),
              ('e', [92.898537, 20.160021]),
          ],
      }
      for (
          tfrecord_count,
          example_id_lon_lat,
      ) in example_id_lon_lat_create_tfrecords.items():
        tfrecord_output_path = (
            f'{examples_dir}/examples/unlabeled/'
            + f'unlabeled_large_{tfrecord_count}.tfrecord'
        )
        _write_example_to_tfrecord(
            example_id_lon_lat,
            tfrecord_output_path,
        )

      labeling.create_buffered_tfrecords(
          examples_pattern,
          filtered_tfrecords_output_dir,
          False,
          None,
          [],
          3,
          78.0,
      )

      filtered_examples = _read_tfrecords(filtered_tfrecords_output_dir)
      filtered_example_ids = []
      for e in filtered_examples:
        filtered_example_ids.append((
            utils.get_bytes_feature(
                feature_name='example_id', example=e
            )[0].decode()
        ))

      self.assertIn(set(filtered_example_ids), [
          set(['a', 'b', 'e']),
          set(['a', 'c', 'e']),
          set(['c', 'd', 'e']),
          set(['d', 'b', 'e']),
      ])

  def test_create_labeling_images(self):
    """Tests create_labeling_images."""
    # Create 5 unlabeled examples in 3 tfrecords.
    with tempfile.TemporaryDirectory() as examples_dir:
      os.mkdir(os.path.join(examples_dir, 'examples'))
      os.mkdir(os.path.join(examples_dir, 'examples', 'unlabeled'))
      os.mkdir(os.path.join(examples_dir, 'labeling_examples'))

      examples_pattern = os.path.join(
          examples_dir, 'examples', 'unlabeled', '*'
      )
      output_dir = os.path.join(
          examples_dir, 'labeling_examples',
      )
      allowed_example_ids_path = os.path.join(
          examples_dir, 'examples', 'allowed_example_ids.csv'
      )

      # a is connected to d within 78 metres
      # c is connected to b within 78 metres
      # e is not connected to any of the other points within 78 metres
      allowed_example_ids = ['a', 'b', 'e']
      df_allowed_example_ids = pd.DataFrame(
          data=['a', 'b', 'e'],
          columns=['example_id'],
      )
      df_allowed_example_ids.to_csv(
          allowed_example_ids_path, index=False, header=False
      )

      example_id_lon_lat_create_tfrecords = {
          '001': [('a', [92.850449, 20.148951]), ('b', [92.889694, 20.157515])],
          '002': [('c', [92.889740, 20.157454])],
          '003': [
              ('d', [92.850479, 20.148664]),
              ('e', [92.898537, 20.160021]),
          ],
      }
      for (
          tfrecord_count,
          example_id_lon_lat,
      ) in example_id_lon_lat_create_tfrecords.items():
        tfrecord_output_path = f'{examples_dir}/examples/unlabeled/unlabeled_large_{tfrecord_count}.tfrecord'
        _write_example_to_tfrecord(
            example_id_lon_lat,
            tfrecord_output_path,
        )

      labeling.create_labeling_images(
          examples_pattern,
          3,
          allowed_example_ids_path,
          None,
          output_dir,
          False,
          None,
          4,
          78.0,
      )

      self.assertCountEqual(
          os.listdir(f'{output_dir}/'),
          [
              'a.png',
              'b.png',
              'e.png',
              'image_metadata.csv',
              'import_file.csv',
          ],
      )

      image_metadata = pd.read_csv(f'{output_dir}/image_metadata.csv')
      self.assertCountEqual(
          image_metadata['example_id'],
          allowed_example_ids,
      )

      import_file_df = pd.read_csv(
          f'{output_dir}/import_file.csv', names=['path']
      )
      self.assertCountEqual(
          import_file_df['path'].values,
          [f'{output_dir}/a.png', f'{output_dir}/b.png', f'{output_dir}/e.png'],
      )

  def testCreateLabeledExamplesFromLabelFile(self):
    # Create unlabeled examples.
    _, unlabeled_examples_path = tempfile.mkstemp(
        dir=absltest.TEST_TMPDIR.value)
    with tf.io.TFRecordWriter(unlabeled_examples_path) as writer:
      for i, example_id in enumerate(['a', 'b', 'c', 'd']):
        example = Example()
        example.features.feature['example_id'].bytes_list.value.append(
            example_id.encode()
        )
        example.features.feature['encoded_coordinates'].bytes_list.value.append(
            str(i).encode()
        )
        example.features.feature['coordinates'].float_list.value.extend([i, i])
        writer.write(example.SerializeToString())

    # Create a label file.
    _, label_file_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
    label_file_contents = pd.DataFrame(
        [
            ('a', 'no_damage', [0, 0]),
            ('b', 'minor_damage', [1, 1]),
            ('c', 'major_damage', [2, 2]),
            ('c', 'no_damage', [2, 2]),
            ('d', 'destroyed', [3, 3]),
            ('d', 'bad_example', [3, 3]),
        ],
        columns=['example_id', 'string_label', 'coordinates'],
    )
    label_file_contents.to_csv(label_file_path, index=False)

    _, train_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
    _, test_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)

    labeling.create_labeled_examples(
        label_file_paths=[label_file_path],
        string_to_numeric_labels=[
            'no_damage=0',
            'minor_damage=0',
            'major_damage=1',
            'destroyed=1',
            'bad_example=0',
        ],
        example_patterns=[unlabeled_examples_path],
        test_fraction=0.333,
        train_output_path=train_path,
        test_output_path=test_path,
        connecting_distance_meters=78,
        use_multiprocessing=False,
        multiprocessing_context=None,
        max_processes=1,
    )

    all_examples = _read_tfrecord(train_path) + _read_tfrecord(test_path)
    self.assertLen(all_examples, 6)

    id_to_float_label = []
    for e in all_examples:
      id_to_float_label.append((
          e.features.feature['example_id'].bytes_list.value[0].decode(),
          e.features.feature['label'].float_list.value[0],
          list(e.features.feature['coordinates'].float_list.value),
      ))

    self.assertSameElements(
        id_to_float_label,
        [
            ('a', 0.0, [0, 0]),
            ('b', 0.0, [1, 1]),
            ('c', 1.0, [2, 2]),
            ('c', 0.0, [2, 2]),
            ('d', 0.0, [3, 3]),
            ('d', 1.0, [3, 3]),
        ],
    )

  def testCreateLabeledExamplesFromLabeledExamples(self):
    # Create labeled examples.
    _, examples_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
    with tf.io.TFRecordWriter(examples_path) as writer:
      for i, example_id in enumerate(['a', 'b', 'c', 'd']):
        example = Example()
        example.features.feature['example_id'].bytes_list.value.append(
            example_id.encode()
        )
        example.features.feature['encoded_coordinates'].bytes_list.value.append(
            str(i).encode()
        )
        example.features.feature['coordinates'].float_list.value.extend([i, i])
        if example_id in ['a', 'b']:
          example.features.feature['string_label'].bytes_list.value.append(
              b'no_damage'
          )
          example.features.feature['label'].float_list.value.append(0)
        else:
          example.features.feature['string_label'].bytes_list.value.append(
              b'destroyed'
          )
          example.features.feature['label'].float_list.value.append(1)
        writer.write(example.SerializeToString())

    _, train_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
    _, test_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)

    labeling.create_labeled_examples(
        label_file_paths=[],
        string_to_numeric_labels=[],
        example_patterns=[examples_path],
        test_fraction=0.25,
        train_output_path=train_path,
        test_output_path=test_path,
        connecting_distance_meters=78,
        use_multiprocessing=False,
        multiprocessing_context=None,
        max_processes=1,
    )

    train_examples = _read_tfrecord(train_path)
    self.assertLen(train_examples, 3)
    test_examples = _read_tfrecord(test_path)
    self.assertLen(test_examples, 1)
    all_examples = train_examples + test_examples
    self.assertLen(all_examples, 4)

    id_to_float_label = []
    for e in all_examples:
      id_to_float_label.append((
          e.features.feature['example_id'].bytes_list.value[0].decode(),
          e.features.feature['label'].float_list.value[0],
          list(e.features.feature['coordinates'].float_list.value),
      ))

    self.assertSameElements(
        id_to_float_label,
        [
            ('a', 0.0, [0, 0]),
            ('b', 0.0, [1, 1]),
            ('c', 1.0, [2, 2]),
            ('d', 1.0, [3, 3]),
        ],
    )

if __name__ == '__main__':
  absltest.main()
