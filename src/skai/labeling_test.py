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
import pathlib
import random
import tempfile

from absl.testing import absltest
import numpy as np
import pandas as pd
import PIL
from skai import labeling
from skai import utils
import tensorflow as tf


Example = tf.train.Example


def _write_example_to_tfrecord(
    example_id_lon_lat: tuple[str, [float, float]],
    tfrecord_output_path: str,
    test_image_path: str,
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
      image = tf.io.encode_png(
          np.array(PIL.Image.open(test_image_path))
      ).numpy()
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
    current_dir = pathlib.Path(__file__).parent
    test_image_path = str(current_dir / 'test_data/blank.tif')
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
            test_image_path,
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
    current_dir = pathlib.Path(__file__).parent
    test_image_path = str(current_dir / 'test_data/blank.tif')
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
            test_image_path,
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

      df_labeling_images = pd.read_csv(f'{output_dir}/image_metadata.csv')
      self.assertCountEqual(
          df_labeling_images['example_id'],
          allowed_example_ids,
      )
      self.assertCountEqual(
          set(os.listdir(f'{output_dir}/')),
          set(['a.png', 'b.png', 'e.png', 'image_metadata.csv']),
      )


if __name__ == '__main__':
  absltest.main()
