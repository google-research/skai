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

"""Tests for cloud_labeling."""

import os
import pathlib
import random
import tempfile
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import geopandas as gpd
import numpy as np
import pandas as pd
import PIL
import shapely
from skai import cloud_labeling
from skai import utils
import tensorflow as tf


Example = tf.train.Example


def _read_tfrecord(path: str) -> List[Example]:
  examples = []
  for record in tf.data.TFRecordDataset([path]):
    example = Example()
    example.ParseFromString(record.numpy())
    examples.append(example)
  return examples


class CloudLabelingTest(parameterized.TestCase):

  def testCreateLabelingImageBasic(self):
    before_image = PIL.Image.new('RGB', (64, 64))
    after_image = PIL.Image.new('RGB', (64, 64))
    labeling_image = cloud_labeling.create_labeling_image(
        before_image, after_image, 'example_id', 'plus_code')
    self.assertEqual(labeling_image.width, 158)
    self.assertEqual(labeling_image.height, 116)

  def testWriteImportFile(self):
    images_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
    image_files = [
        os.path.join(images_dir, f) for f in ['a.png', 'b.png', 'c.png']
    ]
    for filename in image_files:
      open(filename, 'w').close()
    output_path = os.path.join(absltest.TEST_TMPDIR.value, 'import_file.csv')
    cloud_labeling.write_import_file(images_dir, 2, False, output_path)
    with open(output_path, 'r') as f:
      contents = [line.strip() for line in f.readlines()]
    self.assertEqual(contents, image_files[:2])

  def testWriteImportFileMaxImages(self):
    images_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
    image_files = [
        os.path.join(images_dir, f) for f in ['a.png', 'b.png', 'c.png']
    ]
    for filename in image_files:
      open(filename, 'w').close()
    output_path = os.path.join(absltest.TEST_TMPDIR.value, 'import_file.csv')
    cloud_labeling.write_import_file(images_dir, 5, False, output_path)
    with open(output_path, 'r') as f:
      contents = [line.strip() for line in f.readlines()]
    self.assertEqual(contents, image_files)

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
    cloud_labeling.create_labeled_examples(
        project=None,
        location=None,
        dataset_ids=[],
        label_file_paths=[label_file_path],
        string_to_numeric_labels=[
            'no_damage=0',
            'minor_damage=0',
            'major_damage=1',
            'destroyed=1',
            'bad_example=0',
        ],
        export_dir=None,
        examples_pattern=unlabeled_examples_path,
        test_fraction=0.333,
        train_output_path=train_path,
        test_output_path=test_path,
        connecting_distance_meters=78,
        use_multiprocessing=False)

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

  def testCreateLabeledExamplesFromLabelFileWithNoOverlap(self):
    n_test = 100
    for _ in range(n_test):
      # Create unlabeled examples.
      _, unlabeled_examples_path = tempfile.mkstemp(
          dir=absltest.TEST_TMPDIR.value
      )
      # a is connected to d within 78 metres buffer
      # c is connected to b within 78 metres buffer
      # e
      test_fraction = 0.4
      possible_test_ids = [['b', 'c'], ['c', 'b'], ['a', 'd'], ['d', 'a']]
      with tf.io.TFRecordWriter(unlabeled_examples_path) as writer:
        example_id_lon_lat = [
            ('a', [92.850449, 20.148951]),
            ('b', [92.889694, 20.157515]),
            ('c', [92.889740, 20.157454]),
            ('d', [92.850479, 20.148664]),
            ('e', [92.898537, 20.160021]),
        ]
        for i, (example_id, (lon, lat)) in enumerate(example_id_lon_lat):
          example = Example()
          example.features.feature['example_id'].bytes_list.value.append(
              example_id.encode()
          )
          example.features.feature[
              'encoded_coordinates'
          ].bytes_list.value.append(str(i).encode())
          example.features.feature['coordinates'].float_list.value.extend(
              [lon, lat]
          )

          writer.write(example.SerializeToString())

      # Create a label file.
      _, label_file_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
      label_file_contents = pd.DataFrame(
          [
              ('a', 'no_damage', [92.850449,	20.148951]),
              ('b', 'minor_damage', [92.889694,	20.157515]),
              ('c', 'major_damage', [92.889740,	20.157454]),
              ('d', 'destroyed', [92.850479,	20.148664]),
              ('e', 'bad_example', [92.898537,	20.160021]),
          ],
          columns=['example_id', 'string_label', 'coordinates'],
      )
      label_file_contents.to_csv(label_file_path, index=False)

      _, train_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
      _, test_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
      cloud_labeling.create_labeled_examples(
          project=None,
          location=None,
          dataset_ids=[],
          label_file_paths=[label_file_path],
          string_to_numeric_labels=[
              'no_damage=0',
              'minor_damage=0',
              'major_damage=1',
              'destroyed=1',
              'bad_example=0',
          ],
          export_dir=None,
          examples_pattern=unlabeled_examples_path,
          test_fraction=test_fraction,
          train_output_path=train_path,
          test_output_path=test_path,
          connecting_distance_meters=78.0,
          use_multiprocessing=False,
      )

      all_examples = _read_tfrecord(train_path) + _read_tfrecord(test_path)
      test_examples = _read_tfrecord(test_path)
      test_ids = [
          e.features.feature['example_id'].bytes_list.value[0].decode()
          for e in test_examples
      ]
      self.assertLen(all_examples, 5)
      self.assertIn(test_ids, possible_test_ids)

  @parameterized.parameters(
      dict(
          graph=[
              [0, 1, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
          ],
          correct_labels=[0, 0, 0, 1, 1, 2, 0],
      ),
      dict(
          graph=[
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
          ],
          correct_labels=[0, 1, 2, 3],
      ),
      dict(
          graph=[
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
          ],
          correct_labels=[0, 0, 0, 0, 0, 0],
      ),
  )
  def testGetConnectedLabels(self, graph, correct_labels):
    labels = cloud_labeling.get_connected_labels(np.array(graph))

    self.assertSequenceEqual(labels, correct_labels)


class GetConnectionMatrixTest(tf.test.TestCase):
  def testGetConnectionMatrix(self):
    encoded_coordinate_lon_lat = [
        ('a', [92.850449, 20.148951]),
        ('b', [92.889694, 20.157515]),
        ('c', [92.889740, 20.157454]),
        ('d', [92.850479, 20.148664]),
        ('e', [92.898537, 20.160021]),
    ]
    connecting_distance_meters = 78.0
    encoded_coordinates = []
    longitudes = []
    latitudes = []
    for example in encoded_coordinate_lon_lat:
      encoded_coordinates.append(example[0])
      longitudes.append(example[1][0])
      latitudes.append(example[1][1])

    correct_gdf = gpd.GeoDataFrame(
        data=encoded_coordinates,
        geometry=[
            shapely.geometry.Point(484370.937, 2227971.391),
            shapely.geometry.Point(488472.931, 2228915.897),
            shapely.geometry.Point(488477.734, 2228909.144),
            shapely.geometry.Point(484374.044, 2227939.628),
            shapely.geometry.Point(489397.202, 2229192.628),
        ],
        columns=['encoded_coordinates'],
    )
    correct_connection_matrix = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.int32,
    )

    gpd_df, connection_matrix = cloud_labeling.get_connection_matrix(
        longitudes, latitudes, encoded_coordinates, connecting_distance_meters
    )

    self.assertNDArrayNear(connection_matrix, correct_connection_matrix, 1e-15)
    self.assertSameElements(gpd_df, correct_gdf)


class CreateLabelingImagesTestUsingBufferedSampling(tf.test.TestCase):
  """Tests for create_labeling_images using buffered sampling."""

  def testSamplingWithBufferRadiusAndMetaDataExamplesFile(self):
    """Tests buffered sampling when metadata_examples.csv is PRESENT."""
    max_num_images = 2
    current_dir = pathlib.Path(__file__).parent
    test_image_path = str(current_dir / 'test_data/blank.tif')
    # Create 4 unlabeled examples.
    with tempfile.TemporaryDirectory() as examples_dir:
      os.mkdir(os.path.join(examples_dir, 'examples'))
      os.mkdir(os.path.join(examples_dir, 'examples', 'unlabeled'))
      os.mkdir(
          os.path.join(
              examples_dir,
              'examples',
              f'label_images_{max_num_images}',
          )
      )

      examples_pattern = os.path.join(
          examples_dir, 'examples', 'unlabeled', '*'
      )
      metadata_examples_path = os.path.join(
          examples_dir, 'examples', 'metadata_examples.csv'
      )
      output_dir = os.path.join(
          examples_dir, 'examples', f'label_images_{max_num_images}'
      )

      # a is connected to d within 78 metres
      # c is connected to b within 78 metres
      # e is not connected to any of the other points within 78 metres
      possible_example_id_combination = [
          ['a', 'b'],
          ['a', 'c'],
          ['c', 'd'],
          ['d', 'b'],
          ['e', 'a'],
          ['e', 'b'],
          ['e', 'c'],
          ['e', 'd'],
      ]

      # [image_path_x, image_path_y]
      possible_image_paths_from_create_labeling_task = [
          [f'{output_dir}/{id1}.png', f'{output_dir}/{id2}.png']
          for id1, id2 in possible_example_id_combination
      ]

      # [image_path_y, image_path_x]
      possible_image_paths_from_create_labeling_task.extend([
          [f'{output_dir}/{id2}.png', f'{output_dir}/{id1}.png']
          for id1, id2 in possible_example_id_combination
      ])
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
      with tf.io.TFRecordWriter(
          f'{examples_dir}/examples/unlabeled/unlabeled_large.tfrecord'
      ) as writer:
        example_id_lon_lat = [
            ('a', [92.850449, 20.148951]),
            ('b', [92.889694, 20.157515]),
            ('c', [92.889740, 20.157454]),
            ('d', [92.850479, 20.148664]),
            ('e', [92.898537, 20.160021]),
        ]
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

          writer.write(example.SerializeToString())

      eval_num_images, eval_import_file_paths = (
          cloud_labeling.create_labeling_images(
              examples_pattern=examples_pattern,
              max_images=max_num_images,
              allowed_example_ids_path=None,
              excluded_import_file_patterns=None,
              output_dir=output_dir,
              use_multiprocessing=False,
              buffered_sampling_radius=78.0,
          )
      )

      eval_image_paths = pd.read_csv(
          eval_import_file_paths, header=None, names=['image_path']
      )
    self.assertLen(eval_image_paths, 2)
    self.assertEqual(eval_num_images, max_num_images)
    self.assertIn(
        list(eval_image_paths.image_path),
        possible_image_paths_from_create_labeling_task,
    )

  def testSamplingWithBufferRadiusAndMetaDataExamplesFileN50(self):
    """50 random tests of buffered sampling when metadata_examples.csv is PRESENT.
    """
    for _ in range(50):
      self.testSamplingWithBufferRadiusAndMetaDataExamplesFile()

  def testSamplingWithBufferRadiusAndMissingMetaDataExamplesFile(self):
    """Tests buffered sampling when metadata_examples.csv is MISSING."""
    with tempfile.TemporaryDirectory() as examples_dir:
      os.mkdir(os.path.join(examples_dir, 'examples'))
      os.mkdir(os.path.join(examples_dir, 'examples', 'unlabeled'))
      os.mkdir(
          os.path.join(
              examples_dir,
              'examples',
              'label_images',
          )
      )
      examples_pattern = os.path.join(
          examples_dir, 'examples', 'unlabeled', '*'
      )
      output_dir = os.path.join(examples_dir, 'examples', 'label_images')
      with tf.io.TFRecordWriter(
          f'{examples_dir}/examples/unlabeled/unlabeled_large.tfrecord'
      ) as writer:
        example_ids = ['a', 'b', 'c', 'd', 'e']
        for example_id in example_ids:
          example = Example()
          utils.add_bytes_feature(
              feature_name='example_id',
              value=example_id.encode(),
              example=example,
          )
          writer.write(example.SerializeToString())

      with self.assertRaises(SystemExit):
        _, _ = (
            cloud_labeling.create_labeling_images(
                examples_pattern=examples_pattern,
                max_images=2,
                allowed_example_ids_path=None,
                excluded_import_file_patterns=None,
                output_dir=output_dir,
                use_multiprocessing=False,
                buffered_sampling_radius=78.0,
            )
        )


if __name__ == '__main__':
  absltest.main()
