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

import itertools
import os
import random
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
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
    example_metadata: list[tuple[str, float, float]],
    tfrecord_output_path: str,
):
  with tf.io.TFRecordWriter(tfrecord_output_path) as writer:
    random.shuffle(example_metadata)
    for example_id, longitude, latitude in example_metadata:
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
          feature_name='coordinates',
          value=[longitude, latitude],
          example=example,
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


def _write_metadata_csv(
    inputs: list[tuple[str, float, float]],
    metadata_path: str,
):
  metadata_df = pd.DataFrame(
      inputs,
      columns=['example_id', 'longitude', 'latitude'],
  )
  metadata_df['plus_code'] = 'plus_code'
  metadata_df['encoded_coordinates'] = [
      utils.encode_coordinates(lon, lat) for _, lon, lat in inputs
  ]
  metadata_df['int64_id'] = 0
  metadata_df['image_source_path'] = 'image_source_path'
  metadata_df['pre_image_id'] = 'pre_image_path'
  metadata_df['post_image_id'] = 'post_image_path'
  metadata_df['tfrecord_source_path'] = 'tfrecord_source_path'
  metadata_df.to_csv(metadata_path, index=False)


class LabelingTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(sharded_example_metadata=True), dict(sharded_example_metadata=False)
  )
  def test_create_buffered_tfrecords(self, sharded_example_metadata: bool):
    """Tests create_buffered_tfrecords."""
    # Create 5 unlabeled examples in 3 tfrecords.
    with tempfile.TemporaryDirectory() as examples_dir:
      os.mkdir(os.path.join(examples_dir, 'metadata'))
      os.mkdir(os.path.join(examples_dir, 'examples'))
      os.mkdir(os.path.join(examples_dir, 'filtered'))
      os.mkdir(os.path.join(examples_dir, 'examples', 'unlabeled'))

      examples_pattern = os.path.join(
          examples_dir, 'examples', 'unlabeled', '*'
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
      if sharded_example_metadata:
        metadata_dir = os.path.join(examples_dir, 'examples', 'metadata')
        os.mkdir(metadata_dir)
        df_metadata.iloc[:2].to_csv(
            os.path.join(
                metadata_dir,
                'metadata.csv-00000-of-00002',
            ),
            index=False,
        )
        df_metadata.iloc[2:].to_csv(
            os.path.join(
                metadata_dir,
                'metadata.csv-00001-of-00002',
            ),
            index=False,
        )
      else:
        metadata_examples_path = os.path.join(
            examples_dir, 'examples', 'metadata_examples.csv'
        )
        df_metadata.to_csv(metadata_examples_path, index=False)

      shard_to_example_metadata = {
          '001': [('a', 92.850449, 20.148951), ('b', 92.889694, 20.157515)],
          '002': [('c', 92.889740, 20.157454)],
          '003': [
              ('d', 92.850479, 20.148664),
              ('e', 92.898537, 20.160021),
          ],
      }
      for shard, example_metadata in shard_to_example_metadata.items():
        tfrecord_output_path = f'{examples_dir}/examples/unlabeled/unlabeled_large_{shard}.tfrecord'
        _write_example_to_tfrecord(
            example_metadata,
            tfrecord_output_path,
        )
      metadata_df = pd.DataFrame(
          itertools.chain(*shard_to_example_metadata.values()),
          columns=['example_id', 'longitude', 'latitude'],
      )
      metadata_path = os.path.join(examples_dir, 'metadata', 'metadata.csv')
      metadata_df.to_csv(metadata_path, index=False)
      labeling.create_buffered_tfrecords(
          metadata_path,
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
      os.mkdir(os.path.join(examples_dir, 'metadata'))
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
      df_allowed_example_ids = pd.DataFrame(
          data=['a', 'b', 'e'],
          columns=['example_id'],
      )
      df_allowed_example_ids.to_csv(
          allowed_example_ids_path, index=False, header=False
      )

      shard_to_example_metadata = {
          '001': [('a', 92.850449, 20.148951), ('b', 92.889694, 20.157515)],
          '002': [('c', 92.889740, 20.157454)],
          '003': [
              ('d', 92.850479, 20.148664),
              ('e', 92.898537, 20.160021),
              ('a', 92.850449, 20.148951),  # This duplicate should be dropped
          ],
      }
      for shard, example_metadata in shard_to_example_metadata.items():
        tfrecord_output_path = f'{examples_dir}/examples/unlabeled/unlabeled_large_{shard}.tfrecord'
        _write_example_to_tfrecord(
            example_metadata,
            tfrecord_output_path,
        )
      metadata_path = os.path.join(examples_dir, 'metadata', 'metadata.csv')
      _write_metadata_csv(
          list(itertools.chain(*shard_to_example_metadata.values())),
          metadata_path,
      )
      labeling.create_labeling_images(
          metadata_pattern=metadata_path,
          images_dir=None,
          examples_pattern=examples_pattern,
          max_images=3,
          allowed_example_ids_path=allowed_example_ids_path,
          excluded_import_file_patterns=None,
          output_dir=output_dir,
          use_multiprocessing=False,
          multiprocessing_context=None,
          max_processes=4,
          buffered_sampling_radius=78.0,
          score_bins_to_sample_fraction=None,
          scores_path=None,
          filter_by_column=None,
      )

      self.assertCountEqual(
          os.listdir(f'{output_dir}/'),
          [
              'a.png',
              'a_pre.png',
              'a_post.png',
              'b.png',
              'b_pre.png',
              'b_post.png',
              'e.png',
              'e_pre.png',
              'e_post.png',
              'image_metadata.csv',
              'import_file.csv',
              'labeling_examples.tfrecord',
          ],
      )

      image_metadata = pd.read_csv(f'{output_dir}/image_metadata.csv')
      self.assertCountEqual(
          image_metadata['example_id'],
          ['a', 'b', 'e'],
      )
      self.assertCountEqual(
          image_metadata.columns,
          [
              'id',
              'int64_id',
              'example_id',
              'image',
              'image_source_path',
              'pre_image_path',
              'post_image_path',
              'tfrecord_source_path',
              'longitude',
              'latitude',
          ],
      )

      import_file_df = pd.read_csv(
          f'{output_dir}/import_file.csv', names=['path']
      )
      self.assertCountEqual(
          import_file_df['path'].values,
          [f'{output_dir}/a.png', f'{output_dir}/b.png', f'{output_dir}/e.png'],
      )

      ds = tf.data.TFRecordDataset([f'{output_dir}/labeling_examples.tfrecord'])
      num_examples = sum([1 for _ in ds])
      self.assertEqual(num_examples, 3)

  def test_create_labeling_images_from_images_dir(self):
    """Tests create_labeling_images using images_dir."""
    with tempfile.TemporaryDirectory() as examples_dir:
      os.mkdir(os.path.join(examples_dir, 'metadata'))
      os.mkdir(os.path.join(examples_dir, 'images'))
      os.mkdir(os.path.join(examples_dir, 'images', 'pre'))
      os.mkdir(os.path.join(examples_dir, 'images', 'post'))
      os.mkdir(os.path.join(examples_dir, 'images', 'large_pre'))
      os.mkdir(os.path.join(examples_dir, 'images', 'large_post'))
      os.mkdir(os.path.join(examples_dir, 'labeling_examples'))

      image_bytes = tf.image.encode_png(
          np.zeros((256, 256, 3), dtype=np.uint8)
      ).numpy()
      for example_id in ['a', 'b', 'c']:
        for image_subdir in ['pre', 'post', 'large_pre', 'large_post']:
          with open(
              os.path.join(
                  examples_dir, 'images', image_subdir, f'{example_id}.png'
              ),
              'wb',
          ) as f:
            f.write(image_bytes)

      output_dir = os.path.join(
          examples_dir, 'labeling_examples',
      )
      metadata_path = os.path.join(examples_dir, 'metadata', 'metadata.csv')
      _write_metadata_csv(
          [
              ('a', 90, 20),
              ('b', 91, 21),
              ('c', 92, 22),
          ],
          metadata_path,
      )
      labeling.create_labeling_images(
          metadata_pattern=metadata_path,
          images_dir=os.path.join(examples_dir, 'images'),
          examples_pattern=None,
          max_images=3,
          allowed_example_ids_path=None,
          excluded_import_file_patterns=None,
          output_dir=output_dir,
          use_multiprocessing=False,
          multiprocessing_context=None,
          max_processes=1,
          buffered_sampling_radius=78.0,
          score_bins_to_sample_fraction=None,
          scores_path=None,
          filter_by_column=None,
      )

      self.assertCountEqual(
          os.listdir(f'{output_dir}/'),
          [
              'a.png',
              'a_pre.png',
              'a_post.png',
              'b.png',
              'b_pre.png',
              'b_post.png',
              'c.png',
              'c_pre.png',
              'c_post.png',
              'image_metadata.csv',
              'import_file.csv',
              'labeling_examples.tfrecord',
          ],
      )

      image_metadata = pd.read_csv(f'{output_dir}/image_metadata.csv')
      self.assertCountEqual(
          image_metadata['example_id'],
          ['a', 'b', 'c'],
      )
      self.assertCountEqual(
          image_metadata.columns,
          [
              'id',
              'int64_id',
              'example_id',
              'image',
              'image_source_path',
              'pre_image_path',
              'post_image_path',
              'tfrecord_source_path',
              'longitude',
              'latitude',
          ],
      )

      import_file_df = pd.read_csv(
          f'{output_dir}/import_file.csv', names=['path']
      )
      self.assertCountEqual(
          import_file_df['path'].values,
          [f'{output_dir}/a.png', f'{output_dir}/b.png', f'{output_dir}/c.png'],
      )

      ds = tf.data.TFRecordDataset([f'{output_dir}/labeling_examples.tfrecord'])
      num_examples = sum([1 for _ in ds])
      self.assertEqual(num_examples, 3)

  def test_create_labeled_examples_from_label_file(self):
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

  def test_create_labeled_examples_from_labeled_examples(self):
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

  def test_get_diffuse_subset(self):
    points = [
        shapely.geometry.Point(-61.458391, 12.470539),
        shapely.geometry.Point(-61.458325, 12.470466),
        shapely.geometry.Point(-61.457885, 12.470580),
        shapely.geometry.Point(-61.457784, 12.470587),
        shapely.geometry.Point(-61.458100, 12.470049),
    ]
    example_ids = ['a', 'b', 'c', 'd', 'e']
    points_gdf = utils.convert_to_utm(
        gpd.GeoDataFrame({'example_id': example_ids}, geometry=points, crs=4326)
    )
    subset_gdf = labeling.get_diffuse_subset(points_gdf, 50)
    self.assertSameElements(['a', 'c', 'e'], subset_gdf['example_id'].values)

  def test_create_labeling_assets_from_metadata(self):
    with tempfile.TemporaryDirectory(
        dir=absltest.TEST_TMPDIR.value
    ) as temp_dir:
      metadata_path = os.path.join(temp_dir, 'metadata.csv')
      metadata_df = pd.DataFrame({
          'example_id': ['example_id'],
          'int64_id': [123],
          'encoded_coordinates': [utils.encode_coordinates(23, 45)],
          'pre_image_id': ['pre_image_id'],
          'post_image_id': ['post_image_id'],
          'longitude': [23],
          'latitude': [45],
          'label': [0.0],
          'string_label': ['string_label'],
          'plus_code': ['plus_code'],
      })
      metadata_df.to_csv(metadata_path, index=False)

      images_dir = os.path.join(temp_dir, 'images')
      os.mkdir(images_dir)
      os.mkdir(os.path.join(images_dir, 'pre'))
      os.mkdir(os.path.join(images_dir, 'post'))
      os.mkdir(os.path.join(images_dir, 'large_pre'))
      os.mkdir(os.path.join(images_dir, 'large_post'))
      image = np.zeros((256, 256, 3), dtype=np.uint8)
      image_bytes = tf.image.encode_png(image).numpy()
      for path in [
          os.path.join(images_dir, 'pre', 'example_id.png'),
          os.path.join(images_dir, 'post', 'example_id.png'),
          os.path.join(images_dir, 'large_pre', 'example_id.png'),
          os.path.join(images_dir, 'large_post', 'example_id.png'),
      ]:
        with open(path, 'wb') as f:
          f.write(image_bytes)

      output_dir = os.path.join(temp_dir, 'output')
      os.mkdir(output_dir)
      labeling_examples = labeling._create_labeling_assets_from_metadata(
          metadata_path,
          images_dir,
          output_dir,
          ['example_id'],
      )
      self.assertCountEqual(
          os.listdir(output_dir),
          [
              'example_id.png',
              'example_id_pre.png',
              'example_id_post.png',
          ],
      )
      self.assertLen(labeling_examples, 1)
      self.assertEqual(labeling_examples[0].example_id, 'example_id')
      self.assertEqual(labeling_examples[0].int64_id, 123)
      self.assertEqual(
          labeling_examples[0].pre_image_path,
          os.path.join(output_dir, 'example_id_pre.png'),
      )
      self.assertEqual(
          labeling_examples[0].post_image_path,
          os.path.join(output_dir, 'example_id_post.png'),
      )
      self.assertEqual(
          labeling_examples[0].combined_image_path,
          os.path.join(output_dir, 'example_id.png'),
      )
      self.assertEqual(labeling_examples[0].longitude, 23)
      self.assertEqual(labeling_examples[0].latitude, 45)


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  absltest.main()
