# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for inference_lib."""

import glob
import os
import tempfile

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
import fiona
import geopandas as gp
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.wkb
from skai import utils
from skai.model import inference_lib
import tensorflow as tf


def _make_temp_dir() -> str:
  return tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))


def _create_test_model(model_path: str, image_size: int):
  small_input = tf.keras.layers.Input(
      shape=(image_size, image_size, 6), name='small_image'
  )
  large_input = tf.keras.layers.Input(
      shape=(image_size, image_size, 6), name='large_image'
  )
  merge = tf.keras.layers.Concatenate(axis=3)([small_input, large_input])
  flat = tf.keras.layers.Flatten()(merge)
  main_out = tf.keras.layers.Dense(2, name='main')(flat)
  bias_out = tf.keras.layers.Dense(2, name='bias')(flat)
  model = tf.keras.models.Model(
      inputs={'small_image': small_input, 'large_image': large_input},
      outputs={'main': main_out, 'bias': bias_out},
  )
  tf.saved_model.save(model, model_path)


def _create_test_text_tower_model(
    model_path: str, vocab_size: int, output_dim: int
):
  class _TestTextTowerModel(tf.Module):

    def __init__(self, vocab_size, dim):
      self.vec = tf.keras.layers.TextVectorization(
          max_tokens=vocab_size, output_mode='int', output_sequence_length=dim
      )
      self.vec.adapt(tf.convert_to_tensor(['test'], tf.string))
      self.emb = tf.keras.layers.Embedding(input_dim=dim, output_dim=100)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string)]
    )
    def serving_fn(self, inputs):
      tokenized_inputs = self.vec(inputs)
      embeddings = self.emb(tokenized_inputs)
      return tf.reduce_mean(embeddings, axis=-1)

  model = _TestTextTowerModel(vocab_size=vocab_size, dim=output_dim)
  tf.saved_model.save(
      model, model_path
  )


def _create_embedding_test_model(model_path: str, image_size: int):
  small_input = tf.keras.layers.Input(
      shape=(image_size, image_size, 6), name='small_image'
  )
  large_input = tf.keras.layers.Input(
      shape=(image_size, image_size, 6), name='large_image'
  )
  merge = tf.keras.layers.Concatenate(axis=3)([small_input, large_input])
  flat = tf.keras.layers.Flatten()(merge)
  main_out = tf.keras.layers.Dense(64, name='main')(flat)
  model = tf.keras.models.Model(
      inputs={'small_image': small_input, 'large_image': large_input},
      outputs={'main': tf.transpose(
          tf.convert_to_tensor([main_out, main_out]), perm=[1, 0, 2]
      )}
    )
  tf.saved_model.save(model, model_path)


def _create_example_id_embedding(example_id: int) -> tuple[int, np.ndarray]:
  return example_id, np.array([example_id] * 64)


def _create_test_example(
    image_size: int,
    include_small_images: bool,
    include_score: bool,
    score: float = 0.0,
    include_embedding=False,
) -> tf.train.Example:
  example = tf.train.Example()
  image_bytes = tf.image.encode_png(
      np.zeros((image_size, image_size, 3), dtype=np.uint8)
  ).numpy()
  utils.add_bytes_feature('pre_image_png_large', image_bytes, example)
  utils.add_bytes_feature('post_image_png_large', image_bytes, example)
  if include_small_images:
    utils.add_bytes_feature('pre_image_png', image_bytes, example)
    utils.add_bytes_feature('post_image_png', image_bytes, example)
  utils.add_int64_feature('int64_id', 0, example)
  utils.add_float_list_feature('coordinates', [0.0, 0.0], example)
  utils.add_float_list_feature('area_in_meters', [12.0], example)
  utils.add_bytes_list_feature('plus_code', [b'abcdef'], example)
  utils.add_bytes_list_feature('encoded_coordinates', [b'beefdead'], example)
  if include_score:
    utils.add_float_list_feature('score', [score], example)
  if include_embedding:
    embedding = [0] * 64
    utils.add_float_list_feature('embedding', embedding, example)

  footprint_wkb = shapely.wkb.dumps(shapely.geometry.Point(12, 15))
  utils.add_bytes_list_feature('footprint_wkb', [footprint_wkb], example)
  return example


class TestModel(inference_lib.InferenceModel):

  def __init__(self, expected_batch_size: int, id_to_score: dict[int, float]):
    self._expected_batch_size = expected_batch_size
    self._id_to_score = id_to_score
    self._model_prepared = False

  def prepare_model(self):
    self._model_prepared = True

  def predict_scores(self, batch: list[tf.train.Example]) -> np.ndarray:
    if not self._model_prepared:
      raise ValueError('Model not prepared.')
    if not isinstance(batch, list):
      raise ValueError(
          f'Expecting batch to be a list of examples, got {type(batch)}'
      )
    if len(batch) > self._expected_batch_size:
      raise ValueError(
          f'Expected batch size is at most {self._expected_batch_size}, got'
          f' {len(batch)}.'
      )
    return np.array([
        self._id_to_score[utils.get_int64_feature(e, 'int64_id')[0]]
        for e in batch
    ])


class TestEmbeddingGeneration(absltest.TestCase):
  def test_embedding_generation(self):
    model_path = os.path.join(_make_temp_dir(), '_model.keras')
    _create_embedding_test_model(model_path, 288)
    model_ = inference_lib.TF2InferenceModel(
        model_path, 288, False, [], inference_lib.ModelType.CLASSIFICATION
    )
    model_.prepare_model()

    examples = [_create_test_example(288, False, False) for _ in range(3)]
    output_examples = model_.predict_scores(examples)
    self.assertEqual(output_examples.shape, (3, 64))


class InferenceTest(absltest.TestCase):

  def test_do_batch_divisible_array(self):
    input_list = ['dummy input'] * 32
    result = inference_lib._do_batch(input_list, 8)
    for batch in result:
      self.assertLen(batch, 8)

  def test_do_batch_non_divisible_array(self):
    input_list = ['dummy input'] * 31
    result = inference_lib._do_batch(input_list, 8)
    for batch in result[:-1]:
      self.assertLen(batch, 8)

    final_batch = result[-1]
    self.assertLen(final_batch, 7)

  def test_do_batch_small_array(self):
    input_list = ['dummy input'] * 7
    result = inference_lib._do_batch(input_list, 8)
    self.assertLen(result, 1)
    only_batch = result[0]
    self.assertLen(only_batch, 7)

  def test_csv_output(self):
    examples = [_create_test_example(224, False, True) for _ in range(7)]
    output_path = os.path.join(_make_temp_dir(), 'inference.csv')
    temp_prefix = f'{output_path}.tmp/output'
    with test_pipeline.TestPipeline() as pipeline:
      examples_collection = pipeline | beam.Create(examples)
      inference_lib.examples_to_csv(
          examples_collection, 0.5, 0.5, 0.5, temp_prefix
      )
    self.assertNotEmpty(glob.glob(f'{temp_prefix}*'))
    inference_lib.postprocess(temp_prefix, output_path, 'score')
    self.assertTrue(os.path.exists(output_path))
    self.assertEmpty(glob.glob(f'{temp_prefix}*'))
    df = pd.read_csv(output_path)
    self.assertSameElements(
        df.columns,
        [
            'example_id',
            'building_id',
            'longitude',
            'latitude',
            'score',
            'plus_code',
            'area_in_meters',
            'footprint_wkt',
            'damaged',
            'damaged_high_precision',
            'damaged_high_recall',
            'label'
        ],
    )
    self.assertLen(df, 7)

  def test_example_embedding_csv_output(self):
    example_ids_embeddings = [
        _create_example_id_embedding(i) for i in range(25)
    ]
    output_path = os.path.join(_make_temp_dir(), 'embedding.csv')
    temp_prefix = f'{output_path}.tmp/output'
    with test_pipeline.TestPipeline() as pipeline:
      example_ids_embeddings_collection = pipeline | beam.Create(
          example_ids_embeddings
      )
      inference_lib.embeddings_examples_to_csv(
          example_ids_embeddings_collection, temp_prefix
      )
    self.assertNotEmpty(glob.glob(f'{temp_prefix}*'))
    inference_lib.postprocess(temp_prefix, output_path, 'embedding')
    self.assertTrue(os.path.exists(output_path))
    self.assertEmpty(glob.glob(f'{temp_prefix}*'))
    df = pd.read_csv(output_path)
    expected_columns = ['example_id']
    expected_columns.extend([f'embedding_{i}' for i in range(64)])
    self.assertSameElements(
        df.columns,
        expected_columns,
    )
    self.assertLen(df, 25)

  def test_run_text_tower_inference(self):
    positive_labels = ['positive']*100
    negative_labels = ['negative']*100
    model_path = os.path.join(_make_temp_dir(), 'model.keras')
    positive_embedding_path = os.path.join(_make_temp_dir(), 'positive.npy')
    negative_embedding_path = os.path.join(_make_temp_dir(), 'negative.npy')
    _create_test_text_tower_model(model_path, 100, 1024)
    with test_pipeline.TestPipeline() as pipeline:
      inference_lib._run_text_tower_inference(
          pipeline,
          positive_labels,
          negative_labels,
          model_path,
          positive_embedding_path,
          negative_embedding_path
      )

    def _test_embedding_shape(path, expected_shape):
      with tf.io.gfile.GFile(path, 'rb') as f:
        label_embedding = np.load(f)
        self.assertEqual(label_embedding.shape, expected_shape)

    _test_embedding_shape(positive_embedding_path, (1024,))
    _test_embedding_shape(negative_embedding_path, (1024,))

  def test_run_inference(self):
    with test_pipeline.TestPipeline() as pipeline:
      examples = []
      id_to_score = {}
      for example_id in range(10):
        example = tf.train.Example()
        utils.add_int64_feature('int64_id', example_id, example)
        utils.add_bytes_feature(
            'encoded_coordinates', f'{example_id}'.encode(), example
        )
        examples.append(example)
        id_to_score[example_id] = 1 / (example_id + 1)

      examples_collection = pipeline | beam.Create(examples)
      batch_size = 4
      model = TestModel(batch_size, id_to_score)
      result = inference_lib.run_inference(
          examples_collection, 'score', batch_size, model
      )

      def _check_examples(examples):
        assert (
            len(examples) == 10
        ), f'Expected 10 examples in output, got {len(examples)}'
        for example in examples:
          example_id = utils.get_int64_feature(example, 'int64_id')[0]
          expected_score = id_to_score[example_id]
          score = example.features.feature['score'].float_list.value[0]
          assert np.isclose(
              score, expected_score
          ), f'Expected score = {expected_score}, got {score}.'

      util.assert_that(result, _check_examples)

  def test_run_inference_with_duplicates(self):
    coords_and_scores = [
        ('A', 0.1),
        ('A', 0.2),
        ('A', 0.3),
        ('B', 0.0),
        ('B', 1.0),
        ('C', 0.75),
    ]
    with test_pipeline.TestPipeline() as pipeline:
      examples = []
      id_to_score = {}
      for i, (coord, score) in enumerate(coords_and_scores):
        example = tf.train.Example()
        utils.add_int64_feature('int64_id', i, example)
        utils.add_bytes_feature('encoded_coordinates', coord.encode(), example)
        examples.append(example)
        id_to_score[i] = score

      examples_collection = pipeline | beam.Create(examples)
      batch_size = 4
      model = TestModel(batch_size, id_to_score)
      result = inference_lib.run_inference(
          examples_collection, 'score', batch_size, model
      )

      def _check_examples(examples):
        assert (
            len(examples) == 3
        ), f'Expected 3 examples in output, got {len(examples)}'
        for example in examples:
          coord = utils.get_bytes_feature(example, 'encoded_coordinates')[0]
          expected_score = {
              'A': 0.2,
              'B': 0.5,
              'C': 0.75,
          }[coord.decode()]
          score = example.features.feature['score'].float_list.value[0]
          assert np.isclose(
              score, expected_score
          ), f'Expected score = {expected_score}, got {score}.'

      util.assert_that(result, _check_examples)

  def test_tf2_model_prediction(self):
    model_path = os.path.join(_make_temp_dir(), 'model.keras')
    _create_test_model(model_path, 224)
    model = inference_lib.TF2InferenceModel(
        model_path, 224, False, [], inference_lib.ModelType.CLASSIFICATION
    )
    model.prepare_model()

    examples = [_create_test_example(224, True, False) for _ in range(3)]
    output_examples = model.predict_scores(examples)
    self.assertEqual(output_examples.shape, (3,))

  def test_tf2_model_prediction_no_small_images(self):
    model_path = os.path.join(_make_temp_dir(), 'model.keras')
    _create_test_model(model_path, 224)
    model = inference_lib.TF2InferenceModel(
        model_path, 224, False, [], inference_lib.ModelType.CLASSIFICATION
    )
    model.prepare_model()

    examples = [_create_test_example(224, False, False) for _ in range(3)]
    output_examples = model.predict_scores(examples)
    self.assertEqual(output_examples.shape, (3,))

  def test_example_to_row(self):
    example = _create_test_example(224, False, True, 0.7)
    row = inference_lib.example_to_row(example, 0.5, 0.75, 0.25)
    self.assertTrue(row.damaged)
    self.assertFalse(row.damaged_high_precision)
    self.assertTrue(row.damaged_high_recall)

    low_score_example = _create_test_example(224, False, True, 0.1)
    row = inference_lib.example_to_row(low_score_example, 0.5, 0.75, 0.25)
    self.assertFalse(row.damaged)
    self.assertFalse(row.damaged_high_precision)
    self.assertFalse(row.damaged_high_recall)

  def test_postprocess(self):
    output_path = os.path.join(_make_temp_dir(), 'file_to_postprocess.csv')
    output_dictionary = {
        'shard_1_output': pd.DataFrame.from_dict(
            data={
                'example_id': [1, 2],
                'building_id': ['a', 'b'],
                'longitude': [3.0, 4.0],
                'latitude': [5.0, 6.0],
                'score': [0.001, 0.002],
                'plus_code': ['aa', 'bb'],
                'area_in_meters': [np.nan, np.nan],
                'footprint_wkt': [np.nan, np.nan],
                'damaged': [False, True],
                'damaged_high_precision': [False, True],
                'damaged_high_recall': [False, True],
            }
        ),
        'shard_2_output': pd.DataFrame.from_dict({
            'example_id': [3, 4],
            'building_id': ['c', 'd'],
            'longitude': [7.0, 8.0],
            'latitude': [9.0, 10.0],
            'score': [0.003, 0.004],
            'plus_code': ['cc', 'dd'],
            'area_in_meters': [20.0, np.nan],
            'footprint_wkt': [
                shapely.geometry.Polygon(
                    ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0))
                ),
                np.nan,
            ],
            'damaged': [False, True],
            'damaged_high_precision': [False, True],
            'damaged_high_recall': [False, True],
        }),
        'shard_3_output': pd.DataFrame.from_dict({
            'example_id': [5, 6],
            'building_id': ['e', 'f'],
            'longitude': [11.0, 12.0],
            'latitude': [13.0, 14.0],
            'score': [0.05, 0.06],
            'plus_code': ['ee', 'ff'],
            'area_in_meters': [20.0, 30.0],
            'footprint_wkt': [
                shapely.geometry.Polygon(
                    ((0, 0), (10, 0), (10, 10), (0, 10), (0, 0))
                ),
                shapely.geometry.Polygon(
                    ((-81, 35), (-81, 33), (-80, 33), (-80, 35), (-81, 35))
                ),
            ],
            'damaged': [False, True],
            'damaged_high_precision': [False, True],
            'damaged_high_recall': [False, True],
        }),
    }
    temp_prefix = f'{output_path}.tmp'
    for shard, shard_output in output_dictionary.items():
      shard_output.to_csv(f'{temp_prefix}_{shard}.csv', index=False)

    self.assertNotEmpty(glob.glob(f'{temp_prefix}*'))
    inference_lib.postprocess(temp_prefix, output_path, 'score')
    self.assertTrue(os.path.exists(output_path))
    self.assertEmpty(glob.glob(f'{temp_prefix}*'))
    df = pd.read_csv(output_path)
    self.assertSameElements(
        df.columns,
        [
            'example_id',
            'building_id',
            'longitude',
            'latitude',
            'score',
            'plus_code',
            'area_in_meters',
            'footprint_wkt',
            'damaged',
            'damaged_high_precision',
            'damaged_high_recall',
        ],
    )
    self.assertLen(df, 6)

    if 'GPKG' in fiona.supported_drivers:
      gdf = gp.read_file(f'{output_path}.gpkg')
      self.assertSameElements(
          gdf.columns,
          [
              'example_id',
              'building_id',
              'longitude',
              'latitude',
              'score',
              'plus_code',
              'area_in_meters',
              'damaged',
              'damaged_high_precision',
              'damaged_high_recall',
              'geometry'
          ],
      )
      self.assertLen(gdf, 6)

  def test_write_embedding_mean_positive_key(self):
    batch = ('pos', [np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])])
    path = os.path.join(_make_temp_dir(), 'positive_embeddings.npy')
    inference_lib._write_embedding_mean(
        batch, path, ''
    )
    with tf.io.gfile.GFile(path, 'rb') as f:
      embeddings = np.load(f)
    np.testing.assert_allclose(embeddings, np.array([2.5, 3.5, 4.5]))

  def test_write_embedding_mean_negative_key(self):
    batch = ('neg', [np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])])
    path = os.path.join(_make_temp_dir(), 'negative_embeddings.npy')
    inference_lib._write_embedding_mean(batch, '', path)
    with tf.io.gfile.GFile(path, 'rb') as f:
      embeddings = np.load(f)
    np.testing.assert_allclose(embeddings, np.array([2.5, 3.5, 4.5]))

  def test_write_embedding_mean_raise_key_error(self):
    batch = (
        'unknown key',
        [np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])],
    )
    path = os.path.join(_make_temp_dir(), 'negative_embeddings.npy')
    with self.assertRaisesRegex(
        ValueError, 'Unrecognized embedding key "unknown key"'
    ):
      inference_lib._write_embedding_mean(batch, '', path)


if __name__ == '__main__':
  absltest.main()
