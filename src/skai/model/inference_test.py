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

import os
import tempfile

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that
import numpy as np
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


def _create_test_example(
    image_size: int, include_small_images: bool
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
  utils.add_bytes_feature('example_id', b'deadbeef', example)
  utils.add_float_list_feature('coordinates', [0.0, 0.0], example)
  utils.add_float_list_feature('area_in_meters', [12.0], example)
  utils.add_bytes_list_feature('plus_code', [b'abcdef'], example)
  utils.add_bytes_list_feature('encoded_coordinates', [b'beefdead'], example)

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
    return np.array(
        [
            self._id_to_score[utils.get_int64_feature(e, 'int64_id')[0]]
            for e in batch
        ]
    )


class InferenceTest(absltest.TestCase):

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

      examples_collection = (
          pipeline
          | beam.Create(examples)
      )
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

      assert_that(result, _check_examples)

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

      examples_collection = (
          pipeline
          | beam.Create(examples)
      )
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

      assert_that(result, _check_examples)

  def test_tf2_model_prediction(self):
    model_path = os.path.join(_make_temp_dir(), 'model.keras')
    _create_test_model(model_path, 224)
    model = inference_lib.TF2InferenceModel(
        model_path, 224, False, [], inference_lib.ModelType.CLASSIFICATION
    )
    model.prepare_model()

    examples = [_create_test_example(224, True) for i in range(3)]
    output_examples = model.predict_scores(examples)
    self.assertEqual(output_examples.shape, (3,))

  def test_tf2_model_prediction_no_small_images(self):
    model_path = os.path.join(_make_temp_dir(), 'model.keras')
    _create_test_model(model_path, 224)
    model = inference_lib.TF2InferenceModel(
        model_path, 224, False, [], inference_lib.ModelType.CLASSIFICATION
    )
    model.prepare_model()

    examples = [_create_test_example(224, False) for i in range(3)]
    output_examples = model.predict_scores(examples)
    self.assertEqual(output_examples.shape, (3,))

  def test_csv_output(self):
    example = _create_test_example(256, False)
    csv_output = inference_lib._format_example_to_csv_row(example)
    expected_wkt = 'POINT (12.0000000000000000 15.0000000000000000)'
    self.assertEqual(
        csv_output, f'deadbeef,0.0,0.0,,abcdef,12.0,{expected_wkt}\r\n'
    )


if __name__ == '__main__':
  absltest.main()
