"""Tests for inference_lib."""

import os
import tempfile

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that
import numpy as np
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
  model.save(model_path)


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
  return example


class TestModel(inference_lib.InferenceModel):
  def __init__(self, expected_batch_size: int, score: float):
    self._expected_batch_size = expected_batch_size
    self._score = score
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
    return np.ones((self._expected_batch_size,)) * self._score


class InferenceTest(absltest.TestCase):

  def test_run_inference(self):
    with test_pipeline.TestPipeline() as pipeline:
      examples = []
      for example_id in range(10):
        example = tf.train.Example()
        utils.add_int64_feature('int64_id', example_id, example)
        examples.append(example)

      examples_collection = (
          pipeline
          | beam.Create(examples)
      )
      batch_size = 4
      score = 0.5
      model = TestModel(batch_size, score)
      result = inference_lib.run_inference(
          examples_collection, 'score', batch_size, model
      )

      def _check_examples(examples):
        assert (
            len(examples) == 10
        ), f'Expected 10 examples in output, got {len(examples)}'
        for example in examples:
          assert example.features.feature['score'].float_list.value[0] == score

      assert_that(result, _check_examples)

  def test_tf2_model_prediction(self):
    model_path = os.path.join(_make_temp_dir(), 'model.keras')
    _create_test_model(model_path, 224)
    model = inference_lib.TF2InferenceModel(model_path, 224, False)
    model.prepare_model()

    examples = [_create_test_example(224, True) for i in range(3)]
    output_examples = model.predict_scores(examples)
    self.assertEqual(output_examples.shape, (3,))

  def test_tf2_model_prediction_no_small_images(self):
    model_path = os.path.join(_make_temp_dir(), 'model.keras')
    _create_test_model(model_path, 224)
    model = inference_lib.TF2InferenceModel(model_path, 224, False)
    model.prepare_model()

    examples = [_create_test_example(224, False) for i in range(3)]
    output_examples = model.predict_scores(examples)
    self.assertEqual(output_examples.shape, (3,))


if __name__ == '__main__':
  absltest.main()
