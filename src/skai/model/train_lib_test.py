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

"""Tests for training library."""

from collections.abc import Sequence
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from skai.model import data
from skai.model import log_metrics_callback
from skai.model import models
from skai.model import train_lib
import tensorflow as tf


def _make_temp_dir() -> str:
  return tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))


def _make_serialized_image(size: int, pixel_value: int) -> bytes:
  image = np.ones((size, size, 3), dtype=np.uint8) * pixel_value
  return tf.io.encode_png(image).numpy()


def _make_example(
    int64_id: int,
    longitude: float,
    latitude: float,
    encoded_coordinates: str,
    label: float,
    patch_size: int,
    large_patch_size: int,
    before_pixel_value: int,
    after_pixel_value: int,
) -> tf.train.Example:
  example = tf.train.Example()
  example.features.feature['coordinates'].float_list.value.extend(
      (longitude, latitude)
  )
  example.features.feature['int64_id'].int64_list.value.append(
      int64_id
  )
  example.features.feature['encoded_coordinates'].bytes_list.value.append(
      encoded_coordinates.encode()
  )
  example.features.feature['label'].float_list.value.append(label)
  example.features.feature['pre_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size, before_pixel_value)
  )
  example.features.feature['post_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size, after_pixel_value)
  )
  example.features.feature['pre_image_png_large'].bytes_list.value.append(
      _make_serialized_image(large_patch_size, before_pixel_value)
  )
  example.features.feature['post_image_png_large'].bytes_list.value.append(
      _make_serialized_image(large_patch_size, after_pixel_value)
  )
  return example


def _write_tfrecord(examples: Sequence[tf.train.Example], path: str) -> None:
  with tf.io.TFRecordWriter(path) as file_writer:
    for example in examples:
      file_writer.write(example.SerializeToString())


def _create_test_data():
  examples_dir = _make_temp_dir()
  labeled_train_path = os.path.join(
      examples_dir, 'train_labeled_examples.tfrecord')
  labeled_test_path = os.path.join(
      examples_dir, 'test_labeled_examples.tfrecord')
  unlabeled_path = os.path.join(
      examples_dir, 'unlabeled_examples.tfrecord')

  _write_tfrecord([
      _make_example(0, 0, 0, 'A0', 0, 64, 256, 0, 255),
  ], labeled_train_path)

  _write_tfrecord([
      _make_example(1, 1, 0, 'B0', 0, 64, 256, 0, 255),
  ], labeled_test_path)

  _write_tfrecord([
      _make_example(2, 2, 0, 'C0', -1, 64, 256, 0, 255),
  ], unlabeled_path)

  return labeled_train_path, labeled_test_path, unlabeled_path


class TrainLibTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    labeled_train_path, labeled_test_path, unlabeled_path = _create_test_data()
    self.labeled_train_path = labeled_train_path
    self.labeled_test_path = labeled_test_path
    self.unlabeled_path = unlabeled_path

    dataset_builder = data.get_dataset('skai')

    kwargs = {
        'labeled_train_pattern': self.labeled_train_path,
        'unlabeled_train_pattern': self.unlabeled_path,
        'validation_pattern': self.labeled_test_path,
        'use_post_disaster_only': False,
        'load_small_images': True,
        'data_dir': _make_temp_dir(),
    }

    self.dataloader = dataset_builder(
        1,
        initial_sample_proportion=1,
        subgroup_ids=(),
        subgroup_proportions=(),
        **kwargs
    )

    self.output_dir = _make_temp_dir()
    self.model_params_one_head = models.ModelTrainingParameters(
        model_name='',
        train_bias=False,
        num_classes=2,
        num_subgroups=2,
        subgroup_sizes={'0': 2, '1': 1},
        worst_group_label=1,
        num_epochs=1,
        num_channels=6,
        l2_regularization_factor=1.,
        optimizer='adam',
        learning_rate=1e-5,
        batch_size=1,
        load_pretrained_weights=False,
        use_pytorch_style_resnet=False,
        do_reweighting=False,
        reweighting_lambda=0.,
        reweighting_signal=0.
    )
    self.model_params_two_head = models.ModelTrainingParameters(
        model_name='',
        train_bias=True,
        num_classes=2,
        num_subgroups=2,
        subgroup_sizes={'0': 2, '1': 1},
        worst_group_label=1,
        num_epochs=1,
        num_channels=6,
        l2_regularization_factor=1.0,
        optimizer='adam',
        learning_rate=1e-5,
        batch_size=1,
        load_pretrained_weights=False,
        use_pytorch_style_resnet=False,
        do_reweighting=False,  # TODO(jlee24): Test reweighting.
        reweighting_lambda=0.0,
        reweighting_signal=0.0,
    )
    keys_tensor = tf.constant(
        [0, 1, 2], dtype=tf.int64
    )
    vals_tensor = tf.constant([0, 1, 2], dtype=tf.int64)
    self.example_id_to_bias_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1,
    )
    log_metrics_callback.LogMetricsCallback = mock.Mock(
        return_value=tf.keras.callbacks.Callback()
    )
    self.strategy = tf.distribute.get_strategy()

  @parameterized.named_parameters(
      (model_name, model_name)
      for model_name in models.MODEL_REGISTRY.keys()
  )
  def test_init_model_one_head(self, model_name):
    """Tests that each model class can be initialized and ingest data."""
    self.model_params_one_head.model_name = model_name
    one_head_model = train_lib.init_model(
        self.model_params_one_head, 'test_init_model'
    )
    features = list(self.dataloader.train_ds.batch(1))[0]
    pred = one_head_model.call(features['input_feature'])
    self.assertLen(pred['main'][0], self.model_params_one_head.num_classes)

  @parameterized.named_parameters(
      (model_name, model_name)
      for model_name in models.MODEL_REGISTRY.keys()
  )
  def test_init_model_two_head(self, model_name):
    """Tests that each model class can be initialized with two output heads."""
    self.model_params_two_head.model_name = model_name
    two_head_model = train_lib.init_model(
        self.model_params_two_head,
        'test_init_model',
        example_id_to_bias_table=self.example_id_to_bias_table,
    )
    features = list(self.dataloader.train_ds.batch(1))[0]
    pred = two_head_model.call(features['input_feature'])
    self.assertLen(pred['main'][0], self.model_params_two_head.num_classes)
    self.assertLen(pred['bias'][0], self.model_params_two_head.num_classes)

  @parameterized.named_parameters(
      (model_name, model_name)
      for model_name in ['resnet50v2']
  )
  def test_train_model_two_head(self, model_name):
    """Tests that each model class can be trained with two output heads."""
    self.model_params_two_head.model_name = model_name
    two_head_model = train_lib.init_model(
        model_params=self.model_params_two_head,
        experiment_name='test_model_train',
        example_id_to_bias_table=self.example_id_to_bias_table,
    )

    history = two_head_model.fit(
        self.dataloader.train_ds.batch(2),
        validation_data=self.dataloader.eval_ds['val'].batch(2),
        epochs=self.model_params_two_head.num_epochs,
    )

    self.assertFalse(tf.math.is_nan(history.history['main_loss']))
    self.assertFalse(tf.math.is_nan(history.history['val_main_loss']))
    self.assertIsNotNone(two_head_model)

  @parameterized.named_parameters(
      (model_name, model_name)
      for model_name in ['resnet50v2']
  )
  def test_train_and_load_model_from_checkpoint(self, model_name):
    """Tests that each model class can be saved and loaded from a checkpoint."""
    self.model_params_one_head.model_name = model_name
    callbacks = train_lib.create_callbacks(
        self.output_dir,
        save_model_checkpoints=True,
        save_best_model=False,
        early_stopping=False,
        batch_size=2,
        num_train_examples=self.dataloader.num_train_examples)

    # Callback should save checkpoint automatically.
    train_lib.run_train(
        self.dataloader.train_ds.batch(2),
        self.dataloader.eval_ds['val'].batch(2),
        self.model_params_one_head,
        'test_model_eval',
        callbacks=callbacks,
        strategy=self.strategy,
    )
    checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
    self.assertNotEmpty(tf.io.gfile.listdir(checkpoint_dir))

    # Newly initialized model should be able to load checkpoint.
    one_head_model = train_lib.init_model(
        model_params=self.model_params_one_head,
        experiment_name='test_model_load_checkpoint',
    )
    best_latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    load_status = one_head_model.load_weights(best_latest_checkpoint)
    load_status.assert_existing_objects_matched()

    # Model with checkpoint should be able to evaluate data.
    results = one_head_model.evaluate(
        self.dataloader.eval_ds['val'].batch(2), return_dict=True
    )
    self.assertIsInstance(results, dict)
    self.assertNotEmpty(results)
    self.assertIn('main_aucpr_0_vs_rest', results.keys())
    self.assertIn('main_aucpr_1_vs_rest', results.keys())

  @parameterized.named_parameters(
      (model_name, model_name)
      for model_name in models.MODEL_REGISTRY.keys()
  )
  def test_train_and_load_entire_model(self, model_name):
    """Tests that each model class can be saved as a model file."""
    self.model_params_one_head.model_name = model_name
    callbacks = train_lib.create_callbacks(
        self.output_dir,
        save_model_checkpoints=False,
        save_best_model=True,
        early_stopping=False,
        batch_size=2,
        num_train_examples=self.dataloader.num_train_examples)

    # Callback should save model automatically.
    train_lib.run_train(
        self.dataloader.train_ds.batch(2),
        self.dataloader.eval_ds['val'].batch(2),
        self.model_params_one_head,
        'test_model_eval',
        callbacks=callbacks,
        strategy=self.strategy,
    )

    model_dir = self.output_dir
    if tf.io.gfile.exists(os.path.join(model_dir, 'model')):
      model_dir = os.path.join(model_dir, 'model')
    if tf.io.gfile.exists(os.path.join(model_dir, 'saved_model.pb')):
      best_model_dir = model_dir
    else:
      best_model_dir = os.path.join(
          model_dir, sorted(tf.io.gfile.listdir(model_dir))[-1]
      )
    self.assertNotEmpty(tf.io.gfile.listdir(best_model_dir))

    # Model should be able to be compiled and then used for evaluation.
    loaded_model = tf.keras.models.load_model(best_model_dir)
    compiled_model = train_lib.compile_model(
        loaded_model, loaded_model.model.model_params
    )
    results = compiled_model.evaluate(
        self.dataloader.eval_ds['val'].batch(2),
        return_dict=True,
    )
    self.assertIsInstance(results, dict)
    self.assertNotEmpty(results)
    self.assertIn('main_aucpr_0_vs_rest', results.keys())
    self.assertIn('main_aucpr_1_vs_rest', results.keys())

  # TODO(jlee24): Test ensemble functionality.


if __name__ == '__main__':
  absltest.main()
