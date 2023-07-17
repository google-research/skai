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

"""Classifier architectures for base model of MixMatch.

This file was copied from this source:
google3/learning/brain/research/red_team/semi_supervised/libml/models.py
"""

import abc
import functools
import itertools
from typing import Optional

import numpy as np
from skai.semi_supervised import layers
from skai.semi_supervised.dataloader import prepare_ssl_data
import tensorflow.compat.v1 as tf


RESNET = 'resnet'
SHAKENET = 'shake'
SUPPORTED_ARCHITECTURES = [RESNET, SHAKENET]


class Classifier(abc.ABC):
  """Factory method to return classifier function with desired architecture."""

  def __init__(self,
               dataset: prepare_ssl_data.SSLDataset,
               nclass: int,
               scales: int,
               conv_filter_size: int,
               num_residual_repeat_per_stage: int):

    self.dataset = dataset
    self.nclass = nclass
    self.scales = scales
    self.conv_filter_size = conv_filter_size
    self.num_residual_repeat_per_stage = num_residual_repeat_per_stage

  @abc.abstractmethod
  def classify(
      self,
      x: np.ndarray,
      training: Optional[bool],
      getter: Optional[tf.train.ExponentialMovingAverage] = None) -> tf.Tensor:
    raise NotImplementedError()


class ResNetClassifier(Classifier):
  """Wide ResNet classifier implementation."""

  def classify(
      self,
      x: np.ndarray,
      training: Optional[bool],
      getter: Optional[tf.train.ExponentialMovingAverage] = None) -> tf.Tensor:
    """Passes data through classifier and outputs logits."""
    leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
    bn_args = dict(training=training, momentum=0.999)

    def conv_args(k, f):
      return dict(
          padding='same',
          kernel_initializer=tf.random_normal_initializer(
              stddev=tf.rsqrt(0.5 * k * k * f)))

    def residual(x0,
                 conv_filter_size,
                 stride=1,
                 activate_before_residual=False):
      """Function to create a residual block."""
      x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
      if activate_before_residual:
        x0 = x

      x = tf.layers.conv2d(
          x,
          conv_filter_size,
          3,
          strides=stride,
          **conv_args(3, conv_filter_size))
      x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
      x = tf.layers.conv2d(x, conv_filter_size, 3,
                           **conv_args(3, conv_filter_size))

      if x0.get_shape()[3] != conv_filter_size:
        x0 = tf.layers.conv2d(
            x0,
            conv_filter_size,
            1,
            strides=stride,
            **conv_args(1, conv_filter_size))

      return x0 + x

    with tf.variable_scope(
        'classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
      y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3,
                           **conv_args(3, 16))
      for scale in range(self.scales):
        y = residual(
            y,
            self.conv_filter_size << scale,
            stride=2 if scale else 1,
            activate_before_residual=scale == 0)
        for _ in range(self.num_residual_repeat_per_stage - 1):
          y = residual(y, self.conv_filter_size << scale)

      y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
      y = tf.reduce_mean(y, [1, 2])
      logits = tf.layers.dense(
          y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
    return logits


class ShakeNetClassifier(Classifier):
  """Wide ResNet classifier with Shake-Shake regularization implementation."""

  def classify(
      self,
      x: np.ndarray,
      training: Optional[bool],
      getter: Optional[tf.train.ExponentialMovingAverage] = None) -> tf.Tensor:
    """Passes data through classifier and outputs logits."""
    bn_args = dict(training=training, momentum=0.999)

    def conv_args(k, f):
      return dict(
          padding='same',
          use_bias=False,
          kernel_initializer=tf.random_normal_initializer(
              stddev=tf.rsqrt(0.5 * k * k * f)))

    def residual(x0, conv_filter_size, stride=1):
      """Function to create a residual block."""

      def branch():
        x = tf.nn.relu(x0)
        x = tf.layers.conv2d(
            x,
            conv_filter_size,
            3,
            strides=stride,
            **conv_args(3, conv_filter_size))
        x = tf.nn.relu(tf.layers.batch_normalization(x, **bn_args))
        x = tf.layers.conv2d(x, conv_filter_size, 3,
                             **conv_args(3, conv_filter_size))
        x = tf.layers.batch_normalization(x, **bn_args)
        return x

      x = layers.shakeshake(branch(), branch(), training)

      if stride == 2:
        x1 = tf.layers.conv2d(
            tf.nn.relu(x0[:, ::2, ::2]), conv_filter_size >> 1, 1,
            **conv_args(1, conv_filter_size >> 1))
        x2 = tf.layers.conv2d(
            tf.nn.relu(x0[:, 1::2, 1::2]), conv_filter_size >> 1, 1,
            **conv_args(1, conv_filter_size >> 1))
        x0 = tf.concat([x1, x2], axis=3)
        x0 = tf.layers.batch_normalization(x0, **bn_args)
      elif x0.get_shape()[3] != conv_filter_size:
        x0 = tf.layers.conv2d(x0, conv_filter_size, 1,
                              **conv_args(1, conv_filter_size))
        x0 = tf.layers.batch_normalization(x0, **bn_args)

      return x0 + x

    with tf.variable_scope(
        'classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
      y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3,
                           **conv_args(3, 16))
      for scale, i in itertools.product(
          range(self.scales), range(self.num_residual_repeat_per_stage)):
        with tf.variable_scope('layer%d.%d' % (scale + 1, i)):
          if i == 0:
            y = residual(
                y, self.conv_filter_size << scale, stride=2 if scale else 1)
          else:
            y = residual(y, self.conv_filter_size << scale)

      y = tf.reduce_mean(y, [1, 2])
      logits = tf.layers.dense(
          y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
    return logits

