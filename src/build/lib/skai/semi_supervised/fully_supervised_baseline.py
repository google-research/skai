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

"""Class for training fully supervised models as a baseline for SSL methods.

This file was adapted from:
google3/experimental/brain/red_team/fixmatch/fully_supervised/fs_baseline.py
"""

import functools
from typing import Tuple

from skai.semi_supervised import fully_supervised
from skai.semi_supervised import train
from skai.semi_supervised import utils
import tensorflow.compat.v1 as tf


class FullySupervisedBaseline(fully_supervised.ClassifyFullySupervised):
  """Fully supervised baseline that trains base model from only labeled data."""

  def augment(self, training_data: tf.Tensor, labels: tf.Tensor,
              smoothing: float, nclass: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Rather than hard labels of 0 or 1, spread probability across classes."""
    return training_data, labels - smoothing * (labels - 1. / nclass)

  def build_model(self, params: fully_supervised.FullySupervisedTrainingParams):
    """Creates a classifier model with the given arguments.

    Args:
      params: Training parameters.

    Returns:
      A FullySupervisedBaseline model that can be trained.
    """
    hwc = [self._dataset.height, self._dataset.width, self._dataset.channels]
    xt_in = tf.placeholder(tf.float32, [params.batch] + hwc,
                           'xt')  # For training
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
    l_in = tf.placeholder(tf.int32, [params.batch], 'labels')
    weight_decay = params.weight_decay * params.lr

    x, labels_x = self.augment(xt_in, tf.one_hot(l_in, params.nclass),
                               params.smoothing, params.nclass)
    logits_x = self.classify(x, training=True)

    loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    tf.summary.scalar('losses/xe', loss_xe)

    ema = tf.train.ExponentialMovingAverage(decay=params.ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [ema_op]
    post_ops.extend([
        tf.assign(v, v * (1 - weight_decay))
        for v in utils.model_vars('classify')
        if 'kernel' in v.name
    ])

    train_op = tf.train.AdamOptimizer(params.lr).minimize(
        loss_xe, colocate_gradients_with_ops=True)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(*post_ops)

    return train.ModelOps(
        xt=xt_in,
        x=x_in,
        label=l_in,
        train_op=train_op,
        classify_raw=tf.nn.softmax(self.classify(
            x_in, training=False)),  # No EMA, for debugging.
        classify_op=tf.nn.softmax(
            self.classify(x_in, getter=ema_getter, training=False)))
