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

"""MixMatch class.

This file was adapted from this source:
google3/learning/brain/research/red_team/semi_supervised/mixmatch.py

- Ensure class consistency by producing a group of a specified number of
augmentations of the same image and guessing the label for the group.
- Rectify the classifier predicted distribution by re-weighting classes to match
the expected unlabelled distribution.
- Use the rectified distribution directly as a smooth label in MixUp.
"""

import dataclasses
import functools
from typing import Optional

from skai.semi_supervised import layers
from skai.semi_supervised import train
from skai.semi_supervised import utils
from skai.semi_supervised.dataloader import prepare_ssl_data
import tensorflow.compat.v1 as tf


_SHUFFLE_BUFFER_SIZE = 5000  # Size of buffer in dataset to shuffle


@dataclasses.dataclass
class GuessLabel:
  p_target: float
  p_model: float


@dataclasses.dataclass(frozen=True)
class MixMatchTrainingParams(train.TrainingParams):
  """Expected parameters for the base Model class."""
  beta: float
  logit_norm: bool
  sharpening_temperature: float
  mixup_mode: str
  num_augmentations: int
  dbuf: int
  w_match: float
  warmup_kimg: int


class MixMatch(train.ClassifySemi):
  """MixMatch training setup.

  Attributes:
    _nclass: Number of classes.
    _train_data: Tuple of nested structures of tf.Tensors containing the next
      elements of labeled data and unlabeled data, respectively.
    _iterator: Tuple of iterators that return evaluated batches of labeled and
      unlabeled data, respectively.
    ops: Output of build_model. Contains training data, labels, training and
      classification operations.
  """

  _nclass: int
  _labeled_iterator: Optional[train.BatchIterator] = None
  _unlabeled_iterator: Optional[train.BatchIterator] = None
  ops: train.ModelOps

  def __init__(self, params: MixMatchTrainingParams, train_dir: str,
               dataset: prepare_ssl_data.SSLDataset):
    super().__init__(params, train_dir=train_dir, dataset=dataset)
    self._nclass = params.nclass

    # Prepare training data
    train_labeled = self._dataset.train_labeled.repeat().shuffle(
        _SHUFFLE_BUFFER_SIZE).batch(params.batch).prefetch(16)
    self._next_labeled_batch = train_labeled.make_one_shot_iterator().get_next()
    train_unlabeled = self._dataset.train_unlabeled.repeat().shuffle(
        _SHUFFLE_BUFFER_SIZE).batch(params.batch).prefetch(16)
    self._next_unlabeled_batch = train_unlabeled.make_one_shot_iterator(
    ).get_next()

    # Initialize iterators, which will be populated in prepare_iterators
    self._labeled_iterator = None
    self._unlabeled_iterator = None

    self.ops = self.build_model(params)
    self.ops.update_step = tf.assign_add(self._step, params.batch)

  def distribution_summary(self, p_data, p_model, p_target=None):

    def kl(p, q):
      p /= tf.reduce_sum(p)
      q /= tf.reduce_sum(q)
      return -tf.reduce_sum(p * tf.log(q / p))

    tf.summary.scalar('metrics/kld', kl(p_data, p_model))
    if p_target is not None:
      tf.summary.scalar('metrics/kld_target', kl(p_data, p_target))

    for i in range(self._nclass):
      tf.summary.scalar('matching/class%d_ratio' % i, p_model[i] / p_data[i])
    for i in range(self._nclass):
      tf.summary.scalar('matching/val%d' % i, p_model[i])

  def augment(self, x, l, beta):
    assert 0, 'Do not call.'

  def _guess_label(self, y, p_data, p_model, sharpening_temperature):
    logits_y = [self.classify(yi, training=True) for yi in y]
    logits_y = tf.concat(logits_y, 0)
    # Compute predicted probability distribution py.
    p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self._nclass])
    p_model_y = tf.reduce_mean(p_model_y, axis=0)
    # Compute the target distribution.
    p_target = tf.pow(p_model_y, 1. / sharpening_temperature)
    p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
    return GuessLabel(p_target=p_target, p_model=p_model_y)

  def prepare_to_train(self, session: tf.Session):
    """Populates `_iterator` class variable with expected labeled and unlabeled iterators."""
    self._labeled_iterator = train.BatchIterator(
        next_batch=self._next_labeled_batch, session=session)
    self._unlabeled_iterator = train.BatchIterator(
        next_batch=self._next_unlabeled_batch, session=session)

  def train_step(self, train_session: tf.train.MonitoredTrainingSession) -> int:
    """Gets internal iterators and runs next batches through training operations."""
    labeled_batch = next(self._labeled_iterator)
    unlabeled_batch = next(self._unlabeled_iterator)
    return train_session.run(
        [self.ops.train_op, self.ops.update_step],
        feed_dict={
            self.ops.x: labeled_batch[prepare_ssl_data.IMAGE_KEY],
            self.ops.y: unlabeled_batch[prepare_ssl_data.IMAGE_KEY],
            self.ops.label: labeled_batch[prepare_ssl_data.LABEL_KEY]
        })[1]

  def build_model(self, params: MixMatchTrainingParams) -> train.ModelOps:
    hwc = [self._dataset.height, self._dataset.width, self._dataset.channels]
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
    y_in = tf.placeholder(tf.float32, [None, params.num_augmentations] + hwc,
                          'y')
    l_in = tf.placeholder(tf.int32, [None], 'labels')
    weight_decay = params.weight_decay * params.lr
    w_match = params.w_match * tf.clip_by_value(
        tf.cast(self._step, tf.float32) / (params.warmup_kimg << 10), 0, 1)
    augment = layers.MixMode(params.mixup_mode)

    # Moving average of the current estimated label distribution
    p_model = layers.PMovingAverage('p_model', params.nclass, params.dbuf)
    # Rectified distribution (only for plotting)
    p_target = layers.PMovingAverage('p_target', params.nclass, params.dbuf)

    # Known (or inferred) true unlabeled distribution
    p_data = layers.PData(self._dataset)

    y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
    guess = self._guess_label(
        tf.split(y, params.num_augmentations),
        p_data(),
        p_model(),
        sharpening_temperature=params.sharpening_temperature)
    ly = tf.stop_gradient(guess.p_target)
    lx = tf.one_hot(l_in, params.nclass)
    xy, labels_xy = augment([x_in] + tf.split(y, params.num_augmentations),
                            [lx] + [ly] * params.num_augmentations,
                            [params.beta, params.beta])
    x, y = xy[0], xy[1:]
    labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
    del xy, labels_xy

    batches = layers.interleave([x] + y, params.batch)
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    logits = [self.classify(batches[0], training=True)]
    post_ops = [
        v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if v not in skip_ops
    ]
    for batchi in batches[1:]:
      logits.append(self.classify(batchi, training=True))
    logits = layers.interleave(logits, self._batch)
    logits_x = logits[0]
    logits_y = tf.concat(logits[1:], 0)

    loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
    loss_l2u = tf.reduce_mean(loss_l2u)
    tf.summary.scalar('losses/xe', loss_xe)
    tf.summary.scalar('losses/l2u', loss_l2u)
    self.distribution_summary(p_data(), p_model(), p_target())

    ema = tf.train.ExponentialMovingAverage(decay=params.ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops.extend([
        ema_op,
        p_model.update(guess.p_model),
        p_target.update(guess.p_target)
    ])
    if p_data.has_update:
      post_ops.append(p_data.update(lx))
    post_ops.extend([
        tf.assign(v, v * (1 - weight_decay))
        for v in utils.model_vars('classify')
        if 'kernel' in v.name
    ])

    train_op = tf.train.AdamOptimizer(params.lr).minimize(
        loss_xe + w_match * loss_l2u, colocate_gradients_with_ops=True)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(*post_ops)

    # Tuning op: only retrain batch norm.
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.classify(batches[0], training=True)
    train_bn = tf.group(*[
        v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if v not in skip_ops
    ])

    return train.ModelOps(
        x=x_in,
        y=y_in,
        label=l_in,
        train_op=train_op,
        tune_op=train_bn,
        classify_raw=tf.nn.softmax(
            self.classify(x_in, logit_norm=False,
                          training=False)),  # No EMA, for debugging.
        classify_op=tf.nn.softmax(
            self.classify(
                x_in, logit_norm=False, getter=ema_getter, training=False)))
