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

"""FixMatch training.

FixMatch can be viewed as a substantially simplified version of UDA and
ReMixMatch, where it combines two common techniques (pseudo-labeling and
consistency regularization) while removing many components (sharpening, training
signal annealing from UDA, distribution alignment and the rotation loss from
ReMixMatch, etc.).
Relevant paper: https://arxiv.org/pdf/2001.07685.pdf
"""

import dataclasses
import enum
import functools
from typing import Optional

import numpy as np
from skai.semi_supervised import train
from skai.semi_supervised import utils
from skai.semi_supervised.augment import augment_pool
from skai.semi_supervised.augment import ctaugment
from skai.semi_supervised.dataloader import prepare_ssl_data
import tensorflow.compat.v1 as tf


_AUGMENTATION_DEPTH = 2
_AUGMENTATION_DECAY = 0.99
_AUGMENTATION_THRESHOLD = 0.80

_TRAIN_KIMG = 1 << 14  # Duration of training in kibi-samples
_SHUFFLE_BUFFER_SIZE = 5000


@dataclasses.dataclass(frozen=True)
class FixMatchTrainingParams(train.TrainingParams):
  """Expected parameters for the base Model class."""
  pseudo_label_loss_weight: float
  confidence: float
  unlabeled_ratio: float
  num_parallel_calls: int


class FixMatch(train.ClassifySemi):
  """FixMatch class, which defines the training loop and model.

  FixMatch first generates pseudo-labels using the model’s predictions on
  weakly augmented unlabeled images. For a given image, the pseudo-label is only
  retained if the model produces a high-confidence prediction. The model is then
  trained to predict the pseudo-label when fed a strongly augmented version of
  the same image. CTA is the augmentation strategy used by default.

  Attributes:
    ops: Output of build_model. Contains training data, labels, training and
      classification operations.
    _next_labeled_batch: Nested structures of tf.Tensors containing the next
      batch of labeled data.
    _next_unlabeled_batch: Nested structures of tf.Tensors containing the next
      batch of unlabeled data.
    _unlabeled_ratio: Ratio of unlabeled to labeled data in training data.
    _num_parallel_calls: Number of threads to run per machine when augmenting.
  """

  ops: train.ModelOps
  _next_labeled_batch: tf.Tensor
  _next_unlabeled_batch: tf.Tensor
  _unlabeled_ratio: float
  _num_parallel_calls: int

  def __init__(self, params: FixMatchTrainingParams, train_dir: str,
               dataset: prepare_ssl_data.SSLDataset):
    super().__init__(params, train_dir=train_dir, dataset=dataset)

    self._unlabeled_ratio = params.unlabeled_ratio
    self._num_parallel_calls = params.num_parallel_calls

    if not self.inference_mode:
      # Prepare training data
      self._next_labeled_batch = self._dataset.train_labeled.repeat().shuffle(
          _SHUFFLE_BUFFER_SIZE).batch(
              params.batch).prefetch(16).make_one_shot_iterator().get_next()
      train_unlabeled = self._dataset.train_unlabeled.repeat().shuffle(
          _SHUFFLE_BUFFER_SIZE)
      train_unlabeled = train_unlabeled.batch(
          params.batch * self._unlabeled_ratio).prefetch(16)
      self._next_unlabeled_batch = train_unlabeled.make_one_shot_iterator(
      ).get_next()

    self.ops = self.build_model(params)
    self.ops.update_step = tf.assign_add(self._step, self._batch)

  def build_model(self, params: FixMatchTrainingParams) -> train.ModelOps:
    """Creates model for FixMatch.

    Args:
      params: Training parameters.

    Returns:
      FixMatch model that defines the training operation.
    """
    hwc = [self._dataset.height, self._dataset.width, self._dataset.channels]
    xt_in = tf.placeholder(tf.float32, [params.batch] + hwc,
                           'xt')  # Training labeled
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
    y_in = tf.placeholder(tf.float32,
                          [params.batch * params.unlabeled_ratio, 2] + hwc,
                          'y')  # Training unlabeled (weak, strong)
    l_in = tf.placeholder(tf.int32, [params.batch], 'labels')  # Labels

    # TODO(jlee24): Replace channel indices with enum constants.
    if self._dataset.use_pre_disaster_image:
      pre_weak = 0.5 * (1 + y_in[:, 0, :, :, :3])
      post_weak = 0.5 * (1 + y_in[:, 0, :, :, 3:6])
      pre_strong = 0.5 * (1 + y_in[:, 1, :, :, :3])
      post_strong = 0.5 * (1 + y_in[:, 1, :, :, 3:6])
      # Mask is unaugmented, so it is same across weak and strong.
      # TODO(jlee24): Update mask functionality such that it is compatible
      # with the case of using only a single image in training.
      if y_in.shape[4] > 6:
        pre_mask = y_in[:, 0, :, :, 7]
        post_mask = y_in[:, 0, :, :, 8]
        tf.summary.image(
            'pre_mask', pre_mask, max_outputs=3, family='train_unlabeled')
        tf.summary.image(
            'post_mask', post_mask, max_outputs=3, family='train_unlabeled')
      tf.summary.image(
          'pre_weak', pre_weak, max_outputs=3, family='train_unlabeled')
      tf.summary.image(
          'pre_strong', pre_strong, max_outputs=3, family='train_unlabeled')
    else:
      post_weak = 0.5 * (1 + y_in[:, 0, :, :3])  # use all channels
      post_strong = 0.5 * (1 + y_in[:, 1, :, :3])
    tf.summary.image(
        'post_weak', post_weak, max_outputs=3, family='train_unlabeled')
    tf.summary.image(
        'post_strong', post_strong, max_outputs=3, family='train_unlabeled')

    # Gets the current time step as a fraction of the total time
    # Clips the value to be between 0 and 1
    lrate = tf.clip_by_value(
        tf.to_float(self._step) / (_TRAIN_KIMG << 10), 0, 1)
    # Applies cosine learning rate decay
    lr = params.lr * tf.cos(lrate * (7 * np.pi) / (2 * 8))
    tf.summary.scalar('monitors/lr', lr)

    # Compute logits for xt_in and y_in
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    logits = utils.para_cat(lambda x: self.classify(x, training=True),
                            tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0))
    post_ops = [
        v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if v not in skip_ops
    ]
    logits_x = logits[:params.batch]
    logits_weak, logits_strong = tf.split(logits[params.batch:], 2)
    del logits, skip_ops

    # Labeled cross-entropy
    loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=l_in, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    tf.summary.scalar('losses/xe', loss_xe)

    # Pseudo-label cross entropy for unlabeled data
    pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
    loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.argmax(pseudo_labels, axis=1), logits=logits_strong)
    pseudo_mask = tf.to_float(
        tf.reduce_max(pseudo_labels, axis=1) >= params.confidence)
    tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
    loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)
    tf.summary.scalar('losses/xeu', loss_xeu)

    # L2 regularization
    loss_wd = sum(
        tf.nn.l2_loss(v)
        for v in utils.model_vars('classify')
        if 'kernel' in v.name)
    tf.summary.scalar('losses/wd', loss_wd)

    ema = tf.train.ExponentialMovingAverage(decay=params.ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops.append(ema_op)

    train_op = tf.train.MomentumOptimizer(
        lr, 0.9, use_nesterov=True).minimize(
            loss_xe + params.pseudo_label_loss_weight * loss_xeu +
            params.weight_decay * loss_wd,
            colocate_gradients_with_ops=True)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(*post_ops)

    return train.ModelOps(
        xt=xt_in,
        x=x_in,
        y=y_in,
        label=l_in,
        train_op=train_op,
        classify_raw=tf.nn.softmax(self.classify(
            x_in, training=False)),  # No EMA, for debugging.
        classify_op=tf.nn.softmax(
            self.classify(x_in, getter=ema_getter, training=False)))


class FixMatchCTA(FixMatch):
  """Subclass of FixMatch that uses CTAugment to augment data.

  Attributes:
    _cta_object: Instance of CTAugment that updates and prints policy.
    _iterator: Tuple of iterators that return evaluated batches of labeled and
      unlabeled data, respectively.
  """

  _cta_object: ctaugment.CTAugment
  _labeled_iterator: Optional[augment_pool.AugmentPoolCTA] = None
  _unlabeled_iterator: Optional[augment_pool.AugmentPoolCTA] = None

  def __init__(self, params: FixMatchTrainingParams, train_dir: str,
               dataset: prepare_ssl_data.SSLDataset):
    super().__init__(params=params, train_dir=train_dir, dataset=dataset)
    self._cta_object = ctaugment.CTAugment(_AUGMENTATION_DEPTH,
                                           _AUGMENTATION_THRESHOLD,
                                           _AUGMENTATION_DECAY)
    self._labeled_iterator = None
    self._unlabeled_iterator = None

  def prepare_to_train(self, session: tf.Session):
    """Sets `_iterator` class variable expected format of iterators over labeled and unlabeled data."""
    self._labeled_iterator = augment_pool.AugmentPoolCTA(
        next_batch=self._next_labeled_batch,
        session=session,
        num_parallel_calls=self._num_parallel_calls,
        cta=self._cta_object,
        probe=True)
    self._unlabeled_iterator = augment_pool.AugmentPoolCTA(
        next_batch=self._next_unlabeled_batch,
        session=session,
        num_parallel_calls=self._num_parallel_calls,
        cta=self._cta_object,
        probe=False)

  def train_step(self, train_session: tf.train.MonitoredTrainingSession) -> int:
    """Defines a step in the training loop.

    Args:
      train_session: MonitoredTrainingSession.

    Returns:
      Newly updated training step.
    """
    labeled_batch = next(self._labeled_iterator)
    unlabeled_batch = next(self._unlabeled_iterator)
    eval_results, _, step = train_session.run(
        [self.ops.classify_op, self.ops.train_op, self.ops.update_step],
        feed_dict={
            self.ops.y: unlabeled_batch.augmented_image_batch,
            self.ops.x: labeled_batch.probed_image_batch,
            self.ops.xt: labeled_batch.augmented_image_batch,
            self.ops.label: labeled_batch.label_batch
        })
    # `eval_results` has shape (batch_size, num_class), contains class probs
    for i, class_probs in enumerate(eval_results):
      correct_class = labeled_batch.label_batch[i]
      error = 1 - class_probs[correct_class]
      self._cta_object.update_rates(labeled_batch.policy_batch[i], error)
    return step

  def eval_stats(
      self,
      session: tf.Session,
      batch: Optional[int] = None,
      classify_op: Optional[tf.Tensor] = None) -> train.EvaluationType:
    """Evaluate model on train, valid and test."""
    accuracies = super().eval_stats(
        session=session, batch=batch, classify_op=classify_op)
    if not self.inference_mode:
      # Print CTA policy only when training to see updates to policy.
      self._logger.add_to_print_queue(self._cta_object.stats())
    return accuracies


class FixMatchRA(FixMatch):
  """Subclass of FixMatch that uses RandAugment to augment data.

  Attributes:
    _iterator: Tuple of iterators that return evaluated batches of labeled and
      unlabeled data, respectively.
  """

  _labeled_iterator: Optional[train.BatchIterator] = None
  _unlabeled_iterator: Optional[augment_pool.AugmentPoolRAMC] = None

  def __init__(self, params: FixMatchTrainingParams, train_dir: str,
               dataset: prepare_ssl_data.SSLDataset):
    super().__init__(params=params, train_dir=train_dir, dataset=dataset)
    self._labeled_iterator = None
    self._unlabeled_iterator = None

  def prepare_to_train(self, session: tf.Session):
    """Sets `_iterator` class variable to tuple of iterators for labeled and unlabeled data, respectively."""
    self._labeled_iterator = train.BatchIterator(
        self._next_labeled_batch, session=session)
    self._unlabeled_iterator = augment_pool.AugmentPoolRAMC(
        next_batch=self._next_unlabeled_batch,
        session=session,
        num_parallel_calls=self._num_parallel_calls)

  def train_step(self, train_session: tf.train.MonitoredTrainingSession) -> int:
    """Gets internal iterators and runs next batches through training operations.

    Args:
      train_session: Session for the training loop.

    Returns:
      Newly updated training step.
    """
    labeled_batch = next(self._labeled_iterator)
    unlabeled_batch = next(self._unlabeled_iterator)
    return train_session.run(
        [self.ops.train_op, self.ops.update_step],
        feed_dict={
            self.ops.y: unlabeled_batch.augmented_image_batch,
            self.ops.xt: labeled_batch[prepare_ssl_data.IMAGE_KEY],
            self.ops.label: labeled_batch[prepare_ssl_data.LABEL_KEY]
        })[1]


class StrategyOptions(enum.Enum):
  CTA = FixMatchCTA
  RA = FixMatchRA
