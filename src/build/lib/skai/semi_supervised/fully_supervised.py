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

"""Class that defines training step for the fully supervised baseline model.

This file was adapted from:
google3/experimental/brain/red_team/fixmatch/fully_supervised/lib/train.py
"""

import dataclasses
from typing import Optional

from skai.semi_supervised import train
from skai.semi_supervised.dataloader import prepare_ssl_data
import tensorflow.compat.v1 as tf


_SHUFFLE_BUFFER_SIZE = 5000  # Size of buffer in dataset to shuffle


@dataclasses.dataclass(frozen=True)
class FullySupervisedTrainingParams(train.TrainingParams):
  """Expected parameters for the base Model class."""
  embedding_layer_dropout_rate: float
  smoothing: float


class ClassifyFullySupervised(train.ClassifySemi):
  """Defines the training loop for the fully supervised baseline.

  Attributes:
    _train_data: Nested structure of tf.Tensors containing the next element of
      labeled data.
    _labeled_iterator: Iterator that returns evaluated batch of labeled data.
    ops: Output of build_model. Contains training data, labels, training and
      classification operations.
  """
  _train_data: tf.Tensor
  _labeled_iterator: Optional[train.BatchIterator] = None
  ops: train.ModelOps

  def __init__(self, params: FullySupervisedTrainingParams, train_dir: str,
               dataset: prepare_ssl_data.SSLDataset):
    super().__init__(params, train_dir=train_dir, dataset=dataset)
    if not params.inference_mode:
      self._next_labeled_batch = self._dataset.train_labeled.repeat().shuffle(
          _SHUFFLE_BUFFER_SIZE).batch(params.batch).prefetch(
              16).make_one_shot_iterator().get_next()
    self._labeled_iterator = None
    self.ops = self.build_model(params)
    self.ops.update_step = tf.assign_add(self._step, params.batch)

  def prepare_to_train(self, session: tf.Session):
    """Populates `_iterator` class variable with expected iterator over labeled data."""
    self._labeled_iterator = train.BatchIterator(
        next_batch=self._next_labeled_batch, session=session)

  def train_step(self, train_session: tf.train.MonitoredTrainingSession) -> int:
    """Gets internal iterator and runs next batch through training operations."""
    labeled_batch = next(self._labeled_iterator)
    return train_session.run(
        [self.ops.train_op, self.ops.update_step],
        feed_dict={
            self.ops.xt: labeled_batch[prepare_ssl_data.IMAGE_KEY],
            self.ops.label: labeled_batch[prepare_ssl_data.LABEL_KEY]
        })[1]
