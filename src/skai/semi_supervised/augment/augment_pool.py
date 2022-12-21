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

"""Wrapper around train.BatchIterator that augments data in parallel.
"""

import abc
import dataclasses
import functools
import multiprocessing
from typing import Iterable, List, Optional, Tuple

import numpy as np
from skai.semi_supervised import train
from skai.semi_supervised import utils
from skai.semi_supervised.augment import ctaugment
from skai.semi_supervised.dataloader import prepare_ssl_data
import tensorflow.compat.v1 as tf


QUEUE_LENGTH = 4  # Number of pool entries to queue

RANDOM_POLICY_OPS = ('Identity', 'AutoContrast', 'Equalize', 'Rotate',
                     'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
                     'ShearX', 'TranslateX', 'TranslateY', 'Posterize',
                     'ShearY')


@dataclasses.dataclass
class _AugmentedData:
  augmented_image: np.ndarray
  # 'policy', 'probed_image' used by CTA labeled data
  policy: Optional[Iterable[ctaugment.OP]] = None
  probed_image: Optional[np.ndarray] = None


@dataclasses.dataclass
class _AugmentedDataBatch:
  augmented_image_batch: np.ndarray
  # 'label_batch', 'policy_batch', 'probed_image_batch' used by CTA labeled data
  label_batch: Optional[np.ndarray] = None
  policy_batch: Optional[np.ndarray] = None
  probed_image_batch: Optional[np.ndarray] = None


@dataclasses.dataclass
class _PoolEntry:
  augmented_data: multiprocessing.pool.IMapIterator
  label: np.ndarray


@dataclasses.dataclass
class _PolicyAction:
  op: ctaugment.OP
  probability: float


def _numpy_apply_policies_cta(data: np.ndarray, cta: ctaugment.CTAugment,
                              probe: bool) -> _AugmentedData:
  """Numpy augmentation policy, which includes CTAugment and Cutout."""
  assert (data.ndim == 3 and probe) or (data.ndim == 4 and not probe), (
      f'Data has {data.ndim} dimensions, but probe=={probe}.')
  policy = cta.policy(probe=probe)
  if data.ndim == 3:  # labeled data (h, w, c)
    return _AugmentedData(
        augmented_image=data,
        policy=policy,
        probed_image=ctaugment.apply(data, policy, cutout=False))
  # unlabeled data (2, h, w, c)
  weak = data[0]
  strong = data[1:]  # Should only be one
  strong_augments = [ctaugment.apply(y, policy, cutout=True) for y in strong]
  return _AugmentedData(
      augmented_image=np.stack([weak] + strong_augments).astype('f'))


def _apply_policy(
    img: np.ndarray,
    policy_actions: Optional[Iterable[_PolicyAction]]) -> np.ndarray:
  """For RandAugment. Perform operation at rate given by probability."""
  if policy_actions is None:
    return img
  ops_to_perform = []
  for policy_action in policy_actions:
    if np.random.random_sample() <= policy_action.probability:
      ops_to_perform.append(policy_action.op)
  return ctaugment.apply(img, ops_to_perform, cutout=True)


def _numpy_apply_policies_ra(
    args: List[Tuple[np.ndarray, Optional[Iterable[_PolicyAction]]]]
) -> _AugmentedData:
  return _AugmentedData(
      augmented_image=np.stack(
          [_apply_policy(img, policy_actions)
           for img, policy_actions in args]).astype('f'))


class AugmentPool(abc.ABC, train.BatchIterator):
  """A multiprocessing base class for running CTAugment and RandAugment.

  Attributes:
    _pool: Pool object to allow parallel augmentation.
    _queue: Samples to be augmented.
  """

  def __init__(self, next_batch: tf.Tensor,
               session: tf.Session,
               num_parallel_calls: int):
    train.BatchIterator.__init__(self, next_batch=next_batch, session=session)
    parallelism = max(1, len(
        utils.get_available_gpus())) * num_parallel_calls
    self._pool = multiprocessing.Pool(parallelism)
    self._queue = []
    self._fill_queue()

  def __iter__(self):
    return self

  @abc.abstractmethod
  def __next__(self) -> _AugmentedDataBatch:  # pytype: disable=signature-mismatch  # overriding-return-type-checks
    raise NotImplementedError()

  @abc.abstractmethod
  def _queue_images(self):
    raise NotImplementedError()

  def _fill_queue(self):
    for _ in range(QUEUE_LENGTH):
      self._queue_images()

  def _get_next_entry(self) -> _PoolEntry:
    return self._queue.pop(0)


class AugmentPoolCTA(AugmentPool):
  """A multiprocessing class for running CTAugment, performs cutout by default.

    Attributes:
      _queue: Queue of samples that will be added to the multiprocessing pool.
      _cta: CTAugment instance that maintains policy.
      _probe: Boolean that, when true, uses example to update policy.
  """

  def __init__(self, next_batch: tf.Tensor,
               session: tf.Session,
               num_parallel_calls: int,
               cta: ctaugment.CTAugment,
               probe: bool):
    self._cta = cta
    self._probe = probe
    super().__init__(
        next_batch=next_batch,
        session=session,
        num_parallel_calls=num_parallel_calls)

  def _queue_images(self):
    numpy_apply_policies_cta_partial = functools.partial(
        _numpy_apply_policies_cta, cta=self._cta, probe=self._probe)

    batch = train.BatchIterator.__next__(self)
    self._queue.append(
        _PoolEntry(
            augmented_data=self._pool.imap(numpy_apply_policies_cta_partial,
                                           batch[prepare_ssl_data.IMAGE_KEY]),
            label=batch[prepare_ssl_data.LABEL_KEY]))

  def __next__(self) -> _AugmentedDataBatch:
    """Returns _AugmentedDataBatch with images, with policy/probe if labeled."""
    try:
      entry = self._get_next_entry()
    except IndexError as index_error:
      raise StopIteration from index_error
    samples = list(entry.augmented_data)
    augmented_image_batch = np.stack(x.augmented_image for x in samples)
    label_batch = np.stack(label for label in list(entry.label))
    augmented_data_batch = _AugmentedDataBatch(
        augmented_image_batch=augmented_image_batch, label_batch=label_batch)
    if samples[0].probed_image is not None:
      augmented_data_batch.policy_batch = np.stack(x.policy for x in samples)
      augmented_data_batch.probed_image_batch = np.stack(
          x.probed_image for x in samples)
    self._queue_images()
    return augmented_data_batch


class AugmentPoolRAMC(AugmentPool):
  """AugmentPool for RandAugment followed by Cutout.

  Attributes:
    _nops: Number of operations.
    _queue: Queue of samples that will be added to the multiprocessing pool.
  """

  def __init__(self,
               next_batch: tf.Tensor,
               session: tf.Session,
               num_parallel_calls: int,
               nops: int = 2):
    self._nops = nops
    super().__init__(
        next_batch=next_batch,
        session=session,
        num_parallel_calls=num_parallel_calls)

  def _generate_ra_policy(self) -> List[_PolicyAction]:
    """Randomly select augmentations to perform and their severity levels."""
    policy = []
    operations = list(ctaugment.OPS.keys())
    for op in np.random.choice(operations, self._nops):
      operation = ctaugment.OPS[op]
      levels = [
          np.random.randint(1, max_level) / float(max_level)
          for max_level in operation.bins
      ]
      policy.append(_PolicyAction(ctaugment.OP(op, levels), 0.5))
    return policy

  def _queue_images(self):
    """Queue images as PoolEntries to be augmented."""
    args = []
    batch = train.BatchIterator.__next__(self)
    all_img = batch[prepare_ssl_data.IMAGE_KEY]
    if all_img.ndim == 4:  # labeled data (n, h, w, c)
      args = [[(img, self._generate_ra_policy())] for img in all_img]
    else:  # unlabeled data (n, 2, h, w, c)
      for unlabeled_imgs in all_img[:]:
        args_weak = [(unlabeled_imgs[0], None)]  # no strong augment for 1st img
        args_strong = [
            (img, self._generate_ra_policy()) for img in unlabeled_imgs[1:]
        ]  # unlabeled_imgs[1:] has size (1, h, w, c)
        args.append(args_weak + args_strong)
    self._queue.append(
        _PoolEntry(
            augmented_data=self._pool.imap(_numpy_apply_policies_ra, args),
            label=batch[prepare_ssl_data.LABEL_KEY]))

  def __next__(self) -> _AugmentedDataBatch:
    """Returns _AugmentedDataBatch only with images and labels."""
    try:
      entry = self._get_next_entry()
    except IndexError as index_error:
      raise StopIteration from index_error
    samples = list(entry.augmented_data)
    augmented_image_batch = np.stack(x.augmented_image for x in samples)
    self._queue_images()
    return _AugmentedDataBatch(augmented_image_batch=augmented_image_batch)
