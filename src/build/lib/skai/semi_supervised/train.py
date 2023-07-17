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

"""Training setup for semi-supervised learning models.

This module provides a base class for our models, defining the training and
evaluation steps as well as model properties.

The use case of inference mode is to make predictions on examples from an
unlabeled region. Only test data is expected (no training). The test data is
not expected to have ground truth labels. If the test data happens to have
valid labels, then the eval_checkpoint and eval_stats functions will log
reliable metrics. If not, there is a statement logged that the metrics should
be ignored in inference mode as a warning.
"""

import abc
import dataclasses
import functools
import json
import os.path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from skai.semi_supervised import classifiers
from skai.semi_supervised import utils
from skai.semi_supervised.dataloader import prepare_ssl_data
from sklearn.metrics import roc_auc_score
import tensorflow.compat.v1 as tf
import tqdm as tqdm_lib


_SHUFFLE_BUFFER_SIZE = 5000  # Size of buffer in dataset to shuffle
# Note: The model checkpoints and logs must be saved at pre-specified locations
# set by Vertex AI. Expectations are provided at:
# https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec.
_ARGS_DIRNAME = 'args'
_MODEL_AND_CHECKPOINT_DIRNAME = 'checkpoints'
_TENSORBOARD_LOGS_DIRNAME = 'logs'

# Arrays of the images, labels, latitudes, and longitudes for a given dataset.
CacheType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

tf.disable_v2_behavior()


@dataclasses.dataclass
class ModelOps:
  """The data and operations to perform during training.

  Each model class (FullySupervised, MixMatch, and FixMatch) has a build_model
  that implements the overall approach.
  This class specifies the operations returned from build_model().

  Attributes:
    x: Eval images.
    label: Labels.
    train_op: Optimizier for training.
    classify_raw: Output of tf.nn.softmax(classifier) with no EMA.
    classify_op: Output of tf.nn.softmax(classifier) with EMA.
    xt: Labeled training data.
    y: Unlabeled training data.
    update_step: Global step tensor.
    tune_op: Operations to retrain only batchnorm during tuning step.
  """
  x: np.ndarray
  label: np.ndarray
  train_op: tf.train.Optimizer
  classify_raw: tf.Tensor
  classify_op: tf.Tensor
  xt: Optional[np.ndarray] = None
  y: Optional[np.ndarray] = None
  update_step: Optional[tf.Tensor] = None
  tune_op: Optional[List[str]] = None


@dataclasses.dataclass
class Cache:
  """Temporarily stores batch in memory to get accuracy statistics."""
  test: Optional[CacheType] = None
  train_unlabeled: Optional[CacheType] = None
  train_labeled: Optional[CacheType] = None


@dataclasses.dataclass
class PerformanceMetrics:
  """Stores metric, i.e. accuracy or AUC, by dataset."""
  train_labeled_metric: Optional[float] = None
  train_unlabeled_metric: Optional[float] = None
  test_metric: Optional[float] = None


@dataclasses.dataclass
class PredictionsWithCoordinates:
  preds: Optional[np.ndarray] = np.array([])
  lats: Optional[np.ndarray] = np.array([])
  lons: Optional[np.ndarray] = np.array([])


@dataclasses.dataclass
class PredictionsWithCoordinatesPerDataset:
  """Stores predictions and corresponding lat/lon coordinates by dataset."""
  train_label_preds_coords: Optional[PredictionsWithCoordinates] = None
  train_unlabel_preds_coords: Optional[PredictionsWithCoordinates] = None
  test_preds_coords: Optional[PredictionsWithCoordinates] = None


# Accuracies, AUCs, and predictions
EvaluationType = Tuple[PerformanceMetrics, PerformanceMetrics,
                       PredictionsWithCoordinatesPerDataset]


@dataclasses.dataclass(frozen=True)
class TrainingParams:
  """Expected parameters for the base Model class."""
  batch: int
  nclass: int
  lr: float
  ema: float
  weight_decay: float
  arch: str
  scales: int
  conv_filter_size: int
  num_residual_repeat_per_stage: int
  inference_mode: bool

  @functools.lru_cache(maxsize=3)
  def to_dict(self) -> Dict[str, Any]:
    return dataclasses.asdict(self)

  def print(self):
    for k, v in sorted(self.to_dict().items()):
      print('%-32s %s' % (k, v))

  def save_args(self, arg_dir: str):
    with tf.gfile.Open(os.path.join(arg_dir, 'args.json'), 'w') as f:
      json.dump(self.to_dict(), f, sort_keys=True, indent=4)


class _TrainLogger:
  """Logs accuracy statistics during training.

  Attributes:
    cache: Memoized data of current batch to compute accuracy stats.
    _print_queue: Queue of strings, e.g. accuracy stats, printed during
      training.
  """

  def __init__(self):
    self.cache = None
    self._print_queue = []

  def write_during_train_loop(self, loop: tqdm_lib.tqdm):
    """Provides thread-safe way to log while tqdm_lib.trange prints progress."""
    while self._print_queue:
      loop.write(self._print_queue.pop(0))

  def write_out_queue(self):
    """Prints out remaining elements in print_queue."""
    while self._print_queue:
      print(self._print_queue.pop(0))

  def add_to_print_queue(self, text: str):
    self._print_queue.append(text)


class BatchIterator:
  """Iterator that evaluates batches of data using given session.

  Attributes:
    _next_batch: Nested structure of tf.Tensors containing the next element of
      data.
    _session: Session to evaluate batch of data.
  """

  def __init__(self, next_batch: tf.Tensor, session: tf.Session):
    self._next_batch = next_batch
    self._session = session

  def __iter__(self):
    return self

  def __next__(self) -> Dict[str, np.ndarray]:
    try:
      batch = self._session.run(self._next_batch)
    except IndexError as index_error:
      raise StopIteration from index_error
    return batch


class Model(abc.ABC):
  """Model base class, supports for loading, training, evaluating models.

  Attributes:
    train_dir: Directory to save training checkpoints.
    inference_mode: Inference mode (no training, only test data available).
    arg_dir: Subdirectory of train_dir that saves training parameters.
    checkpoint_dir: Subdirectory of train_dir that saves model checkpoints.
    tensorboard_dir: Subdirectory of train_dir that saves Tensorboard events.
    _batch: Batch size.
    _dataset: SSLDataset containing training and evaluation data.
    _logger: Logger to print accuracy stats during training.
    _evaluated_step: Evaluated tf.Tensor value of global step for printing.
    _step: tf.Tensor of global step.
  """

  _dataset: prepare_ssl_data.SSLDataset
  _batch: int
  _logger: _TrainLogger
  _evaluated_step: int
  _step: tf.Variable

  def __init__(self, params: TrainingParams, train_dir: str,
               dataset: prepare_ssl_data.SSLDataset):
    self._dataset = dataset
    self._train_dir = train_dir
    self.inference_mode = params.inference_mode
    self._batch = params.batch
    self._logger = _TrainLogger()
    self._evaluated_step = None
    self._step = tf.train.get_or_create_global_step()
    self._print_model_config(params)

  def _print_model_config(self, params: TrainingParams):
    """Print model configuration info, i.e. training directory and params."""
    print(' Config '.center(80, '-'))
    print('train_dir', self._train_dir)
    print('%-32s %s' % ('Model', self.__class__.__name__))
    print('%-32s %s' % ('Dataset', self._dataset.name))
    params.print()
    print(' Model '.center(80, '-'))

    def format_var(v):
      return tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)])

    to_print = [format_var(v) for v in utils.model_vars(None)]
    to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
    sizes = []
    for i in range(3):
      sizes.append(max([len(x[i]) for x in to_print]))
    fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
    for x in to_print[:-1]:
      print(fmt % x)
    print()
    print(fmt % to_print[-1])
    print('-' * 80)

    self._create_initial_files(params)

  @property
  def train_dir(self):
    return self._train_dir

  @property
  def arg_dir(self):
    return os.path.join(self._train_dir, _ARGS_DIRNAME)

  @property
  def checkpoint_dir(self):
    return os.path.join(self._train_dir, _MODEL_AND_CHECKPOINT_DIRNAME)

  @property
  def tensorboard_dir(self):
    return os.path.join(self._train_dir, _TENSORBOARD_LOGS_DIRNAME)

  def _create_initial_files(self, params: TrainingParams):
    for folder in (self.checkpoint_dir, self.arg_dir):
      if not tf.gfile.IsDirectory(folder):
        tf.gfile.MakeDirs(folder)
    params.save_args(self.arg_dir)

  @classmethod
  def load(cls: Type['Model'], train_dir: str):
    with open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
      params = json.load(f)
    instance = cls(train_dir=train_dir, **params)
    return instance

  def experiment_name(self, params: TrainingParams):
    args = [x + str(y) for x, y in sorted(params.to_dict().items())]
    return os.path.join(self.__class__.__name__, '_'.join(args))

  def load_checkpoint(self,
                      session: tf.Session,
                      ckpt: str,
                      saver: Optional[tf.train.Saver] = None):
    """Load checkpoint to new Saver or existing scaffold's Saver for finetuning."""
    if saver is None:
      saver = tf.train.Saver()
    saver.restore(session, ckpt)
    self._evaluated_step = session.run(self._step)
    print('Eval model %s at global_step %d' %
          (self.__class__.__name__, self._evaluated_step))

  @abc.abstractmethod
  def build_model(self, params: TrainingParams) -> ModelOps:
    """Builds the model graph."""
    raise NotImplementedError()


class ClassifySemi(Model):
  """Semi-supervised classification.

  Attributes:
    classifier: Final part of model that returns logits.
  """

  classifier: classifiers.Classifier

  def __init__(self, params: TrainingParams, train_dir: str,
               dataset: prepare_ssl_data.SSLDataset):
    if params.arch not in classifiers.SUPPORTED_ARCHITECTURES:
      raise ValueError('Model %s does not exist, available ones are %s' %
                       (params.arch, classifiers.SUPPORTED_ARCHITECTURES))
    classifier_model = classifiers.ResNetClassifier if params.arch == classifiers.RESNET else classifiers.ShakeNetClassifier
    self.classifier = classifier_model(
        dataset=dataset,
        nclass=params.nclass,
        scales=params.scales,
        conv_filter_size=params.conv_filter_size,
        num_residual_repeat_per_stage=params.num_residual_repeat_per_stage)

    super().__init__(params, train_dir=train_dir, dataset=dataset)

  def classify(
      self,
      x: np.ndarray,
      training: Optional[bool],
      logit_norm: Optional[bool] = False,
      getter: Optional[tf.train.ExponentialMovingAverage] = None) -> tf.Tensor:
    """Runs classify function based on architecture to get output logits."""
    logits = self.classifier.classify(x=x, training=training, getter=getter)
    if logit_norm:
      logits *= tf.rsqrt(tf.reduce_mean(tf.square(logits)) + 1e-8)
    return logits

  @abc.abstractmethod
  def prepare_to_train(self, session: tf.Session):
    """Performs one-time initialization before training loop is run.

    Graph is already frozen by the time this function is called in `train`.

    Args:
      session: Session used to evaluate data batches.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def train_step(self, train_session: tf.train.MonitoredTrainingSession) -> int:
    """Runs training operations and returns next evaluated step."""
    raise NotImplementedError()

  def train(self,
            batch: int,
            train_nimg: int,
            save_nimg: int,
            keep_ckpt: int,
            finetune_ckpt: Optional[str] = ''):
    """Defines and runs training loop of the model."""

    def _init_fn(scaffold: tf.train.Scaffold, session: tf.Session):
      """Loads existing checkpoint to continue training."""
      if finetune_ckpt:
        self.load_checkpoint(session, finetune_ckpt, scaffold.saver)

    scaffold = tf.train.Scaffold(
        init_fn=_init_fn,
        saver=tf.train.Saver(max_to_keep=keep_ckpt, pad_step_number=10))
    sess = tf.Session(config=utils.get_config())
    self.cache_eval(sess)

    report_nimg = save_nimg
    with tf.train.MonitoredTrainingSession(
        scaffold=scaffold,
        checkpoint_dir=self.checkpoint_dir,
        config=utils.get_config(),
        save_checkpoint_steps=save_nimg,
        save_summaries_steps=report_nimg - batch) as train_session:
      # To avoid generating summary stats, use underlying session to evaluate.
      unwrapped_session = train_session._tf_sess()  # pylint: disable=protected-access
      self.prepare_to_train(session=unwrapped_session)
      self._summary_session = unwrapped_session
      self._evaluated_step = unwrapped_session.run(self._step)

      while self._evaluated_step < train_nimg:
        loop = tqdm_lib.trange(
            self._evaluated_step % report_nimg,
            report_nimg,
            batch,
            leave=False,
            unit_scale=batch,
            desc='Epoch %d/%d' %
            (1 +
             (self._evaluated_step // report_nimg), train_nimg // report_nimg))
        for _ in loop:
          self._evaluated_step = self.train_step(train_session)
          self._logger.write_during_train_loop(loop)
      self._logger.write_out_queue()

  def eval_checkpoint(self, ckpt: str) -> Tuple[EvaluationType, EvaluationType]:
    """Load and evaluate a checkpoint."""
    session = tf.Session(config=utils.get_config())
    self.load_checkpoint(session=session, ckpt=ckpt)
    self.cache_eval(session)
    raw_acc, raw_auc, raw_preds = self.eval_stats(
        classify_op=self.ops.classify_raw, session=session)
    ema_acc, ema_auc, ema_preds = self.eval_stats(
        classify_op=self.ops.classify_op, session=session)
    if self.inference_mode:
      # Print only the test set metrics
      print('Model is in inference mode. If the test data does not have ground '
            'truth labels, then the following metrics will not be valid.')
      print('%16s %8s' % ('', 'test'))
      print('%16s %8s' % ('raw_acc', '%.2f' % raw_acc.test_metric))
      print('%16s %8s' % ('ema_acc', '%.2f' % ema_acc.test_metric))
      print('%16s %8s' % ('raw_auc', '%.2f' % raw_auc.test_metric))
      print('%16s %8s' % ('ema_auc', '%.2f' % ema_auc.test_metric))
    else:
      # Print metrics for all three datasets
      print('%16s %8s %8s %8s' % ('', 'train_label', 'train_unlabel', 'test'))
      print('%16s %8s %8s %8s' % (('raw_acc',) + tuple('%.2f' % x for x in [
          raw_acc.train_labeled_metric, raw_acc.train_unlabeled_metric,
          raw_acc.test_metric
      ])))
      print('%16s %8s %8s %8s' % (('ema_acc',) + tuple('%.2f' % x for x in [
          ema_acc.train_labeled_metric, ema_acc.train_unlabeled_metric,
          ema_acc.test_metric
      ])))
      print('%16s %8s %8s %8s' % (('raw_auc',) + tuple('%.2f' % x for x in [
          raw_auc.train_labeled_metric, raw_auc.train_unlabeled_metric,
          raw_auc.test_metric
      ])))
      print('%16s %8s %8s %8s' % (('ema_auc',) + tuple('%.2f' % x for x in [
          ema_auc.train_labeled_metric, ema_auc.train_unlabeled_metric,
          ema_auc.test_metric
      ])))
    return (raw_acc, raw_auc, raw_preds), (ema_acc, ema_auc, ema_preds)

  def cache_eval(self, session: tf.Session):
    """Cache datasets for computing eval stats."""

    def collect_samples(dataset):
      """Return numpy arrays of all the samples from a dataset."""
      it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
      images, labels, lons, lats = [], [], [], []
      while 1:
        try:
          v = session.run(it)
        except tf.errors.OutOfRangeError:
          break
        images.append(v[prepare_ssl_data.IMAGE_KEY])
        labels.append(v[prepare_ssl_data.LABEL_KEY])
        lons.append(v[prepare_ssl_data.COORDS_KEY][0][0])
        lats.append(v[prepare_ssl_data.COORDS_KEY][0][1])

      lons = np.asarray(lons)
      lats = np.asarray(lats)

      if images:
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
      else:  # may be empty since fully supervised does not use unlabeled data
        images = np.array(images)
        labels = np.array(labels)
      return images, labels, lons, lats

    if self._logger.cache is None:
      train_labeled = [] if self.inference_mode else collect_samples(
          self._dataset.eval_labeled)
      train_unlabeled = [] if self.inference_mode else collect_samples(
          self._dataset.unlabeled_validation_examples)
      self._logger.cache = Cache(
          test=collect_samples(self._dataset.test),
          train_unlabeled=train_unlabeled,
          train_labeled=train_labeled)

  def _get_preds_labels_coords(
      self, cached_dataset: CacheType, session: tf.Session, batch: int,
      classify_op: tf.Tensor) -> Tuple[PredictionsWithCoordinates, np.ndarray]:
    """Tetrieve predictions, labels, and coordinates of cached dataset."""
    images, labels, lons, lats = cached_dataset
    preds = []

    total_num_examples = images.shape[0]
    total_num_batches = int(np.ceil(total_num_examples / batch))
    eval_loop = tqdm_lib.trange(0, total_num_examples, batch, unit_scale=batch)
    for batch_start in eval_loop:
      p = session.run(
          classify_op,
          feed_dict={self.ops.x: images[batch_start:batch_start + batch]})
      preds.append(p)
      eval_loop.set_description(
          f'Batch {1 + int(batch_start / batch)}/{total_num_batches}')
      self._logger.write_during_train_loop(eval_loop)

    preds = np.concatenate(preds, axis=0)
    preds_with_coords = PredictionsWithCoordinates(
        preds=preds, lons=lons, lats=lats)
    return preds_with_coords, labels

  def _get_acc_auc(self, preds: np.ndarray,
                   labels: np.ndarray) -> Tuple[float, float]:
    """Given predictions and labels, return accuracy and AUC values."""
    acc = (preds.argmax(1) == labels).mean() * 100
    if self._dataset.nclass > 2:
      # Multi-class classification.
      scores = preds  # (batch, num_classes)
    else:
      # Binary classification.
      scores = np.take_along_axis(preds, labels[:, None], axis=1)  # (batch, 1)
      scores = np.squeeze(scores, axis=1)  # (batch,)
      if len(scores.shape) != 1:
        raise ValueError(
            'Scores must have same single-dimensional shape (batch-size) to '
            'calculate metrics in binary classification case.'
            f'Scores: {scores.shape}')
    if scores.shape[0] != labels.shape[0]:
      raise ValueError(
          'Scores and Labels must have same number of elements (batch-size).',
          f'Scores: {scores.shape}, Labels: {labels.shape}')
    auc = 0
    if preds.shape[0] > 1 and not np.all(labels == labels[0]):
      # First condition checks if the batch is non-empty.
      # Second condition checks that there is more than one class in labels.
      # If there is only one class, roc_auc_score returns a ValueError that
      # the score is undefined in this case.
      auc = roc_auc_score(labels, scores, multi_class='ovr')
    return acc, auc

  def eval_stats(self,
                 session: tf.Session,
                 batch: Optional[int] = None,
                 classify_op: Optional[tf.Tensor] = None) -> EvaluationType:
    """Evaluate model on train_labeled, train_unlabeled, and test sets.

    Args:
      session: Session to evaluate data.
      batch: Batch size.
      classify_op: Classification operation to get logits.

    Returns:
      Accuracy on train, validation, and test sets.
    """
    batch = batch or self._batch
    classify_op = self.ops.classify_op if classify_op is None else classify_op
    preds_per_dataset = PredictionsWithCoordinatesPerDataset()
    accuracies = PerformanceMetrics()
    aucs = PerformanceMetrics()

    # Test data is non-empty in both eval and training mode
    self._logger.add_to_print_queue('Evaluating batches of test set...')
    if self.inference_mode:
      self._logger.add_to_print_queue('Model is in inference mode. If the test '
                                      'data does not have ground truth labels, '
                                      'then these metrics will not be valid.')
    preds_with_coords, labels = self._get_preds_labels_coords(
        self._logger.cache.test,
        session=session,
        batch=batch,
        classify_op=classify_op)
    acc, auc = self._get_acc_auc(preds_with_coords.preds, labels)
    preds_per_dataset.test_preds_coords = preds_with_coords
    accuracies.test_metric = acc
    aucs.test_metric = auc
    self._logger.add_to_print_queue(
        f'kimg {self._evaluated_step >> 10} test acc {acc}')
    self._logger.add_to_print_queue(
        f'kimg {self._evaluated_step >> 10} test AUC {auc}')

    if not self.inference_mode:
      # Get predictions and metrics for train_labeled
      self._logger.add_to_print_queue(
          'Evaluating batches of train_labeled set...')
      preds_with_coords, labels = self._get_preds_labels_coords(
          self._logger.cache.train_labeled,
          session=session,
          batch=batch,
          classify_op=classify_op)
      preds_per_dataset.train_label_preds_coords = preds_with_coords
      acc, auc = self._get_acc_auc(preds_with_coords.preds, labels)
      accuracies.train_labeled_metric = acc
      aucs.train_labeled_metric = auc
      self._logger.add_to_print_queue(
          f'kimg {self._evaluated_step >> 10} train_labeled acc {acc}')
      self._logger.add_to_print_queue(
          f'kimg {self._evaluated_step >> 10} train_labeled AUC {auc}')

      # Get predictions and metrics for train_unlabeled
      self._logger.add_to_print_queue(
          'Evaluating batches of train_unlabeled set...')
      preds_with_coords, labels = self._get_preds_labels_coords(
          self._logger.cache.train_unlabeled,
          session=session,
          batch=batch,
          classify_op=classify_op)
      preds_per_dataset.train_unlabel_preds_coords = preds_with_coords
      acc, auc = self._get_acc_auc(preds_with_coords.preds, labels)
      accuracies.train_unlabeled_metric = acc
      aucs.train_unlabeled_metric = auc
      self._logger.add_to_print_queue(
          f'kimg {self._evaluated_step >> 10} train_unlabeled acc {acc}')
      self._logger.add_to_print_queue(
          f'kimg {self._evaluated_step >> 10} train_unlabeled AUC {auc}')

    if not preds_per_dataset.test_preds_coords.preds.shape[
        0] == preds_per_dataset.test_preds_coords.lons.shape[
            0] == preds_per_dataset.test_preds_coords.lats.shape[0]:
      raise ValueError('Prediction, latitude, and longitude arrays should have',
                       'same number of elements.')
    return accuracies, aucs, preds_per_dataset
