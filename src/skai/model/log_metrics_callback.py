"""Keras callback for logging metrics to XManager."""

import abc
from typing import Mapping, Optional, Sequence, Union

import tensorflow as tf

_ScalarMetric = Union[float, int]
_MetricDict = Mapping[str, _ScalarMetric]


class MetricLogger(abc.ABC):
  """Abstract base class for logging metrics.

  `MetricLoggers` are typically used in conjunction with the
  `LogMetricsCallback`.
  """

  @abc.abstractmethod
  def log_scalar_metric(
      self, metric_label: str, metric_value: _ScalarMetric, step: int,
      is_val_metric: bool
      ) -> None:
    """Logs a metric name and value at the specified step.

    For example, to log the training accuracy at the end of the first epoch, one
    might call this function as:

      `log_scalar_metric('epoch_accuracy', 0.89, examples_per_epoch, False)`.

    Args:
      metric_label: The name of the metric being logged. Typically we assume the
        'val_' prefix for validation metrics has been removed from the metric
        label prior to being passed to this function, and that a prefix of
        'epoch_' or 'batch_' has been added to the metric label to indicate if
        this metric corresponds to an epoch or a batch. See the
        `LogMetricsCallback` for more details.
      metric_value: The value of the metric being logged.
      step: The step number at which to log this metric. Metrics are normally
        visualized on `metric_value` vs. `step` plots (e.g., on XManager or
        TensorBoard). The `LogMetricsCallback` sets this value equal to the
        number of training steps that have been seen up to the point the metric
        is logged.
      is_val_metric: A boolean specifying whether this metric was computed on a
        validation set.
    """


class LogMetricsCallback(tf.keras.callbacks.Callback):
  """A callback for logging metrics, for example to TensorBoard or XManager.

  During training, this callback will log all metrics after every training batch
  where the total number of examples seen up to that point in the training epoch
  is a multiple of the specified logging frequency, as well as at the end of
  every epoch. This callback logs metrics by invoking the `log_scalar_metric`
  function on all the metric loggers that are provided to the callback's
  constructor. The metric loggers are objects, such as `XManagerMetricLogger`
  and `TensorBoardMetricLogger`, which derive from `MetricLogger`.
  """

  def __init__(
      self,
      metric_loggers: Sequence[MetricLogger],
      logging_frequency: int,
      batch_size: int,
      num_train_examples_per_epoch: int,
      ) -> None:
    """Initializes the `LogMetricsCallback`.

    Args:
      metric_loggers: A list of `MetricLogger` objects that are invoked to log
        training/validation metrics during the course of a Keras training run.
      logging_frequency: How frequently, in terms of the number of training
        examples seen during an epoch, to log metrics. For example, if
        `logging_frequency` is 128, and batch_size is 64, then the metrics would
        get logged every other batch. `logging_frequency` must be a multiple of
        `batch_size`.
      batch_size: The batch size used during training.
      num_train_examples_per_epoch: The total number of training examples seen
        during the course of an epoch.
    """
    super().__init__()
    if not metric_loggers:
      raise ValueError('Must specify at least one MetricLogger.')
    if logging_frequency % batch_size != 0:
      raise ValueError(
          'logging_frequency must be a multiple of batch_size.'
          )
    self._metric_loggers = metric_loggers
    self._logging_frequency = logging_frequency
    self._batch_size = batch_size
    self._num_train_examples_per_epoch = num_train_examples_per_epoch
    self._epoch = -1

  def _log_metrics(
      self, logs: _MetricDict, num_examples_seen: int,
      metric_format_str: str, is_val_metric: bool = False,
      ) -> None:
    """Logs all metrics in 'logs' dictionary."""
    for metric_name, metric_value in logs.items():
      metric_label = metric_format_str.format(metric_name)
      for metric_logger in self._metric_loggers:
        metric_logger.log_scalar_metric(
            metric_label, metric_value, num_examples_seen, is_val_metric,
            )

  def on_epoch_begin(
      self, epoch: int, logs: Optional[_MetricDict] = None
      ) -> None:
    """Stores the epoch number at the beginning of every epoch."""
    self._epoch = epoch

  def on_epoch_end(
      self, epoch: int, logs: Optional[_MetricDict] = None
      ) -> None:
    """Logs all metrics at the end of every epoch."""
    num_examples_seen = (epoch + 1) * self._num_train_examples_per_epoch
    if logs:
      # Separate the train vs. validation metrics in `logs`, and log them.
      train_metrics = {metric_name: metric_value
                       for metric_name, metric_value in logs.items()
                       if not metric_name.startswith('val_')}
      val_metrics = {metric_name.replace('val_', ''): metric_value
                     for metric_name, metric_value in logs.items()
                     if metric_name.startswith('val_')}
      self._log_metrics(train_metrics, num_examples_seen, 'epoch_{}',
                        is_val_metric=False)
      self._log_metrics(val_metrics, num_examples_seen, 'epoch_{}',
                        is_val_metric=True)

  def on_train_batch_end(
      self, batch: int, logs: Optional[_MetricDict] = None
      ) -> None:
    """Logs all metrics after training batches at specified intervals."""
    num_examples_seen_in_prior_epochs = (self._epoch *
                                         self._num_train_examples_per_epoch)
    num_examples_seen_in_this_epoch = (batch + 1) * self._batch_size
    num_examples_seen_total = (num_examples_seen_in_prior_epochs +
                               num_examples_seen_in_this_epoch)
    if logs and num_examples_seen_in_this_epoch % self._logging_frequency == 0:
      # Log all the metrics in the `logs` dictionary.
      self._log_metrics(logs, num_examples_seen_total, 'batch_{}',
                        is_val_metric=False)