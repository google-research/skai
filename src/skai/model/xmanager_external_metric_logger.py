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

"""Keras callback for logging metrics to XManager."""
import os
from skai.model import log_metrics_callback

import tensorflow as tf
from xmanager.vizier.vizier_cloud import vizier_worker


class XManagerMetricLogger(log_metrics_callback.MetricLogger):
  """Class for logging metrics to XManager."""

  def __init__(self, trial_name: str | None, output_dir: str | None) -> None:
    """Constructor for XManagerMetricLogger.

    Args:
      trial_name: A string containing the Vizier trial name. None if
        running on local machine.
      output_dir: Directory where Tensorboard metrics will be logged when
      running locally.
    """
    self.trial_name = trial_name
    if trial_name:
      self.worker = vizier_worker.VizierWorker(trial_name)
    else:  # Local run. Write to Tensorboard.
      self._train_summary_writer = (
          tf.summary.create_file_writer(
              os.path.join(output_dir, 'tensorboard', 'train')
          ),
      )
      self._val_summary_writer = (
          tf.summary.create_file_writer(
              os.path.join(output_dir, 'tensorboard', 'val')
          ),
      )

  def log_scalar_metric(
      self,
      metric_label: str,
      metric_value: float | int,
      step: int,
      is_val_metric: bool,
  ) -> None:
    if self.trial_name:
      xm_label = metric_label + '_val' if is_val_metric else metric_label
      if xm_label == 'epoch_main_aucpr_1_vs_rest_val':
        self.worker.add_trial_measurement(step, {xm_label: metric_value})
    else:
      with self._get_summary_writer(is_val_metric)[0].as_default():
        tf.summary.scalar(metric_label, metric_value, step=step)

  def _get_summary_writer(
      self, get_val_writer: bool
  ) -> tf.summary.SummaryWriter:
    return (
        self._val_summary_writer
        if get_val_writer
        else self._train_summary_writer
    )
