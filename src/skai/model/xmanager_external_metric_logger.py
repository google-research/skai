"""Keras callback for logging metrics to XManager."""

from log_metrics_callback import MetricLogger
from xmanager.vizier.vizier_cloud import vizier_worker 

class XManagerMetricLogger(MetricLogger):
  """Class for logging metrics to XManager."""

  def __init__(self, trial_name: str = None) -> None:
    self.trial_name = trial_name
    self.worker = vizier_worker.VizierWorker(trial_name)

  def log_scalar_metric(
      self,
      metric_label: str,
      metric_value: float | int,
      step: int,
      is_val_metric: bool,
  ) -> None:
    xm_label = metric_label + '_val' if is_val_metric else metric_label
    if xm_label == 'epoch_main_aucpr_1_vs_rest_val':
      self.worker.add_trial_measurement(step, {xm_label: metric_value})