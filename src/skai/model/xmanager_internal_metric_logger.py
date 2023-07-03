"""Keras callback for logging metrics to XManager."""
from typing import Union

from google3.learning.deepmind.xmanager2.client import xmanager_api
from log_metrics_callback import MetricLogger

_ScalarMetric = Union[float, int]


class XManagerMetricLogger(MetricLogger):
    """Class for logging metrics to XManager."""

    def __init__(self, xmanager_work_unit: xmanager_api.WorkUnit) -> None:
        self._work_unit = xmanager_work_unit

    def log_scalar_metric(
        self,
        metric_label: str,
        metric_value: _ScalarMetric,
        step: int,
        is_val_metric: bool,
    ) -> None:
        xm_label = metric_label + "_val" if is_val_metric else metric_label
        measurements = self._work_unit.get_measurement_series(label=xm_label)
        measurements.create_measurement(metric_value, step=step)
