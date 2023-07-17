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

"""Defines model options and provides functions to return initialized models.

This file is used in ssl_train.py and ssl_eval.py to provide model options and
return initialized models.
"""

from typing import Any, Dict

from skai.semi_supervised import fixmatch
from skai.semi_supervised import fully_supervised
from skai.semi_supervised import fully_supervised_baseline
from skai.semi_supervised import mixmatch
from skai.semi_supervised.dataloader import prepare_ssl_data


def _create_mixmatch(dataset: prepare_ssl_data.SSLDataset, log_width: int,
                     params: Dict[str, Any]):
  """Initialize training parameters and model for MixMatch."""
  mixmatch_params = mixmatch.MixMatchTrainingParams(
      lr=params['lr'],
      weight_decay=params['weight_decay'],
      arch=params['arch'],
      batch=params['batch'],
      nclass=dataset.nclass,
      ema=params['ema'],
      beta=params['beta'],
      logit_norm=params['logit_norm'],
      sharpening_temperature=params['sharpening_temperature'],
      mixup_mode=params['mixup_mode'],
      num_augmentations=params['num_augmentations'],
      dbuf=params['dbuf'],
      w_match=params['w_match'],
      warmup_kimg=params['warmup_kimg'],
      scales=params['scales'] or (log_width - 2),
      conv_filter_size=params['conv_filter_size'],
      num_residual_repeat_per_stage=params['num_residual_repeat_per_stage'],
      inference_mode=params['inference_mode'])
  return mixmatch.MixMatch(
      mixmatch_params,
      train_dir=params['train_dir'],
      dataset=dataset)


def _create_fixmatch(dataset: prepare_ssl_data.SSLDataset, log_width: int,
                     params: Dict[str, Any]):
  """Initialize training parameters and model for FixMatch."""
  fixmatch_params = fixmatch.FixMatchTrainingParams(
      lr=params['lr'],
      weight_decay=params['weight_decay'],
      arch=params['arch'],
      batch=params['batch'],
      nclass=dataset.nclass,
      ema=params['ema'],
      pseudo_label_loss_weight=params['pseudo_label_loss_weight'],
      confidence=params['confidence'],
      unlabeled_ratio=params['unlabeled_ratio'],
      scales=params['scales'] or (log_width - 2),
      conv_filter_size=params['conv_filter_size'],
      num_residual_repeat_per_stage=params['num_residual_repeat_per_stage'],
      num_parallel_calls=params['num_parallel_calls'],
      inference_mode=params['inference_mode'])
  fixmatch_class = fixmatch.StrategyOptions[
      params['augmentation_strategy']].value
  return fixmatch_class(
      fixmatch_params,
      train_dir=params['train_dir'],
      dataset=dataset)


def _create_fully_supervised(dataset: prepare_ssl_data.SSLDataset,
                             log_width: int, params: Dict[str, Any]):
  """Initialize training parameters and model for Fully Supervised Baseline."""
  fully_supervised_params = fully_supervised.FullySupervisedTrainingParams(
      lr=params['lr'],
      weight_decay=params['weight_decay'],
      arch=params['arch'],
      batch=params['batch'],
      nclass=dataset.nclass,
      ema=params['ema'],
      embedding_layer_dropout_rate=params['embedding_layer_dropout_rate'],
      smoothing=params['smoothing'],
      scales=params['scales'] or (log_width - 2),
      conv_filter_size=params['conv_filter_size'],
      num_residual_repeat_per_stage=params['num_residual_repeat_per_stage'],
      inference_mode=params['inference_mode'])
  return fully_supervised_baseline.FullySupervisedBaseline(
      fully_supervised_params,
      train_dir=params['train_dir'],
      dataset=dataset)


MODELS = {
    'mixmatch': _create_mixmatch,
    'fixmatch': _create_fixmatch,
    'fully_supervised': _create_fully_supervised
}


def create_model(method: str, dataset: prepare_ssl_data.SSLDataset,
                 log_width: int, params: Dict[str, Any]):
  return MODELS[method](dataset, log_width, params)
