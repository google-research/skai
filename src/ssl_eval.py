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

r"""Continuously monitors for new model checkpoints and evaluates them.

Evaluation results are written to TensorBoard logs.

The use case of inference mode is to make predictions on examples from an
unlabeled region. Only test data is expected (no training). The test data is
not expected to have ground truth labels. If the test data happens to have
valid labels, then the _ProcessCheckpoint function will log reliable metrics.
If not, there is a warning logged that the metrics should be ignored in
inference mode.
"""

import os
import re
import time
from typing import Dict, Set

from absl import app
from absl import flags
from absl import logging
import geopandas as gpd
import pandas as pd
from skai import ssl_flags
from skai.semi_supervised import ssl_train_library
from skai.semi_supervised import train
import tensorflow.compat.v1 as tf

###COPYBARA_PLACEHOLDER_01

flags.adopt_module_key_flags(ssl_flags)
FLAGS = flags.FLAGS


# Number of seconds to sleep between checking for new checkpoints.
_SLEEP_SECONDS = 5
# Number of seconds to wait for new activity. Terminate the job once exceeded.
_TIMEOUT_SECONDS = 3600
# Batch size when reading validation data.
_BATCH_SIZE = 64

_get_checkpoint_base = lambda path: re.sub(r'\.meta$', '', path)
LAST_PROCESSED_EPOCH_FILE = 'last_processed_epoch'
TENSORBOARD_EVENTS_DIR = 'logs'


def _GetCheckpointEpoch(checkpoint_path: str) -> int:
  """Extracts epoch number from checkpoint path."""
  match = re.search('model.*-([0-9]+).*', checkpoint_path)
  if not match:
    return -1
  return int(match.group(1))


def _GetCheckpointWallTime(checkpoint_path: str) -> int:
  """Gets wall time for a checkpoint in seconds using file modification time."""
  return tf.gfile.Stat(checkpoint_path).mtime_nsec * 1e-9  # nano to sec


def _CreateFileWriter(
    tensorboard_dir: str) -> Dict[str, tf.summary.FileWriter]:
  """Creates a FileWriter within the given directory.

  Args:
    tensorboard_dir: Full path to model's Tensorboard directory, where event
      files will be written.

  Returns:
    Newly created FileWriter.
  """
  return tf.summary.FileWriter(os.path.join(tensorboard_dir))


def _MakeSummary(tag: str, value: float) -> tf.Summary:
  """Creates a Tensorflow Summary object.

  Args:
    tag: Event tag.
    value: Value to record for event.

  Returns:
    Tensorflow summary.
  """
  summary = tf.Summary()
  summary_value = tf.Summary.Value()
  summary_value.tag = tag
  summary_value.simple_value = value
  summary.value.append(summary_value)
  return summary


def _MakeEvent(tag: str, value: float, wall_time: int, step: int) -> tf.Event:
  """Creates a Tensorflow Event object.

  Args:
    tag: Event tag.
    value: Value to record for event.
    wall_time: Wall time in seconds.
    step: Training step.

  Returns:
    Tensorflow event.
  """
  event = tf.Event()
  event.wall_time = wall_time
  event.step = step
  event.summary.CopyFrom(_MakeSummary(tag, value))
  return event


def _SaveLastProcessedEpoch(checkpoint_dir: str, epoch: int):
  """Saves a file to record last processed epoch.

  Args:
    checkpoint_dir: Directory containing checkpoints, where file will be saved.
    epoch: Integer of last processed epoch.
  """
  with tf.gfile.Open(
      os.path.join(checkpoint_dir, LAST_PROCESSED_EPOCH_FILE),
      'w') as last_processed_epoch_file:
    last_processed_epoch_file.write(str(epoch))


def _ProcessCheckpoint(
    model: train.ClassifySemi, epoch: int, checkpoint: str,
    file_writers: Dict[str, tf.summary.FileWriter]
) -> train.PredictionsWithCoordinatesPerDataset:
  """Loads a checkpoint and evaluates it on provided test set.

  Args:
    model: Initialized model that will run inference.
    epoch: Current epoch number.
    checkpoint: Path to model checkpoint file.
    file_writers: Dict mapping the directory name of a checkpoint to Tensorflow
      event file writers.

  Returns:
    PredictionsWithCoordinates object.
  """
  wall_time = _GetCheckpointWallTime(checkpoint)
  raw_metrics, _ = model.eval_checkpoint(
      _get_checkpoint_base(checkpoint))
  acc, auc, preds_per_dataset = raw_metrics
  if model.inference_mode:
    logging.warning('Model is in inference mode. If the test data does not '
                    'have ground truth labels, then the following metrics will '
                    'not be valid.')
  logging.info('Epoch: %d, walltime: %d, Test AUC: %f, Test Accuracy: %f',
               epoch, wall_time, auc.test_metric, acc.test_metric)
  if model.checkpoint_dir not in file_writers:
    file_writers[model.checkpoint_dir] = _CreateFileWriter(
        model.tensorboard_dir)
  file_writer = file_writers[model.checkpoint_dir]
  file_writer.add_event(
      _MakeEvent('metrics/auc/test_auc', auc.test_metric, wall_time, epoch))
  file_writer.add_event(
      _MakeEvent('metrics/acc/test_acc', acc.test_metric, wall_time, epoch))

  if not model.inference_mode:
    logging.info(
        'Epoch: %d, walltime: %d, Train_Label AUC: %f, Train_Label Accuracy: %f',
        epoch, wall_time, auc.train_labeled_metric, acc.train_labeled_metric)
    logging.info(
        'Epoch: %d, walltime: %d, Train_Unlabel AUC: %f, Train_Unlabel Accuracy: %f',
        epoch, wall_time, auc.train_unlabeled_metric,
        acc.train_unlabeled_metric)
    file_writer.add_event(
        _MakeEvent('metrics/auc/train_label_auc', auc.train_labeled_metric,
                   wall_time, epoch))
    file_writer.add_event(
        _MakeEvent('metrics/acc/train_label_acc', acc.train_labeled_metric,
                   wall_time, epoch))
    file_writer.add_event(
        _MakeEvent('metrics/auc/train_unlabel_auc', auc.train_unlabeled_metric,
                   wall_time, epoch))
    file_writer.add_event(
        _MakeEvent('metrics/acc/train_unlabel_acc', acc.train_unlabeled_metric,
                   wall_time, epoch))

  file_writer.flush()
  logging.info('Processed checkpoint "%s".', checkpoint)
  _SaveLastProcessedEpoch(model.checkpoint_dir, epoch)
  return preds_per_dataset


def _WritePreds(preds_with_coords: train.PredictionsWithCoordinates,
                dataset: str, epoch: int):
  """Writes data to GeoJSON with fields: prediction, longitude, and latitude.

  Args:
    preds_with_coords: PredictionsWithCoordinates object.
    dataset: Type of dataset (e.g. train_label, train_unlabel, or test).
    epoch: Training epoch number.
  """
  preds_file_name = f'{dataset}_ckpt_{epoch}.geojson'
  preds_file_path = os.path.join(FLAGS.train_dir, 'predictions',
                                 preds_file_name)
  with tf.gfile.GFile(preds_file_path, 'w') as preds_file:
    preds_dict = {
        'longitude': preds_with_coords.lons,
        'latitude': preds_with_coords.lats
    }
    for i in range(preds_with_coords.preds.shape[1]):
      preds_dict[f'class_{i}'] = preds_with_coords.preds[:, i]
    preds_df = pd.DataFrame.from_dict(preds_dict)
    geo_preds_df = gpd.GeoDataFrame(
        preds_df,
        geometry=gpd.points_from_xy(preds_with_coords.lons,
                                    preds_with_coords.lats))
    geo_preds_df.to_file(preds_file, index=False, driver='GeoJSON')


def _SavePredictionsToFile(
    preds_per_dataset: train.PredictionsWithCoordinatesPerDataset, epoch: int,
    inference_mode: bool):
  """Saves model prediction for damaged class, latitude, and longitude.

  Args:
    preds_per_dataset: PredictionsWithCoordinatesPerDataset object, containing
      predictions and corresponding coordinates by dataset. In eval mode, only
      the test set fields will be populated.
    epoch: Model's training epoch or any other integer for file name.
    inference_mode: Boolean for inference mode. When True, only test data saved.
  """
  _WritePreds(preds_per_dataset.test_preds_coords, 'test', epoch)
  if not inference_mode:
    _WritePreds(preds_per_dataset.train_label_preds_coords, 'train_labeled',
                epoch)
    _WritePreds(preds_per_dataset.train_unlabel_preds_coords, 'train_unlabeled',
                epoch)


def _GetAlreadyProcessedCheckpoints(all_checkpoints: Set[str]) -> Set[str]:
  """Gets the set already processed checkpoints."""
  processed_checkpoints = set()
  last_processed_epoch_file_path = os.path.join(FLAGS.train_dir,
                                                FLAGS.dataset_name,
                                                LAST_PROCESSED_EPOCH_FILE)
  if tf.gfile.Exists(last_processed_epoch_file_path):
    with tf.gfile.Open(last_processed_epoch_file_path,
                       'r') as last_processed_epoch_file:
      last_processed_epoch = int(last_processed_epoch_file.read())
      for checkpoint in all_checkpoints:
        epoch = _GetCheckpointEpoch(checkpoint)
        if epoch <= last_processed_epoch:
          processed_checkpoints.add(checkpoint)

  return processed_checkpoints


def _FindCheckpoints(checkpoint_dir: str) -> Set[str]:
  """Acquires all model checkpoints in given directory.

  Args:
    checkpoint_dir: Directory containing model checkpoints.

  Returns:
    Set of all checkpoints in specified directory.
  """
  checkpoints = set()
  checkpoint_pattern = re.compile(
      os.path.join(checkpoint_dir, 'model.ckpt-[0-9]*.meta'))
  for dirname, _, filenames in tf.gfile.Walk(checkpoint_dir):
    for filename in filenames:
      full_filepath = os.path.join(dirname, filename)
      match = checkpoint_pattern.match(full_filepath)
      if match:
        checkpoints.add(match.group())
  return checkpoints


def main(_):
  ###COPYBARA_PLACEHOLDER_02

  # Create dataset with only validation data
  dataset = ssl_train_library.create_dataset(
      shuffle=FLAGS.shuffle_seed is not None)
  ssl_train_library.set_experiment_hyperparams()
  model = ssl_train_library.create_model(dataset)

  if FLAGS.eval_ckpt:
    # User has specified a checkpoint
    raw_metrics, _ = model.eval_checkpoint(
        _get_checkpoint_base(FLAGS.eval_ckpt))
    current_epoch = _GetCheckpointEpoch(FLAGS.eval_ckpt)
    _, _, preds = raw_metrics
  else:
    # When a hyperparameter sweep is launched, a subdirectory is created for
    # each combination. We will create a FileWriter for each subdirectory to
    # evaluate checkpoints and write TensorBoard event files separately. We keep
    # track of them using a dictionary mapping subdirectory to FileWriter.
    file_writers = {}

    all_checkpoints = _FindCheckpoints(model.checkpoint_dir)
    processed_checkpoints = _GetAlreadyProcessedCheckpoints(all_checkpoints)
    current_epoch = 0
    last_activity_secs = time.time()
    preds = None
    while True:
      wait_secs = time.time() - last_activity_secs
      if wait_secs >= _TIMEOUT_SECONDS:
        if FLAGS.save_predictions and preds is not None:
          _SavePredictionsToFile(preds, current_epoch, model.inference_mode)
        logging.warning('Timed out waiting for new checkpoints')
        break

      new_checkpoints = (all_checkpoints - processed_checkpoints)
      for checkpoint in sorted(new_checkpoints):
        current_epoch = _GetCheckpointEpoch(checkpoint)
        preds = _ProcessCheckpoint(model, current_epoch, checkpoint,
                                   file_writers)
        processed_checkpoints.add(checkpoint)
        last_activity_secs = time.time()
      if FLAGS.last_epoch > 0 and processed_checkpoints:
        max_processed_checkpoint = max(
            _GetCheckpointEpoch(c) for c in processed_checkpoints)
        if max_processed_checkpoint >= FLAGS.last_epoch:
          break

      all_checkpoints = _FindCheckpoints(model.checkpoint_dir)
      time.sleep(_SLEEP_SECONDS)

  if FLAGS.save_predictions:
    if preds is None:
      logging.warning('No predictions generated for saving.')
    else:
      _SavePredictionsToFile(preds, current_epoch, model.inference_mode)


if __name__ == '__main__':
  app.run(main)
