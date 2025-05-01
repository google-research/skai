"""Ensemble model prediction CSVs, creating an averaged damage_score."""

import os
from typing import Any

from absl import logging
import pandas as pd
import tensorflow as tf


def ensemble_prediction_csvs(input_dir: str) -> pd.DataFrame:
  """Reads model prediction files and provides an ensembled damage_score.

  The damage_score is the mean of the damage_score across all models.

  Args:
      input_dir: The path to the directory containing the CSV files.

  Returns:
      A pandas DataFrame containing the ensembled data from all CSVs.
  """
  pred_file_paths = tf.io.gfile.glob(os.path.join(input_dir, '*_output.csv'))
  if not pred_file_paths:
    raise ValueError(f'No CSV files found in {input_dir}')
  logging.info('Found %d CSV files in %s', len(pred_file_paths), input_dir)

  pred_dfs = []
  num_predictions = None
  for i, file_path in enumerate(pred_file_paths):
    df = pd.read_csv(file_path).drop_duplicates(subset=['example_id'])
    if not num_predictions:
      num_predictions = df.shape[0]
    elif num_predictions != df.shape[0]:
      logging.warning(
          'Number of predictions are inconsistent across CSV files, so some '
          'rows may only have a subset of the predictions. '
          'Previous CSV has %d predictions, while current CSV has %d.',
          num_predictions,
          df.shape[0],
      )
    df[f'damage_score_{i}'] = df['damage_score']
    pred_dfs.append(df)

  pred_df = pd.concat(pred_dfs)

  def keep_value(x: Any):
    """Keeps only the unique values in a column for an aggregation group."""
    unique_values = list(set(x.dropna()))
    if len(unique_values) == 1:
      # Usually 1 unique value for the group being aggregated, e.g. example_id.
      return unique_values[0]
    return unique_values

  # Set the new damage_score to be the mean of the ensemble.
  aggregate_fn = {
      'damage_score': 'mean',
      'longitude': keep_value,
      'latitude': keep_value,
      'is_cloudy': keep_value,
      'building_id': keep_value,
      'example_id': keep_value,
      'int64_id': keep_value,
      'plus_code': keep_value,
      'label': keep_value,
      'damage': keep_value,
  }
  for i in range(len(pred_dfs)):
    aggregate_fn[f'damage_score_{i}'] = keep_value

  ensembled_df = pred_df.groupby('example_id').agg(aggregate_fn)
  ensembled_df['label'] = (ensembled_df['damage_score'] > 0.5).astype(float)
  ensembled_df['damage'] = ensembled_df['damage_score'] > 0.5
  return ensembled_df


def save_ensemble_csv(output_dir: str, ensembled_df: pd.DataFrame) -> None:
  """Saves the ensembled CSV to the specified directory."""
  ensembled_df.to_csv(os.path.join(output_dir, 'ensemble_predictions.csv'))

