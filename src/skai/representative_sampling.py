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

"""Library for sampling examples to be sent for labeling."""

import math
import geopandas as gpd
import numpy as np
import pandas as pd
from skai import utils

# For each cell, we identify the examples that fall in it and get its score
# distribution. The score distribution is then divided into quartiles, and each
# quartile is divided into 4 equally sized buckets. The method samples from each
# bucket. For example, if scores of a quartile range from 0 to 0.25, we sample
# from the score range buckets of:
# (1) 0 to 0.0625, (2) 0.0625 to 0.125, (3) 0.125 to 0.1875, (4) 0.1875 to 0.25.
_NUM_BUCKETS_PER_QUARTILE = 4


def _divide_aoi_into_grid(
    aoi_gdf: gpd.GeoDataFrame, grid_rows: int, grid_cols: int
) -> gpd.GeoDataFrame:
  """Creates a grid to represent the geographic area covered by the AOI.

  Adds a 'grid_cell_id' column to the dataframe.

  Args:
    aoi_gdf: A dataframe of buildings across entire AOI, where each row
      represents a building.
    grid_rows: Number of rows to divide the AOI into.
    grid_cols: Number of columns to divide the AOI into.

  Returns:
    The dataframe with an additional 'grid_cell_id' column.
  """
  # Calculate grid cell boundaries.
  x_borders = np.linspace(
      aoi_gdf['longitude'].min(), aoi_gdf['longitude'].max(), grid_cols + 1
  )
  x_borders[-1] += 0.000001  # Add small offset to include points on the edge.
  y_borders = np.linspace(
      aoi_gdf['latitude'].min(), aoi_gdf['latitude'].max(), grid_rows + 1
  )
  y_borders[-1] += 0.000001  # Add small offset to include points on the edge.
  grid_cells = []
  print('Grid Cell Boundaries: ')
  for i in range(grid_cols):
    for j in range(grid_rows):
      cell = (x_borders[i], x_borders[i + 1], y_borders[j], y_borders[j + 1])
      grid_cells.append(cell)
      print(
          f'Grid Cell {len(grid_cells) - 1}: Longitude ({cell[0]:.4f},'
          f' {cell[1]:.4f}), Latitude ({cell[2]:.4f}, {cell[3]:.4f})'
      )

  aoi_gdf['grid_cell'] = aoi_gdf.apply(
      lambda row: _get_grid_cell(row['longitude'], row['latitude'], grid_cells),
      axis=1,
  )
  return aoi_gdf


def _get_grid_cell(
    longitude: float,
    latitude: float,
    grid_cells: list[tuple[float, float, float, float]],
):
  """Returns the index of the grid cell that contains the given coordinates."""
  for i, cell in enumerate(grid_cells):
    if cell[0] <= longitude < cell[1] and cell[2] <= latitude < cell[3]:
      return i
  raise ValueError(
      f'Coordinates ({longitude}, {latitude}) not found in any grid cell.'
  )


def _get_quartile_gdf(
    cell_gdf: gpd.GeoDataFrame, quartiles: list[float], quartile_idx: int
) -> tuple[gpd.GeoDataFrame, float]:
  """Retrieves rows that belong to a given quartile from the grid cell.

  Args:
    cell_gdf: A dataframe of rows from a grid cell.
    quartiles: A list of quartiles based on damage score for the given cell.
    quartile_idx: The index of the quartile to retrieve.

  Returns:
    A dataframe of rows that belong to the given quartile and the size of each
    bucket.
  """
  if quartile_idx == 0:
    quartile_gdf = cell_gdf[cell_gdf['damage_score'] <= quartiles[0]]
    bucket_score_increment = float(quartiles[0] / _NUM_BUCKETS_PER_QUARTILE)
  elif quartile_idx == 1:
    quartile_gdf = cell_gdf[
        (cell_gdf['damage_score'] > quartiles[0])
        & (cell_gdf['damage_score'] <= quartiles[1])
    ]
    bucket_score_increment = float(
        (quartiles[1] - quartiles[0]) / _NUM_BUCKETS_PER_QUARTILE
    )
  elif quartile_idx == 2:
    quartile_gdf = cell_gdf[
        (cell_gdf['damage_score'] > quartiles[1])
        & (cell_gdf['damage_score'] <= quartiles[2])
    ]
    bucket_score_increment = float(
        (quartiles[2] - quartiles[1]) / _NUM_BUCKETS_PER_QUARTILE
    )
  else:
    quartile_gdf = cell_gdf[cell_gdf['damage_score'] > quartiles[2]]
    bucket_score_increment = (
        float(quartile_gdf['damage_score'].max() - quartiles[2])
        / _NUM_BUCKETS_PER_QUARTILE
    )
  return quartile_gdf, bucket_score_increment


def _get_bucket_gdf(
    quartile_gdf: gpd.GeoDataFrame,
    quartile_start: float,
    bucket_score_increment: float,
    bucket_idx: int,
    all_samples: list[int],
) -> gpd.GeoDataFrame:
  """Retrieves rows that belong to a given bucket from the quartile.

  Since method will be sampling from the bucket, it also excludes any already
  sampled examples.

  Args:
    quartile_gdf: A dataframe of rows from a quartile.
    quartile_start: The starting score of the quartile.
    bucket_score_increment: The increment of the score range for each bucket.
    bucket_idx: The index of the bucket to retrieve.
    all_samples: A list of already sampled examples.

  Returns:
    A Geodataframe of rows that belong to the given bucket.
  """
  min_score = quartile_start + bucket_idx * bucket_score_increment
  max_score = min_score + bucket_score_increment
  inclusive = 'both' if bucket_idx == _NUM_BUCKETS_PER_QUARTILE - 1 else 'left'
  bucket_gdf = quartile_gdf[
      quartile_gdf['damage_score'].between(min_score, max_score, inclusive)
  ]
  if all_samples:
    bucket_gdf = bucket_gdf[
        ~bucket_gdf.index.isin(all_samples)
    ]
  return bucket_gdf


def _drop_points_within_sample(
    sample: gpd.array.GeometryArray,
    bucket_gdf: gpd.GeoDataFrame,
    buffer_meters: float,
) -> gpd.GeoDataFrame:
  """Drops points within a buffer distance of the sampled point.

    Points are kept or dropped based on the row order in the dataframe, so it's
    important for the input to already be randomly shuffled.

  Args:
    sample: A geometry array of the sampled point.
    bucket_gdf: A GeoDataFrame of all points in the bucket.
    buffer_meters: The buffer distance in meters.

  Returns:
    Points with neighbors dropped.
  """
  buffers = gpd.GeoDataFrame(geometry=sample.buffer(buffer_meters))
  joined = bucket_gdf.sjoin(buffers, how='inner')
  return bucket_gdf.drop(index=joined.index)


def _sample_from_bucket(
    bucket_gdf: gpd.GeoDataFrame,
    num_to_sample_from_bucket: int,
    num_to_sample_total: int,
    all_samples: list[int],
    buffer_meters: float):
  """Samples from a bucket and excludes any already sampled examples."""
  for _ in range(num_to_sample_from_bucket):
    sample = bucket_gdf.sample(1)
    all_samples.extend(list(sample.index))
    if len(all_samples) >= num_to_sample_total:
      break
    # Every time we sample, make sure to remove any overlapping
    # buildings from the bucket.
    bucket_gdf = _drop_points_within_sample(
        sample.geometry, bucket_gdf, buffer_meters
    )


def sample_examples(
    gdf: gpd.GeoDataFrame,
    num_to_sample_total: int,
    num_top_examples: int,
    grid_cell_idx: list[int],
    buffer_meters: float,
) -> list[int]:
  """Samples a given number of examples from a dataframe evenly across a grid.

  The method iterates over each cell in the grid and samples from each as
  follows. For each cell, we calculate the quartiles for its example score
  distribution, and for each quartile, we divide the score range into equally
  sized buckets. From each bucket, the method samples a set number of examples.
  For the train set, it also ranks the remaining examples and then takes a set
  number of top-scoring examples to ensure positives.

  Args:
    gdf: A GeoDataFrame of buildings across the AOI, where each row represents a
      building.
    num_to_sample_total: The total number of examples to sample.
    num_top_examples: The number of examples to take from the top of the score
      distribution after ranking.
    grid_cell_idx: A list of grid cell indices from highest to lowest count.
    buffer_meters: The buffer distance between two examples to consider them
      overlapping.

  Returns:
    A list of indices of the sampled examples.
  """
  num_to_sample_per_cell = int(
      math.ceil(num_to_sample_total / len(grid_cell_idx))
  )
  # For every cell, we sample a certain number evenly over the quartile-based
  # buckets, then collect from the remaining top-scoring buildings
  num_to_sample_from_top = int(math.ceil(num_top_examples / len(grid_cell_idx)))
  num_to_sample_from_buckets = num_to_sample_per_cell - num_to_sample_from_top
  num_to_sample_per_bucket = max(
      math.ceil(num_to_sample_from_buckets / (4 * _NUM_BUCKETS_PER_QUARTILE)), 1
  )
  print(f'Number of Populated Grid Cells: {len(grid_cell_idx)}')

  all_samples = []
  all_samples_count_in_previous_round = -1

  while len(all_samples) < num_to_sample_total:
    for grid_idx in grid_cell_idx:
      cell_gdf = gdf[gdf['grid_cell'] == grid_idx]
      if len(cell_gdf) < num_to_sample_per_cell:
        continue
      # Step 1: Calculate quartiles of this grid cell.
      quartiles = list(cell_gdf['damage_score'].quantile([0.25, 0.5, 0.75]))
      for quartile_idx in range(4):  # 4 quartiles total.
        # Step 2: Divide each quartile into equally sized score range buckets.
        quartile_gdf, bucket_score_increment = _get_quartile_gdf(
            cell_gdf, quartiles, quartile_idx
        )
        quartile_start = 0 if quartile_idx == 0 else quartiles[quartile_idx - 1]
        for bucket_idx in range(_NUM_BUCKETS_PER_QUARTILE):
          # Step 3: Sample from each score range bucket.
          bucket_gdf = _get_bucket_gdf(
              quartile_gdf,
              quartile_start,
              bucket_score_increment,
              bucket_idx,
              all_samples,
          )
          if not bucket_gdf.empty:
            num_to_sample_from_bucket = min(
                num_to_sample_per_bucket, len(bucket_gdf)
            )
            _sample_from_bucket(
                bucket_gdf,
                num_to_sample_from_bucket,
                num_to_sample_total,
                all_samples,
                buffer_meters
            )
          if len(all_samples) >= num_to_sample_total:
            break
        if len(all_samples) >= num_to_sample_total:
          break

      # Step 4: Now get the top-scoring examples from the remaining buildings
      # if we're collecting for the train set.
      if num_to_sample_from_top > 0:
        remain_gdf = cell_gdf[~cell_gdf.index.isin(all_samples)]
        top_gdf = remain_gdf.nlargest(
            min(num_to_sample_from_top, num_to_sample_total - len(all_samples)),
            'damage_score'
        )
        all_samples.extend(list(top_gdf.index))
      if len(all_samples) >= num_to_sample_total:
        break
    if len(all_samples) >= num_to_sample_total:
      break
    # If the number of samples hasn't changed after iterating over the grid
    # cells again, then there are no more samples to pick.
    if (
        all_samples_count_in_previous_round >= 0
        and all_samples_count_in_previous_round == len(all_samples)
    ):
      print('No more examples to sample.')
      break
    all_samples_count_in_previous_round = len(all_samples)
  if len(set(all_samples)) < len(all_samples):
    raise ValueError('Duplicate samples found.')
  return all_samples


def run_representative_sampling(
    scores_df: pd.DataFrame,
    num_examples_to_sample_total: int,
    num_examples_to_take_from_top: int,
    grid_rows: int,
    grid_cols: int,
    train_ratio: float,
    buffer_meters: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Creates a train and test dataset with representative sampling.

  Takes into account the geographic and score distribution of buildings across
  the AOI, which are represented in the scores dataframe. Selects the set number
  of examples total to be sent to labelers, with an additional column in the
  output dataframe to indicate the train/test split.

  Args:
    scores_df: The Dataframe containing the prediction scores.
    num_examples_to_sample_total: The total number of examples to sample.
    num_examples_to_take_from_top: The number of examples to take from the top
      of the score distribution after ranking.
    grid_rows: Number of rows to divide the AOI into.
    grid_cols: Number of columns to divide the AOI into.
    train_ratio: The ratio of examples to sample for the train set.
    buffer_meters: The buffer distance between two examples to consider them
      overlapping.

  Returns:
    A tuple containing the dataframes for the train and test sets, respectively.
  """
  if 'damage_score' not in scores_df.columns:
    raise ValueError(
        'The predictions file must contain a column named "damage_score".'
    )
  if (
      'latitude' not in scores_df.columns
      or 'longitude' not in scores_df.columns
  ):
    raise ValueError(
        'The predictions file must contain columns "latitude" and "longitude".'
    )
  print('Number of Buildings Total: ', scores_df.shape)
  # Remove clouds and duplicate rows.
  scores_df = scores_df[scores_df['is_cloudy'] != 1]
  scores_df = scores_df.drop_duplicates()

  scores_gdf = gpd.GeoDataFrame(
      scores_df,
      geometry=gpd.points_from_xy(scores_df.longitude, scores_df.latitude),
      crs='EPSG:4326',
  )
  scores_gdf = scores_gdf.to_crs(
      utils.get_utm_crs(
          np.mean(scores_gdf['longitude']), np.mean(scores_gdf['latitude'])
      )
  )

  scores_gdf = _divide_aoi_into_grid(scores_gdf, grid_rows, grid_cols)
  # Sort the grid cells from highest to lowest count and drop any empty ones.
  grid_cell_counts = (
      scores_gdf['grid_cell'].value_counts().sort_values(ascending=False)
  )
  grid_cell_idx = list(grid_cell_counts.index)

  # Sample for the train and test sets and add a column to indicate the split.
  num_train_examples = int(num_examples_to_sample_total * train_ratio)
  num_test_examples = int(num_examples_to_sample_total * (1 - train_ratio))
  print(f'Number of Train Examples Requested: {num_train_examples}')
  print(f'Number of Test Examples Requested: {num_test_examples}')
  train_index = sample_examples(
      scores_gdf,
      num_train_examples,
      num_examples_to_take_from_top,
      grid_cell_idx,
      buffer_meters
  )
  train_set = scores_gdf.loc[train_index]
  train_set['split'] = 'train'
  train_set_buffer = gpd.GeoDataFrame(geometry=train_set.buffer(buffer_meters))
  close_to_train_set = scores_gdf.sjoin(train_set_buffer, how='inner')
  remaining = scores_gdf.drop(index=close_to_train_set.index)
  test_index = sample_examples(
      remaining,
      num_test_examples,
      0,
      grid_cell_idx,
      buffer_meters,
  )
  test_set = scores_gdf.loc[test_index]
  test_set['split'] = 'test'
  train_test_intersection = test_set.sjoin(train_set_buffer, how='inner')
  if not train_test_intersection.empty:
    raise ValueError('Train and test sets have overlapping examples.')
  print(f'Number of Train Examples Sampled: {len(train_index)}')
  print(f'Number of Test Examples Sampled: {len(test_index)}')
  return (
      train_set.drop(columns=['geometry']),
      test_set.drop(columns=['geometry']),
  )
