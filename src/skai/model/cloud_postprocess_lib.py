"""Post-processing utilities for clouds."""

import geopandas as gpd
import pandas as pd
from skai import utils


def convert_to_geopandas(df: pd.DataFrame) -> gpd.GeoDataFrame:
  """Converts a dataframe to a geopandas dataframe with UTM projection."""
  gdf = gpd.GeoDataFrame(
      index=df.index,
      geometry=gpd.points_from_xy(df.longitude, df.latitude),
      crs='EPSG:4326',
  )
  c = gdf.unary_union.centroid
  new_proj = utils.convert_wgs_to_utm(c.x, c.y)
  return gdf.to_crs(new_proj)


def identify_clouds(
    df: pd.DataFrame, distance_threshold: float
) -> pd.DataFrame:
  """Identifies clouds in a dataframe.

  It idenifies a row in a dataframe to be cloudy if:
    1. The row has `cloud_score` > 0.5.
    2. The row is within `distance_threshold` of a row with
      `cloud_score` > 0.5.

  Args:
    df: Dataframe that has at least the following columns: - `cloud_score`: The
      cloud score of the row. - `latitude`: The latitude of the row.
      -`longitude`: The longitude of the row. - 'int64_id': The unique id of the
      row.
    distance_threshold: The distance threshold in metres to use for identifying
      clouds.

  Returns:
    A dataframe with the same columns as `df` and an additional column
    `is_cloudy` that indicates whether the row is cloudy.
  """
  if 'cloud_score' not in df.columns:
    raise ValueError('no `cloud_score` column found')

  if 'int64_id' not in df.columns:
    raise ValueError('no `int64_id` column found')

  if 'longitude' not in df.columns or 'latitude' not in df.columns:
    raise ValueError('no `longitude` or `latitude` column found')

  # Convert to geopandas dataframe and project to UTM.
  gdf = convert_to_geopandas(df)

  # Split into cloudy and non-cloudy and filter out non-cloudy that are close to
  # cloudy.
  cloudy_gdf = gdf[df['cloud_score'] > 0.5]
  non_cloudy_gdf = gdf[df['cloud_score'] <= 0.5]

  joined_gdf = gpd.sjoin_nearest(
      non_cloudy_gdf, cloudy_gdf, how='inner', max_distance=distance_threshold
  )

  cloudy_idx = joined_gdf.index.drop_duplicates()

  df['is_cloudy'] = 0
  df.loc[df['cloud_score'] > 0.5, 'is_cloudy'] = 1
  df.loc[cloudy_idx, 'is_cloudy'] = 1
  return df
