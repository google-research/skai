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

"""Functions for reading building centroids from files."""

import logging
import os
import geopandas as gpd
import pandas as pd
import shapely.geometry
import tensorflow as tf

Point = shapely.geometry.point.Point
Polygon = shapely.geometry.polygon.Polygon


def _read_buildings_csv(path: str) -> gpd.GeoDataFrame:
  """Reads CSV file containing building footprints to GeoDataFrame.

  The file should contain "longitude" and "latitude" columns.

  Args:
    path: Path to CSV file.

  Returns:
    List of (longitude, latitude) coordinates.

  Raises:
    ValueError if CSV file isn't formatted correctly.
  """
  with tf.io.gfile.GFile(path, 'r') as csv_file:
    df = pd.read_csv(csv_file)
  if 'geometry' in df.columns:
    logging.info('Parsing %d WKT strings. This could take a while.', len(df))
    geometries = gpd.GeoSeries.from_wkt(df['geometry'])
    df.drop(columns=['geometry'], inplace=True)
  elif 'wkt' in df.columns:
    logging.info('Parsing %d WKT strings. This could take a while.', len(df))
    geometries = gpd.GeoSeries.from_wkt(df['wkt'])
    df.drop(columns=['wkt'], inplace=True)
  elif 'longitude' in df.columns and 'latitude' in df.columns:
    geometries = gpd.points_from_xy(df['longitude'], df['latitude'])
    df.drop(columns=['longitude', 'latitude'], inplace=True)
  else:
    raise ValueError(f'No geometry information found in file "{path}"')

  return gpd.GeoDataFrame(df, geometry=geometries, crs=4326)


def convert_buildings_file(
    path: str, regions: list[Polygon], output_path: str
) -> None:
  """Converts an input file encoding building footprints to the standard format.

  Also filters out any buildings that don't fall in one of the specified region
  polygons.

  Supported file formats are csv and anything that GeoPandas handles.

  Args:
    path: Path to buildings file.
    regions: Regions to where building coordinates should come from.
    output_path: Path to write buildings GeoDataFrame to.
  """
  if path.lower().endswith('.csv'):
    buildings_gdf = _read_buildings_csv(path)
  elif path.lower().endswith('.parquet'):
    buildings_gdf = read_buildings_file(path)
  else:
    with tf.io.gfile.GFile(path, 'rb') as f:
      buildings_gdf = gpd.read_file(f).to_crs(epsg=4326)

  combined_regions = gpd.GeoSeries(regions).unary_union
  in_regions = buildings_gdf.intersects(combined_regions)
  write_buildings_file(buildings_gdf[in_regions], output_path)


def read_aois(path: str) -> list[Polygon]:
  """Reads area of interest polygons from a file.

  Common file formats such as shapefile and GeoJSON are supported. However, the
  file must contain only polygons. All polygons will be converted to EPSG:4326
  (longitude, latitude) coordinates.

  Args:
    path: Path to file containing polygons.

  Returns:
    List of polygons.

  Raises:
    ValueError if file contains geometry types other than polygons (such as
    lines or points).
  """
  # Convert all data to long/lat
  df = gpd.read_file(path).to_crs(epsg=4326)
  geometries = list(df.geometry.values)
  for g in geometries:
    if g.geometryType() not in ['Polygon', 'MultiPolygon']:
      raise ValueError(
          f'Unexpected geometry for area of interest: "{g.geometryType()}"')
  return geometries


def write_buildings_file(gdf: gpd.GeoDataFrame, output_path: str) -> None:
  """Writes a GeoDataFrame of building geometries to file.

  Serializes GeoDataFrame using Parquet file format to allow fast reading of
  individual columns, such as longitude and latitude, in large datasets.

  Args:
    gdf: GeoDataFrame of building geometries.
    output_path: Output path.
  """
  if 'longitude' not in gdf.columns and 'latitude' not in gdf.columns:
    centroids = gdf.geometry.centroid
    output_gdf = gdf.copy().to_crs(4326)
    output_gdf['longitude'] = [c.x for c in centroids]
    output_gdf['latitude'] = [c.y for c in centroids]
  else:
    output_gdf = gdf.to_crs(4326)

  output_dir = os.path.dirname(output_path)
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  with tf.io.gfile.GFile(output_path, 'wb') as f:
    f.closed = False
    output_gdf.to_parquet(f, index=False)


def read_buildings_file(path: str) -> gpd.GeoDataFrame:
  """Reads a GeoDataFrame of building geometries from file.

  The GeoDataFrame must have been serialized by the write_buildings_file
  function defined above.

  Args:
    path: Path to serialized GeoDataFrame.

  Returns:
    Buildings GeoDataFrame.
  """
  with tf.io.gfile.GFile(path, 'rb') as f:
    f.closed = False  # Work-around for GFile issue.
    return gpd.read_parquet(f).to_crs(4326)


def read_building_coordinates(path: str) -> pd.DataFrame:
  """Reads only the longitude and latitude columns of a buildings file.

  The GeoDataFrame must have been serialized by the write_buildings_file
  function defined above.

  Args:
    path: Path to buildings file. Should be a GeoDataFrame in parquet format.

  Returns:
    DataFrame (not GeoDataFrame) containing
  """
  with tf.io.gfile.GFile(path, 'rb') as f:
    f.closed = False  # Work-around for GFile issue.
    return pd.read_parquet(f, columns=['longitude', 'latitude'])
