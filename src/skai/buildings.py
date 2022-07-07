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

from typing import List, Tuple
import geopandas as gpd
import pandas as pd
import shapely.geometry
import tensorflow as tf

Point = shapely.geometry.point.Point
Polygon = shapely.geometry.polygon.Polygon


def _read_buildings_csv(path: str) -> List[Tuple[float, float]]:
  """Reads (longitude, latitude) coordinates from a CSV file.

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
  if 'longitude' not in df.columns or 'latitude' not in df.columns:
    raise ValueError(
        f'Malformed CSV file "{path}". File does not contain "longitude" and '
        '"latitude" columns')
  return [(row.longitude, row.latitude) for _, row in df.iterrows()]


def read_buildings_file(path: str,
                        regions: List[Polygon]) -> List[Tuple[float, float]]:
  """Extracts building coordinates from a file.

  Supported file formats are csv, shapefile, and geojson.

  Args:
    path: Path to buildings file.
    regions: Regions to where building coordinates should come from.

  Returns:
    List of (longitude, latitude) building coordinates.
  """
  if path.lower().endswith('.csv'):
    coords = _read_buildings_csv(path)
  else:
    coords = []
    df = gpd.read_file(path).to_crs(epsg=4326)
    geometries = list(df.geometry.values)
    for g in geometries:
      centroid = g.centroid
      coords.append((centroid.x, centroid.y))

  filtered_coords = []
  for lon, lat in coords:
    point = Point(lon, lat)
    for region in regions:
      if region.intersects(point):
        filtered_coords.append((lon, lat))
        break

  return filtered_coords


def read_aois(path: str) -> List[Polygon]:
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
