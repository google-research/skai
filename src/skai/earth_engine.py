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
"""Library of functions for interacting with Google Earth Engine.
"""

import json
import shutil
from typing import List, Optional, Tuple
import urllib.request

from absl import logging
import ee  # pytype: disable=import-error  # mapping-is-not-sequence
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import tensorflow as tf

ShapelyGeometry = shapely.geometry.base.BaseGeometry

# When auto-selecting before images, this is the maximum number of images to
# choose from. If there are more images available in the date range and AOI
# specified, only the most recent images will be considered.
_MAX_BEFORE_IMAGE_POOL = 100


def _shapely_to_ee_feature(shapely_geometry: ShapelyGeometry) -> ee.Feature:
  """Converts shapely geometry into Earth Engine Feature."""
  geojson = json.loads(gpd.GeoSeries([shapely_geometry]).to_json())
  return ee.Feature(geojson['features'][0]['geometry'])


def _get_open_building_feature_centroid(feature: ee.Feature) -> ee.Feature:
  """Extracts centroid information from Open Buildings feature."""
  centroid = ee.Geometry(feature.get('longitude_latitude')).coordinates()
  return ee.Feature(
      None, {'longitude': centroid.get(0), 'latitude': centroid.get(1)})


def _download_feature_collection(
    collection: ee.FeatureCollection) -> gpd.GeoDataFrame:
  """Downloads a FeatureCollection from Earth Engine as a GeoDataFrame.

  Args:
    collection: EE FeatureCollection to download.

  Returns:
    FeatureCollection data as a GeoDataFrame.
  """
  url = collection.getDownloadURL('CSV')
  with urllib.request.urlopen(url) as url_file:
    df = pd.read_csv(url_file)
  geometry = df['.geo'].apply(json.loads).apply(shapely.geometry.shape)
  properties = df.drop(columns=['.geo'])
  return gpd.GeoDataFrame(properties, geometry=geometry, crs=4326)


def _download_feature_collection_to_file(
    collection: ee.FeatureCollection, properties: List[str],
    output_path: str) -> gpd.GeoDataFrame:
  """Downloads a FeatureCollection from Earth Engine as a GeoDataFrame.

  Args:
    collection: EE FeatureCollection to download.
    properties: List of properties to download.
    output_path: Path to save CSV file to.

  Returns:
    FeatureCollection data as a GeoDataFrame.
  """
  url = collection.getDownloadURL('CSV', properties)
  with urllib.request.urlopen(url) as url_file, tf.io.gfile.GFile(
      output_path, 'wb') as output:
    shutil.copyfileobj(url_file, output)
  with tf.io.gfile.GFile(output_path, 'r') as f:
    try:
      df = pd.read_csv(f)
    except pd.errors.EmptyDataError:
      # Initialize an empty dataframe.
      df = pd.DataFrame(columns=['longitude', 'latitude'])
  if '.geo' in df.columns:
    geometry = df['.geo'].apply(json.loads).apply(shapely.geometry.shape)
    properties = df.drop(columns=['.geo'])
  elif 'longitude' in df.columns and 'latitude' in df.columns:
    geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
    properties = df.drop(columns=['longitude', 'latitude'])
  else:
    geometry = None
    properties = None

  return gpd.GeoDataFrame(properties, geometry=geometry, crs=4326)


def _image_to_feature(image_object: ee.ComputedObject) -> ee.Feature:
  """Converts an image into a feature.

  The feature's geometry will be an approximate outline of the non-blank pixels
  of the image. The feature will have important properties of the image.

  Args:
    image_object: The image.

  Returns:
    A feature representing the image.
  """
  image = ee.Image(image_object)
  feature = image.select(0).multiply(0).reduceToVectors(scale=100).first()
  feature = feature.set({
      'image_id': image.get('system:id'),
      'resolution': image.projection().nominalScale(),
      'time': image.get('system:time_start'),
      'provider': image.get('provider_id'),
      'name': image.get('name'),
      'label': None,
      'count': None
  })
  return feature


def _get_image_collection_gdf(
    image_collection: ee.ImageCollection,
    max_images: int) -> gpd.GeoDataFrame:
  """Returns a GeoDataFrame for the images in a collection.

  Each row of the GeoDataFrame corresponds to one image in the collection.
  The row's geometry will be the image's geometry. The GeoDataFrame will have
  the following properties:

  image_id - EE id of the image.
  resolution - Resolution of the image in meters.
  time - Timestamp for when the image was taken.
  provider - Image provider code.
  name - The image name in EE.

  Args:
    image_collection: The image collection.
    max_images: Maximum number of rows to return.

  Returns:
    GeoDataFrame.
  """
  fc = ee.FeatureCollection(
      image_collection.limit(max_images, 'system:time_start', False)
      .toList(max_images)
      .map(_image_to_feature)
  )
  gdf = _download_feature_collection(fc)
  gdf['time'] = pd.to_datetime(gdf['time'], unit='ms')
  return gdf


def get_open_buildings(regions: List[ShapelyGeometry],
                       collection: str,
                       confidence: float,
                       as_centroids: bool,
                       output_path: str) -> gpd.GeoDataFrame:
  """Downloads Open Buildings footprints for the Area of Interest from EE.

  Args:
    regions: List of shapely Geometries to extract buildings from.
    collection: Name of Earth Engine FeatureCollection containing footprints.
    confidence: Confidence threshold for included buildings.
    as_centroids: If true, download centroids instead of full footprints.
    output_path: Save footprints to this file in addition to returning the
      GeoDataFrame.

  Returns:
    GeoDataFrame of building footprints.
  """
  bounds = ee.FeatureCollection([_shapely_to_ee_feature(r) for r in regions])
  open_buildings = ee.FeatureCollection(collection)
  aoi_buildings = open_buildings.filterBounds(bounds)
  aoi_buildings = aoi_buildings.filter(f'confidence >= {confidence}')
  if as_centroids:
    centroids = aoi_buildings.map(_get_open_building_feature_centroid)
    return _download_feature_collection_to_file(
        centroids, ['longitude', 'latitude'], output_path
    )
  else:
    return _download_feature_collection_to_file(
        aoi_buildings, ['.geo'], output_path
    )


def get_open_buildings_centroids(
    regions: List[ShapelyGeometry],
    collection: str,
    confidence: float,
    output_path: str) -> List[Tuple[float, float]]:
  """Downloads Open Buildings footprints as centroids of regions of interest.

  Args:
    regions: List of regions as shapely geometries.
    collection: Name of Earth Engine FeatureCollection containing footprints.
    confidence: Confidence threshold for included buildings.
    output_path: Save footprints to this file in addition to returning the
      GeoDataFrame.

  Returns:
    List of (longitude, latitude) building centroids.
  """
  gdf = get_open_buildings(regions, collection, confidence, True, output_path)
  return list(zip(gdf['geometry'].x, gdf['geometry'].y))


def initialize(service_account: str, private_key: Optional[str]) -> bool:
  """Initializes EE server connection.

  When not using a service account, this function assumes that the user has
  already authenticated with EE using the shell command
  "earthengine authenticate".

  Args:
    service_account: If not empty, the service account to use. Otherwise
      defaults to caller's personal account.
    private_key: Private key for service account.

  Returns:
    True if EE was successfully initialized, otherwise False.
  """
  try:
    if service_account:
      credentials = ee.ServiceAccountCredentials(service_account, private_key)
      ee.Initialize(credentials)
    else:
      ee.Initialize()
  except ee.EEException as e:
    logging.error('Error initializing Earth Engine: %s', e)
    return False
  return True


def find_covering_images(
    building_centroids: List[Tuple[float, float]],
    image_collection_id: str,
    project_ids: List[int],
    start_date: str,
    end_date: str,
    max_images: int) -> List[str]:
  """Finds a set of images that best covers a set of points.

  Args:
    building_centroids: Buildings to cover.
    image_collection_id: EE image collection to search from.
    project_ids: Project ids for images.
    start_date: Starting date, e.g. 2021-01-01.
    end_date: Ending date, e.g. 2023-12-31.
    max_images: Maximum number of images to return.

  Returns:
    A list of image ids that best covers the points.
  """
  points = gpd.GeoDataFrame(
      geometry=[shapely.geometry.Point(c) for c in building_centroids],
      crs=4326
  )
  aoi = _shapely_to_ee_feature(points.unary_union.convex_hull).geometry()
  image_collection = ee.ImageCollection(image_collection_id)
  if project_ids:
    project_filters = [
        ee.Filter.listContains('project_membership', pid) for pid in project_ids
    ]
    image_collection = image_collection.filter(ee.Filter.Or(*project_filters))

  images = image_collection.filterBounds(aoi).filterDate(start_date, end_date)
  images_gdf = _get_image_collection_gdf(images, _MAX_BEFORE_IMAGE_POOL)

  # Create a MxN matrix where M is the number of points and N is the number of
  # images. The element (i, j) is True if and only if point i intersects image
  # j.
  images_gdf_for_join = gpd.GeoDataFrame(geometry=images_gdf.geometry, crs=4326)
  joined = gpd.sjoin(images_gdf_for_join, points)
  intersections = np.zeros((len(points), len(images_gdf)), dtype=bool)
  intersections[joined.index_right.values, joined.index.values] = True

  # Use a greedy algorithm to choose images. Always choose the image that covers
  # the most number of remaining points.
  uncovered_points = np.ones(len(points), dtype=bool)
  unused_images = np.arange(len(images_gdf))
  chosen_image_ids = []
  while (
      np.any(uncovered_points)
      and len(unused_images)
      and len(chosen_image_ids) < max_images
  ):
    # num_points_covered is a K-length array, where K is the number of unused
    # images. num_points_covered[i] contains the number of remaining points
    # covered by the ith unused image.
    num_points_covered = np.sum(
        intersections[np.ix_(uncovered_points, unused_images)], axis=0)
    assert len(num_points_covered) == len(unused_images)
    best_index = np.argmax(num_points_covered)
    if num_points_covered[best_index] == 0:
      # Remaining images don't cover any more points, so terminate.
      break
    best_image = unused_images[best_index]
    chosen_image_ids.append(images_gdf.iloc[best_image]['image_id'])
    unused_images = np.delete(unused_images, [best_index])
    uncovered_points &= ~intersections[:, best_image]
    num_points_left = np.sum(uncovered_points)
    print(
        f'Chose image {best_image}, '
        f'covers {num_points_covered[best_index]} additional points, '
        f'{num_points_left} points left.'
    )

  return chosen_image_ids
