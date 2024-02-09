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
import time
import urllib.request

from absl import logging
import geopandas as gpd
import pandas as pd
import shapely.geometry
from skai import buildings
import tensorflow as tf

import ee

ShapelyGeometry = shapely.geometry.base.BaseGeometry


def _shapely_to_ee_feature(shapely_geometry: ShapelyGeometry) -> ee.Feature:
  """Converts shapely geometry into Earth Engine Feature."""
  geojson = json.loads(gpd.GeoSeries([shapely_geometry]).to_json())
  return ee.Feature(geojson['features'][0]['geometry']).geometry()


def _get_open_building_feature_centroid(feature: ee.Feature) -> ee.Feature:
  """Extracts centroid information from Open Buildings feature."""
  centroid = ee.Geometry(feature.get('longitude_latitude')).coordinates()
  return ee.Feature(
      None, {'longitude': centroid.get(0), 'latitude': centroid.get(1)})


def _wait_for_task(task):
  task.start()
  while True:
    state = task.status()['state']
    if state == 'FAILED':
      error_message = task.status()['error_message']
      raise RuntimeError(f'EE task failed with message "{error_message}"')
    elif state not in ['UNSUBMITTED', 'READY', 'RUNNING']:
      return
    else:
      print(f'Task state: {state}')
      time.sleep(5)


def _download_feature_collection(
    collection: ee.FeatureCollection, properties: list[str],
    output_path: str) -> None:
  """Downloads a FeatureCollection from Earth Engine as a GeoDataFrame.

  Args:
    collection: EE FeatureCollection to download.
    properties: List of properties to download.
    output_path: Path of GeoDataFrame feather output.
  """
  temp_path = output_path + '.temp.csv'
  if temp_path.startswith('gs://'):
    bucket, _, file_name_prefix = temp_path[5:].partition('/')
    task = ee.batch.Export.table.toCloudStorage(
        collection=collection,
        fileNamePrefix=file_name_prefix[:-4],
        bucket=bucket,
        fileFormat='CSV',
        selectors=properties,
    )
    _wait_for_task(task)
  elif temp_path.startswith('/bigstore/'):
    bucket, file_name_prefix = temp_path.split('/', maxsplit=3)[2:]
    task = ee.batch.Export.table.toCloudStorage(
        collection=collection,
        fileNamePrefix=file_name_prefix[:-4],
        bucket=bucket,
        fileFormat='CSV',
        selectors=properties,
    )
    _wait_for_task(task)
  else:
    url = collection.getDownloadURL('csv', properties)
    with urllib.request.urlopen(url) as url_file, tf.io.gfile.GFile(
        temp_path, 'wb') as output:
      shutil.copyfileobj(url_file, output)

  with tf.io.gfile.GFile(temp_path, 'r') as f:
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
    properties = df
  else:
    raise ValueError('No geometries found in feature collection.')

  buildings.write_buildings_file(
      gpd.GeoDataFrame(properties, geometry=geometry, crs=4326), output_path
  )


def get_open_buildings(regions: list[ShapelyGeometry],
                       collection: str,
                       confidence: float,
                       as_centroids: bool,
                       output_path: str) -> None:
  """Downloads Open Buildings footprints for the Area of Interest from EE.

  Args:
    regions: List of shapely Geometries to extract buildings from.
    collection: Name of Earth Engine FeatureCollection containing footprints.
    confidence: Confidence threshold for included buildings.
    as_centroids: If true, download centroids instead of full footprints.
    output_path: Path to save footprints to.
  """
  bounds = ee.FeatureCollection(
      [_shapely_to_ee_feature(r) for r in regions]
  ).geometry()
  open_buildings = ee.FeatureCollection(collection)
  aoi_buildings = open_buildings.filterBounds(bounds)
  aoi_buildings = aoi_buildings.filter(f'confidence >= {confidence}')
  if as_centroids:
    centroids = aoi_buildings.map(_get_open_building_feature_centroid)
    _download_feature_collection(
        centroids,
        [
            'longitude',
            'latitude',
            'full_plus_code',
            'confidence',
            'area_in_meters',
        ],
        output_path,
    )
  else:
    _download_feature_collection(
        aoi_buildings,
        ['.geo', 'full_plus_code', 'confidence', 'area_in_meters'],
        output_path,
    )


def initialize(service_account: str, private_key: str | None) -> bool:
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
