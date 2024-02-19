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
"""Fetches building footprints from OpenStreetMap using Overpass API.

Please see https://wiki.openstreetmap.org/wiki/Overpass_API for details.
"""

from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
import geopandas as gpd
import requests
import shapely.geometry

from skai import buildings

Polygon = shapely.geometry.polygon.Polygon
Point = shapely.geometry.point.Point


def _read_nodes(xml: str, region: Polygon) -> Dict[str, Point]:
  """Parses OSM Overpass response XML into a dict of points.

  Args:
    xml: XML response from an Overpass "node" query.
    region: Region of interest.

  Returns:
    Dictionary mapping node id to (longitude, latitude) points.
  """
  root = ET.fromstring(xml)
  nodes = {}
  for element in root:
    if element.tag == 'node':
      point = Point(float(element.attrib['lon']), float(element.attrib['lat']))
      if region.contains(point):
        nodes[element.attrib['id']] = point
  return nodes


def _read_closed_way(element: ET.Element,
                     nodes: Dict[str, Point]) -> Optional[Polygon]:
  """Parses an Overpass way element into a polygon.

  Only returns a polygon if (1) all nodes are known, and (2) the way is an
  actual polygon (i.e. it has at least 3 points and is closed).

  Args:
    element: XML way element.
    nodes: Dictionary mapping node ids to coordinates.

  Returns:
    The parsed polygon if the way is a closed polygon.
  """
  node_refs = []
  for child in element:
    if child.tag == 'nd':
      node_refs.append(child.attrib['ref'])

  if len(node_refs) <= 2 or node_refs[0] != node_refs[-1]:
    return None

  coords = []
  for node_ref in node_refs:
    try:
      point = nodes[node_ref]
    except KeyError:
      return None
    coords.append((point.x, point.y))

  return Polygon(coords)


def _read_polygons(xml: str, nodes: Dict[str, Point]) -> List[Polygon]:
  """Parses OSM Overpass response XML into a list of polygons.

  Args:
    xml: XML response from an Overpass "way" query.
    nodes: Dictionary mapping node ids to coordinates.

  Returns:
    List of polygons.
  """
  root = ET.fromstring(xml)
  polygons = []
  for element in root:
    if element.tag == 'way':
      # Only get closed polygons
      polygon = _read_closed_way(element, nodes)
      if polygon:
        polygons.append(polygon)
  return polygons


def get_buildings_in_region(region: Polygon,
                            overpass_url: str) -> List[Polygon]:
  """Queries OpenStreetMap Overpass API for all buildings in a region.

  Args:
    region: Region of interest polygon. Must be in EPSG:4326 (long/lat).
    overpass_url: Overpass URL e.g. https://lz4.overpass-api.de/api/interpreter.

  Returns:
    A list of building polygons in the region.
  """
  left, bottom, right, top = region.bounds
  node_query = f'node({bottom},{left},{top},{right});out;'
  ways_query = f'way[building]({bottom},{left},{top},{right});out;'

  r = requests.post(overpass_url, data=node_query)
  nodes = _read_nodes(r.text, region)

  r = requests.post(overpass_url, data=ways_query)
  polygons = _read_polygons(r.text, nodes)
  return polygons


def get_building_centroids_in_regions(
    regions: List[Polygon], overpass_url: str, output_path: str) -> None:
  """Queries OpenStreetMap Overpass API for all building centroids in a region.

  Args:
    regions: Regions of interest as polygons. Must be in EPSG:4326 (long/lat).
    overpass_url: Overpass URL e.g. https://lz4.overpass-api.de/api/interpreter.
    output_path: Save footprints to this path as a GeoPackage.
  """
  polygons = []
  for region in regions:
    polygons.extend(get_buildings_in_region(region, overpass_url))
  buildings.write_buildings_file(
      gpd.GeoDataFrame(geometry=polygons, crs=4326), output_path
  )
