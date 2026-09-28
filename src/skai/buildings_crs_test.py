# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Building coordinate columns must be written in geographic coordinates."""

import pathlib
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from absl.testing import absltest, parameterized
from geopandas import testing as geo_testing
from shapely import geometry

from skai import buildings


class BuildingsCrsTest(parameterized.TestCase):

  def _round_trip(self, frame):
    original = frame.copy(deep=True)
    with tempfile.TemporaryDirectory() as directory:
      path = str(pathlib.Path(directory) / 'buildings.parquet')
      buildings.write_buildings_file(frame, path)
      restored = buildings.read_buildings_file(path)
      coordinates = buildings.read_building_coordinates(path)
    geo_testing.assert_geodataframe_equal(frame, original)
    geo_testing.assert_geoseries_equal(
        restored.geometry,
        frame.to_crs(4326).geometry.reset_index(drop=True),
    )
    self.assertEqual(restored.crs.to_epsg(), 4326)
    self.assertEqual(list(restored['building_id']), list(frame['building_id']))
    pd.testing.assert_frame_equal(
        restored[['longitude', 'latitude']], coordinates
    )
    return coordinates

  @parameterized.product(
      location=[(3857, -3.25, 55.5), (32630, -3.25, 55.5),
                (32756, 152.1, -33.3)],
      kind=['point', 'polygon', 'multipolygon'],
      custom_index=[False, True],
  )
  def test_projected_centroids_are_transformed(self, location, kind,
                                             custom_index):
    epsg, longitude, latitude = location
    x, y = pyproj.Transformer.from_crs(
        4326, epsg, always_xy=True
    ).transform(longitude, latitude)
    centers = [(x, y), (x + 200, y + 150)]
    shapes = []
    for cx, cy in centers:
      if kind == 'point':
        shapes.append(geometry.Point(cx, cy))
      elif kind == 'polygon':
        shapes.append(geometry.box(cx - 10, cy - 20, cx + 10, cy + 20))
      else:
        shapes.append(geometry.MultiPolygon([
            geometry.box(cx - 40, cy - 10, cx - 20, cy + 10),
            geometry.box(cx + 20, cy - 10, cx + 40, cy + 10),
        ]))
    frame = gpd.GeoDataFrame(
        {'building_id': ['first', 'second']}, geometry=shapes, crs=epsg,
        index=[19, 3] if custom_index else None,
    )
    expected = np.column_stack(pyproj.Transformer.from_crs(
        epsg, 4326, always_xy=True
    ).transform(*np.array(centers).T))
    actual = self._round_trip(frame)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

  @parameterized.parameters('point', 'polygon')
  def test_geographic_input_retains_its_coordinates(self, kind):
    shapes = [geometry.Point(-3.25, 55.5)]
    if kind == 'polygon':
      shapes = [geometry.box(-3.26, 55.49, -3.24, 55.51)]
    frame = gpd.GeoDataFrame(
        {'building_id': ['first']}, geometry=shapes, crs=4326
    )
    actual = self._round_trip(frame)
    np.testing.assert_allclose(actual, [[-3.25, 55.5]], rtol=0, atol=1e-12)

  @parameterized.parameters(4326, 3857)
  def test_explicit_coordinate_columns_are_preserved(self, epsg):
    frame = gpd.GeoDataFrame(
        {'building_id': ['first'], 'longitude': [12.25], 'latitude': [-3.75]},
        geometry=[geometry.Point(0, 0)], crs=epsg,
    )
    np.testing.assert_array_equal(self._round_trip(frame), [[12.25, -3.75]])


if __name__ == '__main__':
  absltest.main()
