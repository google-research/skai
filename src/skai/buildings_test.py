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
"""Tests for buildings.py."""

import pathlib
import tempfile
from absl.testing import absltest
import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import shapely.geometry
from skai import buildings

Polygon = shapely.geometry.polygon.Polygon


def get_test_file_path(relative_test_data_path: str) -> str:
  """Returns path to a test data file.

  Args:
    relative_test_data_path: Relative data path, e.g. "test_data/blank.tif".

  Returns:
    Absolute path to test data.
  """
  current_dir = pathlib.Path(__file__).parent
  return str(current_dir / relative_test_data_path)


def get_temp_file(suffix: str = '') -> str:
  return tempfile.mkstemp(suffix=suffix, dir=absltest.TEST_TMPDIR.value)[1]


def create_test_buildings_gdf(num_buildings: int) -> gpd.GeoDataFrame:
  longitudes = np.random.uniform(low=-180, high=180, size=(num_buildings,))
  latitudes = np.random.uniform(low=-90, high=90, size=(num_buildings,))
  centroids = gpd.points_from_xy(longitudes, latitudes)
  ids = [str(i) for i in range(num_buildings)]
  return gpd.GeoDataFrame({'id': ids}, geometry=centroids, crs=4326)


class BuildingsTest(absltest.TestCase):

  def test_convert_buildings_file_from_geojson(self):
    path = get_test_file_path('test_data/building_centroids.geojson')
    output_path = get_temp_file()
    regions = [Polygon.from_bounds(178.78737, -16.65851, 178.81098, -16.63617)]
    buildings.convert_buildings_file(path, regions, output_path)
    buildings_gdf = buildings.read_buildings_file(output_path)

    # The input file, "building_centroids.geojson", has 102 centroids. However,
    # one centroid will be filtered out because it doesn't fall within the
    # region boundaries. Hence there should be 101 centroids.
    self.assertLen(buildings_gdf, 101)

  def test_convert_buildings_file_from_csv(self):
    path = get_test_file_path('test_data/building_centroids.csv')
    output_path = get_temp_file()
    regions = [Polygon.from_bounds(178.78737, -16.65851, 178.81098, -16.63617)]
    buildings.convert_buildings_file(path, regions, output_path)
    buildings_gdf = buildings.read_buildings_file(output_path)

    # The input file, "building_centroids.csv", has 102 centroids. However,
    # one centroid will be filtered out because it doesn't fall within the
    # region boundaries. Hence there should be 101 centroids.
    self.assertLen(buildings_gdf, 101)

  def test_convert_buildings_file_from_csv_with_wkt(self):
    buildings_data = [
        {
            'latitude': 29.99202352,
            'longitude': -4.99707854,
            'area_in_meters': 26.1683,
            'confidence': 0.8192,
            'geometry': (
                'POLYGON((-4.9970664523855 29.9919846910361, -4.99705832764922'
                ' 29.9920597015168, -4.99709061807397 29.992062351673,'
                ' -4.99709874278586 29.9919873411914, -4.9970664523855'
                ' 29.9919846910361))'
            ),
            'full_plus_code': '7CXQX2R3+R53V',
        },
        {
            'latitude': 29.99214936,
            'longitude': -4.99758957,
            'area_in_meters': 31.8337,
            'confidence': 0.8393,
            'geometry': (
                'POLYGON((-4.99758309542527 29.9921067182802, -4.99755994204994'
                ' 29.9921837854051, -4.99759604572576 29.9921920041646,'
                ' -4.99761919907427 29.9921149370345, -4.99758309542527'
                ' 29.9921067182802))'
            ),
            'full_plus_code': '7CXQX2R2+VX3R',
        },
    ]
    path = get_temp_file(suffix='.csv')
    pd.DataFrame(buildings_data).to_csv(path, index=False)
    regions = [Polygon.from_bounds(-180, -90, 180, 90)]
    output_path = get_temp_file()
    buildings.convert_buildings_file(path, regions, output_path)
    buildings_gdf = buildings.read_buildings_file(output_path)

    self.assertLen(buildings_gdf, 2)
    self.assertContainsSubset(
        [
            'longitude',
            'latitude',
            'area_in_meters',
            'confidence',
            'full_plus_code',
        ],
        buildings_gdf.columns,
    )
    expected_geometries = gpd.GeoSeries.from_wkt(
        [b['geometry'] for b in buildings_data], crs=4326
    )
    geopandas.testing.assert_geoseries_equal(
        buildings_gdf.geometry, expected_geometries, check_less_precise=True
    )

  def test_read_aois(self):
    path = get_test_file_path('test_data/aoi.geojson')
    aois = buildings.read_aois(path)
    self.assertLen(aois, 2)
    self.assertTrue(aois[0].almost_equals(
        Polygon([[178.785123173833853, -16.649524158515156],
                 [178.797216557828847, -16.642442516881495],
                 [178.809691443511156, -16.645264468917965],
                 [178.805980534182027, -16.650878537925099],
                 [178.795635728839926, -16.659201457281625],
                 [178.786057603145593, -16.656632874209055],
                 [178.785123173833853, -16.649524158515156]])))
    self.assertTrue(aois[1].almost_equals(
        Polygon([[178.823626242664915, -16.641714920481871],
                 [178.826624371965153, -16.644967777260199],
                 [178.823782315343209, -16.646737061078003],
                 [178.820579612784513, -16.644811608514196],
                 [178.823626242664915, -16.641714920481871]])))

  def test_read_write_building_files(self):
    path = get_temp_file()
    buildings_gdf = create_test_buildings_gdf(20)
    buildings.write_buildings_file(buildings_gdf, path)
    loaded_buildings_gdf = buildings.read_buildings_file(path)
    geopandas.testing.assert_geoseries_equal(
        buildings_gdf.geometry, loaded_buildings_gdf.geometry
    )
    self.assertContainsSubset(
        ['longitude', 'latitude', 'id'], loaded_buildings_gdf.columns
    )
    np.testing.assert_allclose(
        loaded_buildings_gdf.longitude.values,
        [c.x for c in buildings_gdf.geometry],
    )
    np.testing.assert_allclose(
        loaded_buildings_gdf.latitude.values,
        [c.y for c in buildings_gdf.geometry],
    )

  def test_read_write_building_coordinates(self):
    path = get_temp_file()
    buildings_gdf = create_test_buildings_gdf(20)
    buildings.write_buildings_file(buildings_gdf, path)
    coordinates_df = buildings.read_building_coordinates(path)
    self.assertContainsSubset(
        ['longitude', 'latitude'], coordinates_df.columns
    )
    np.testing.assert_allclose(
        coordinates_df.longitude.values,
        [c.x for c in buildings_gdf.geometry],
    )
    np.testing.assert_allclose(
        coordinates_df.latitude.values,
        [c.y for c in buildings_gdf.geometry],
    )

if __name__ == '__main__':
  absltest.main()
