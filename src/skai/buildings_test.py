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
from absl.testing import absltest
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


class BuildingsTest(absltest.TestCase):

  def testReadBuildingsFileGeoJSON(self):
    path = get_test_file_path('test_data/building_centroids.geojson')
    regions = [Polygon.from_bounds(178.78737, -16.65851, 178.81098, -16.63617)]
    coords = buildings.read_buildings_file(path, regions)

    # The input file, "building_centroids.geojson", has 102 centroids. However,
    # one centroid will be filtered out because it doesn't fall within the
    # region boundaries. Hence there should be 101 centroids.
    self.assertLen(coords, 101)

  def testReadBuildingsFileCSV(self):
    path = get_test_file_path('test_data/building_centroids.csv')
    regions = [Polygon.from_bounds(178.78737, -16.65851, 178.81098, -16.63617)]
    coords = buildings.read_buildings_file(path, regions)

    # The input file, "building_centroids.csv", has 102 centroids. However,
    # one centroid will be filtered out because it doesn't fall within the
    # region boundaries. Hence there should be 101 centroids.
    self.assertLen(coords, 101)

  def testReadAOIs(self):
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


if __name__ == '__main__':
  absltest.main()
