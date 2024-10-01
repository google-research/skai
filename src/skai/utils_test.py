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

"""Tests for utils."""
import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

import geopandas as gpd
import shapely.geometry

from skai import utils


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          longitude=38.676355712551015,
          latitude=58.92948405901603,
          correct_utm='EPSG:32637',
      ),
      dict(
          longitude=74.38574398625497,
          latitude=6.082091014059927,
          correct_utm='EPSG:32643',
      ),
      dict(
          longitude=19.09160680029879,
          latitude=30.920059879467274,
          correct_utm='EPSG:32634',
      ),
      dict(
          longitude=-16.321149967123773,
          latitude=62.08112825201508,
          correct_utm='EPSG:32628',
      ),
      dict(
          longitude=149.78080000677284,
          latitude=25.31143284356746,
          correct_utm='EPSG:32655',
      )
  )
  def test_get_utm_crs(self, longitude, latitude, correct_utm):
    self.assertEqual(utils.get_utm_crs(longitude, latitude), correct_utm)

  def test_convert_to_utm(self):
    points = [
        shapely.geometry.Point(30, 50),
        shapely.geometry.Point(35, 55),
        shapely.geometry.Point(25, 45),
    ]
    gdf = gpd.GeoDataFrame(geometry=points, crs=4326)
    utm_gdf = utils.convert_to_utm(gdf)
    self.assertEqual(str(utm_gdf.crs), 'EPSG:32636')

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_pattern',
          files=[],
          patterns=[],
      ),
      dict(
          testcase_name='no_match',
          files=['a.tif', 'b.tif'],
          patterns=['*.tif', 'c.tif'],
      ),
      dict(
          testcase_name='duplicate_matches',
          files=['a.tif', 'b.tif'],
          patterns=['*.tif', 'a.*'],
      ),
  )
  def testExpandPatternsRaises(self, files, patterns):
    temp_dir = pathlib.Path(tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value))
    patterns_with_dir = [str(temp_dir / p) for p in patterns]
    for f in files:
      (temp_dir / f).touch()
    with self.assertRaises(ValueError):
      utils.expand_file_patterns(patterns_with_dir)

  @parameterized.named_parameters(
      dict(
          testcase_name='match',
          files=['a.tif', 'b.tif', 'c.jpg'],
          patterns=['*.tif'],
          expected_matches=['a.tif', 'b.tif'],
      ),
  )
  def testExpandPatternsNoRaises(self, files, patterns, expected_matches):
    temp_dir = pathlib.Path(tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value))
    patterns_with_dir = [str(temp_dir / p) for p in patterns]
    expected_matches_with_dir = [str(temp_dir / f) for f in expected_matches]
    for f in files:
      (pathlib.Path(temp_dir) / f).touch()
    matches = utils.expand_file_patterns(patterns_with_dir)
    self.assertSameElements(expected_matches_with_dir, matches)


if __name__ == '__main__':
  absltest.main()
