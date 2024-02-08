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
from absl.testing import absltest
from absl.testing import parameterized
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
  def testConvertWGStoUTM(self, longitude, latitude, correct_utm):
    self.assertEqual(utils.convert_wgs_to_utm(longitude, latitude), correct_utm)


if __name__ == '__main__':
  absltest.main()
