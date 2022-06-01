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
"""Tests for open_street_map.py."""

from absl.testing import absltest
import shapely.geometry

from skai import open_street_map

Polygon = shapely.geometry.polygon.Polygon
Point = shapely.geometry.point.Point


class OpenStreetMapTest(absltest.TestCase):

  def testReadNodes(self):
    xml = """
    <?xml version="1.0" encoding="UTF-8"?>
    <osm version="0.6" generator="Overpass API 0.7.57.1 74a55df1">
    <meta osm_base="2021-11-15T20:15:42Z"/>

      <node id="123" lat="37" lon="-122"/>
      <node id="456" lat="38" lon="-123">
        <tag k="highway" v="turning_circle"/>
      </node>
      <node id="789" lat="-10" lon="100"/>
    </osm>""".strip()

    region = Polygon.from_bounds(-130, 0, 0, 40)
    nodes = open_street_map._read_nodes(xml, region)
    self.assertSameStructure(nodes, {
        '123': Point(-122.0, 37.0),
        '456': Point(-123.0, 38.0)
    })

  def testReadPolygons(self):
    xml = """
    <?xml version="1.0" encoding="UTF-8"?>
    <osm version="0.6" generator="Overpass API 0.7.57.1 74a55df1">
    <meta osm_base="2021-11-15T20:15:42Z"/>

      <way id="1">
        <nd ref="11"/>
        <nd ref="12"/>
        <nd ref="13"/>
        <tag k="highway" v="residential"/>
      </way>
      <way id="2">
        <nd ref="21"/>
        <nd ref="22"/>
        <nd ref="23"/>
        <nd ref="21"/>
      </way>
      <way id="3">
        <nd ref="31"/>
        <nd ref="31"/>
      </way>
      <way id="4">
        <nd ref="41"/>
        <nd ref="42"/>
        <nd ref="43"/>
        <nd ref="44"/>
        <nd ref="41"/>
      </way>
      <way id="5">
        <nd ref="51"/>
        <nd ref="52"/>
        <nd ref="53"/>
        <nd ref="51"/>
      </way>
    </osm>""".strip()

    nodes = {
        '11': Point(-122.0, 30.0),
        '12': Point(-122.0, 31.0),
        '13': Point(-122.0, 32.0),
        '21': Point(-122.0, 33.0),
        '22': Point(-122.0, 34.0),
        '23': Point(-122.0, 35.0),
        '24': Point(-122.0, 36.0),
        '31': Point(-122.0, 37.0),
        '41': Point(-122.0, 38.0),
        '42': Point(-122.0, 39.0),
        '43': Point(-122.0, 40.0),
        '44': Point(-122.0, 41.0),
        '45': Point(-122.0, 42.0),
        '51': Point(-122.0, 43.0),
        '52': Point(-122.0, 44.0),
    }
    polygons = open_street_map._read_polygons(xml, nodes)
    self.assertLen(polygons, 2)
    self.assertCountEqual(polygons, [
        Polygon([(-122.0, 33.0), (-122.0, 34.0), (-122.0, 35.0),
                 (-122.0, 33.0)]),
        Polygon([(-122.0, 38.0), (-122.0, 39.0), (-122.0, 40.0), (-122.0, 41.0),
                 (-122.0, 38.0)]),
    ])


if __name__ == '__main__':
  absltest.main()
