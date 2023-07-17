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
"""Tests for beam_utils.py."""

import os
import tempfile

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline

from skai import beam_utils


class BeamUtilsTest(absltest.TestCase):

  def testWriteRecordsAsFiles(self):
    output_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
    temp_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)

    with test_pipeline.TestPipeline() as pipeline:
      records = (
          pipeline
          | beam.Create([('abc.txt', b'abc'), ('def.csv', b'def')]))
      beam_utils.write_records_as_files(records, output_dir, temp_dir, 'write')

    self.assertCountEqual(os.listdir(output_dir), ['abc.txt', 'def.csv'])

    with open(os.path.join(output_dir, 'abc.txt'), 'r') as f:
      self.assertEqual(f.read(), 'abc')

    with open(os.path.join(output_dir, 'def.csv'), 'r') as f:
      self.assertEqual(f.read(), 'def')


if __name__ == '__main__':
  absltest.main()
