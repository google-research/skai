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

"""Tests for cloud_labeling."""

import os
import tempfile

from absl.testing import absltest
import PIL.Image
from skai import cloud_labeling


class CloudLabelingTest(absltest.TestCase):

  def testCreateLabelingImageBasic(self):
    before_image = PIL.Image.new('RGB', (64, 64))
    after_image = PIL.Image.new('RGB', (64, 64))
    labeling_image = cloud_labeling.create_labeling_image(
        before_image, after_image)
    self.assertEqual(labeling_image.width, 158)
    self.assertEqual(labeling_image.height, 116)

  def testWriteImportFile(self):
    images_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
    image_files = [
        os.path.join(images_dir, f) for f in ['a.png', 'b.png', 'c.png']
    ]
    for filename in image_files:
      open(filename, 'w').close()
    output_path = os.path.join(absltest.TEST_TMPDIR.value, 'import_file.csv')
    cloud_labeling.write_import_file(images_dir, 2, False, output_path)
    with open(output_path, 'r') as f:
      contents = [line.strip() for line in f.readlines()]
    self.assertEqual(contents, image_files[:2])

  def testWriteImportFileMaxImages(self):
    images_dir = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
    image_files = [
        os.path.join(images_dir, f) for f in ['a.png', 'b.png', 'c.png']
    ]
    for filename in image_files:
      open(filename, 'w').close()
    output_path = os.path.join(absltest.TEST_TMPDIR.value, 'import_file.csv')
    cloud_labeling.write_import_file(images_dir, 5, False, output_path)
    with open(output_path, 'r') as f:
      contents = [line.strip() for line in f.readlines()]
    self.assertEqual(contents, image_files)


if __name__ == '__main__':
  absltest.main()
