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

import io

from skai import cloud_labeling
from absl.testing import absltest

import PIL.Image
import tensorflow.compat.v1 as tf


def _create_example():
  image = PIL.Image.new("RGB", (64, 64))
  buffer = io.BytesIO()
  image.save(buffer, format='PNG')
  image_bytes = buffer.getvalue()

  example = tf.train.Example()
  example.features.feature["pre_image_png"].bytes_list.value.append(
      image_bytes)
  example.features.feature["post_image_png"].bytes_list.value.append(
      image_bytes)
  example.features.feature["coordinates"].float_list.value.extend([45, 90])
  return example


class CloudLabelingTest(absltest.TestCase):

  def testCreateLabelingImageBasic(self):
    example = _create_example()
    labeling_image = cloud_labeling.create_labeling_image_from_example(example)
    self.assertEqual(labeling_image.width, 158)
    self.assertEqual(labeling_image.height, 116)


  def testGetLabelingImageAnnotations(self):
    example = _create_example()
    annotations = cloud_labeling.get_labeling_image_annotations(
        example,
        'gs://output/path',
        'gs://src/path/examples.tfrecord',
        2)
    self.assertEqual(annotations, {
        'dataItemResourceLabels': {
            'origin': 'examples__tfrecord',
            'index': 2,
            'longitude_e8': '4500000000',
            'latitude_e8': '9000000000'},
        'imageGcsUri': 'gs://output/path'})


if __name__ == '__main__':
  absltest.main()
