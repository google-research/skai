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
from typing import List

from absl.testing import absltest
import pandas as pd
import PIL.Image
from skai import cloud_labeling
import tensorflow as tf

Example = tf.train.Example


def _read_tfrecord(path: str) -> List[Example]:
  examples = []
  for record in tf.data.TFRecordDataset([path]):
    example = Example()
    example.ParseFromString(record.numpy())
    examples.append(example)
  return examples


class CloudLabelingTest(absltest.TestCase):

  def testCreateLabelingImageBasic(self):
    before_image = PIL.Image.new('RGB', (64, 64))
    after_image = PIL.Image.new('RGB', (64, 64))
    labeling_image = cloud_labeling.create_labeling_image(
        before_image, after_image, 'example_id', 'plus_code')
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

  def testCreateLabeledExamplesFromLabelFile(self):
    # Create unlabeled examples.
    _, unlabeled_examples_path = tempfile.mkstemp(
        dir=absltest.TEST_TMPDIR.value)
    with tf.io.TFRecordWriter(unlabeled_examples_path) as writer:
      for i, example_id in enumerate(['a', 'b', 'c', 'd']):
        example = Example()
        example.features.feature['example_id'].bytes_list.value.append(
            example_id.encode()
        )
        example.features.feature['encoded_coordinates'].bytes_list.value.append(
            str(i).encode()
        )
        writer.write(example.SerializeToString())

    # Create a label file.
    _, label_file_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
    label_file_contents = pd.DataFrame([
        ('a', 'no_damage'),
        ('b', 'minor_damage'),
        ('c', 'major_damage'),
        ('c', 'no_damage'),
        ('d', 'destroyed'),
        ('d', 'bad_example')], columns=['example_id', 'string_label'])
    label_file_contents.to_csv(label_file_path, index=False)

    _, train_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
    _, test_path = tempfile.mkstemp(dir=absltest.TEST_TMPDIR.value)
    cloud_labeling.create_labeled_examples(
        project=None,
        location=None,
        dataset_ids=[],
        label_file_paths=[label_file_path],
        string_to_numeric_labels=[
            'no_damage=0',
            'minor_damage=0',
            'major_damage=1',
            'destroyed=1',
            'bad_example=0',
        ],
        export_dir=None,
        examples_pattern=unlabeled_examples_path,
        test_fraction=0.333,
        train_output_path=train_path,
        test_output_path=test_path,
        use_multiprocessing=False)

    all_examples = _read_tfrecord(train_path) + _read_tfrecord(test_path)
    self.assertLen(all_examples, 6)

    id_to_float_label = [
        (
            e.features.feature['example_id'].bytes_list.value[0].decode(),
            e.features.feature['label'].float_list.value[0],
        )
        for e in all_examples
    ]

    self.assertSameElements(
        id_to_float_label,
        [
            ('a', 0.0),
            ('b', 0.0),
            ('c', 1.0),
            ('c', 0.0),
            ('d', 0.0),
            ('d', 1.0),
        ],
    )


if __name__ == '__main__':
  absltest.main()
