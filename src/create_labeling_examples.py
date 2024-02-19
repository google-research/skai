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

# pylint: disable=line-too-long

r"""Samples labeling examples, generates labeling images and metadata for use in creating labeling tasks.

This script converts a subset of unlabeled TF examples into labeling image
format (before and after images of each building juxtaposed side-by-side),
saves the images to a path.[1]

Example invocation:

python labeling_task.py \
  --examples_pattern=gs://bucket/disaster/examples/unlabeled-large/*.tfrecord \
  --output_dir=gs://bucket/disaster/examples/labeling-images \
  --max_images=1000 \

"""
# pylint: enable=line-too-long

import sys

from absl import app
from absl import flags
from absl import logging
from skai import labeling
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'examples_pattern', None, 'Pattern matching TFRecords.', required=True
)
flags.DEFINE_string(
    'output_dir', None, 'Directory to write images to.', required=True
)
flags.DEFINE_list('exclude_import_file_patterns', [],
                  'File patterns for import files listing images not to '
                  'generate.')
flags.DEFINE_integer('max_images', 1000, 'Maximum number of images to label.')
flags.DEFINE_string(
    'allowed_example_ids_path',
    None,
    'If specified, only allow example ids found in this text file.')
flags.DEFINE_bool(
    'use_multiprocessing',
    True,
    'If true, starts multiple processes to run task.',
)
flags.DEFINE_float(
    'buffered_sampling_radius',
    70.0,
    'The minimum distance between two examples for the two examples to be in'
    ' the labeling task.',
)


def main(unused_argv):
  # Check if --output_dir is empty
  if tf.io.gfile.isdir(FLAGS.output_dir) and tf.io.gfile.listdir(
      FLAGS.output_dir
  ):
    sys.exit(
        f'\nError: {FLAGS.output_dir} is not empty.\n\nUse an empty directory'
        ' for --output_dir.'
    )

  num_images, images_to_label_import_csv_path = labeling.create_labeling_images(
      FLAGS.examples_pattern,
      FLAGS.max_images,
      FLAGS.allowed_example_ids_path,
      FLAGS.exclude_import_file_patterns,
      FLAGS.output_dir,
      FLAGS.use_multiprocessing,
      None,
      FLAGS.buffered_sampling_radius,
  )

  if num_images == 0:
    logging.fatal('No labeling images found.')
    return

  logging.info('Wrote %d labeling images.', num_images)
  logging.info(
      'Wrote images to label import file %s.', images_to_label_import_csv_path
  )

if __name__ == '__main__':
  app.run(main)
