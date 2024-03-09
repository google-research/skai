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

r"""Drops all overlapping examples and outputs filtered tfrecords.

This script filters and drops overlapping examples.

Example invocation:

python drop_overlapping_examples.py \
  --examples_pattern=gs://bucket/disaster/examples/unlabeled-large/*.tfrecord \
  --output_dir=gs://bucket/disaster/filtered_examples/ \

"""
# pylint: enable=line-too-long

import multiprocessing
import sys

from absl import app
from absl import flags
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
flags.DEFINE_bool(
    'use_multiprocessing',
    True,
    'If true, starts multiple processes to run task.',
)
flags.DEFINE_integer(
    'max_processes',
    multiprocessing.cpu_count(),
    'If using multiprocessing, the maximum number of processes to use.',
)
flags.DEFINE_float(
    'buffered_sampling_radius',
    78.0,
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

  labeling.create_buffered_tfrecords(
      FLAGS.examples_pattern,
      FLAGS.output_dir,
      FLAGS.use_multiprocessing,
      None,
      FLAGS.exclude_import_file_patterns,
      FLAGS.max_processes,
      FLAGS.buffered_sampling_radius,
  )


if __name__ == '__main__':
  app.run(main)
