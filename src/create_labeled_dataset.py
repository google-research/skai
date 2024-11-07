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

"""Creates labeled train and test data from labels and TFRecords.

There are 3 sources of labels that this script can use.

1. Download labels from a VertexAI dataset. You must provide a value for the
   "cloud_dataset_ids" flag.

2. A CSV file. You must provide a value for the "label_file_paths" flag.
   The files must have "example_id" and "string_label" columns. This serves as
   a mapping from examples to labels.

3. Use the label features already in the examples. You must not provide values
   for either of the above flags. This can be used to reshuffle existing labeled
   examples into new train and test splits, or to combine multiple existing
   labeled datasets.
"""

import multiprocessing
import random

from absl import app
from absl import flags

from skai import labeling

FLAGS = flags.FLAGS
flags.DEFINE_list('label_file_paths', [], 'List of paths to label files')
flags.DEFINE_string('cloud_temp_dir', None, 'GCS temporary directory.')
flags.DEFINE_list('example_patterns', None,
                  'Patterns for TFRecords of examples to merge with labels.',
                  required=True)
flags.DEFINE_string('train_output_path', None,
                    'Path to output labeled training examples.', required=True)
flags.DEFINE_string('test_output_path', None,
                    'Path to output labeled test examples.', required=True)
flags.DEFINE_integer('random_seed', None,
                     'If specified, random seed for train/test split.')
flags.DEFINE_float('test_fraction', 0.2,
                   'Fraction of labeled examples to use for testing.')
flags.DEFINE_float('connecting_distance_meters', 77.0,
                   'Maximum distance for two points to be connected.')
flags.DEFINE_list(
    'string_to_numeric_labels',
    [
        'bad_example=0',
        'no_damage=0',
        'minor_damage=1',
        'major_damage=1',
        'destroyed=1',
    ],
    'List of "class=label" strings, e.g. "no_damage=0,minor_damage=0,...".',
)
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


def main(unused_argv):
  if FLAGS.random_seed is not None:
    random.seed(FLAGS.random_seed)
  labeling.create_labeled_examples(
      FLAGS.label_file_paths,
      FLAGS.string_to_numeric_labels,
      FLAGS.example_patterns,
      FLAGS.test_fraction,
      FLAGS.train_output_path,
      FLAGS.test_output_path,
      FLAGS.connecting_distance_meters,
      FLAGS.use_multiprocessing,
      None,
      FLAGS.max_processes)


if __name__ == '__main__':
  app.run(main)
