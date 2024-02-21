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

"""Merges labels from the Vertex AI labeling tool with original TF Examples.

After a dataset is labeled on the Vertex AI labeling tool, the labels must be
merged with the original Tensorflow Examples in order to create a labeled
training and test set.
"""
import multiprocessing
import random

from absl import app
from absl import flags

from skai import cloud_labeling
from skai import labeling

FLAGS = flags.FLAGS
flags.DEFINE_string('cloud_project', None, 'GCP project name.')
flags.DEFINE_string('cloud_location', None, 'Project location.')
flags.DEFINE_list('cloud_dataset_ids', [], 'Dataset IDs.')
flags.DEFINE_list('label_file_paths', [], 'List of paths to label files')
flags.DEFINE_string('cloud_temp_dir', None, 'GCS temporary directory.')
flags.DEFINE_string('examples_pattern', None,
                    'Pattern for TFRecords of examples to merge with labels.',
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
  if FLAGS.cloud_dataset_ids:
    cloud_labeling.create_labeled_examples(
        FLAGS.cloud_project,
        FLAGS.cloud_location,
        FLAGS.cloud_dataset_ids,
        FLAGS.label_file_paths,
        FLAGS.string_to_numeric_labels,
        FLAGS.cloud_temp_dir,
        FLAGS.examples_pattern,
        FLAGS.test_fraction,
        FLAGS.train_output_path,
        FLAGS.test_output_path,
        FLAGS.connecting_distance_meters,
        FLAGS.use_multiprocessing)
  else:
    labeling.create_labeled_examples(
        FLAGS.label_file_paths,
        FLAGS.string_to_numeric_labels,
        FLAGS.examples_pattern,
        FLAGS.test_fraction,
        FLAGS.train_output_path,
        FLAGS.test_output_path,
        FLAGS.connecting_distance_meters,
        FLAGS.use_multiprocessing,
        None,
        FLAGS.max_processes)


if __name__ == '__main__':
  app.run(main)
