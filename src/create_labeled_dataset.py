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
import random

from absl import app
from absl import flags

from skai import cloud_labeling

FLAGS = flags.FLAGS
flags.DEFINE_string('cloud_project', None, 'GCP project name.', required=True)
flags.DEFINE_string('cloud_location', None, 'Project location.', required=True)
flags.DEFINE_string('cloud_dataset_id', None, 'Dataset ID.', required=True)
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


def main(unused_argv):
  if FLAGS.random_seed is not None:
    random.seed(FLAGS.random_seed)

  cloud_labeling.create_labeled_examples(
      FLAGS.cloud_project,
      FLAGS.cloud_location,
      FLAGS.cloud_dataset_id,
      FLAGS.cloud_temp_dir,
      FLAGS.examples_pattern,
      FLAGS.test_fraction,
      FLAGS.train_output_path,
      FLAGS.test_output_path)


if __name__ == '__main__':
  app.run(main)
