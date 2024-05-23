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

python create_labeling_examples.py \
  --examples_pattern=gs://bucket/disaster/examples/unlabeled-large/*.tfrecord \
  --output_dir=gs://bucket/disaster/examples/labeling-images \
  --max_images=1000 \

"""
# pylint: enable=line-too-long

import ast
import multiprocessing
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
flags.DEFINE_list(
    'exclude_import_file_patterns',
    [],
    'File patterns for import files listing images not to generate.',
)
flags.DEFINE_integer('max_images', 1000, 'Maximum number of images to label.')
flags.DEFINE_string(
    'allowed_example_ids_path',
    None,
    'If specified, only allow example ids found in this text file.',
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
flags.DEFINE_float(
    'buffered_sampling_radius',
    78.0,
    'The minimum distance between two examples for the two examples to be in'
    ' the labeling task.',
)
flags.DEFINE_string(
    'scores_path',
    None,
    'Path to a CSV with example_id and score columns that assigns a score to'
    ' each example, for sampling based on example scores.',
)
flags.DEFINE_string(
    'score_bins_to_sample_fraction',
    '{'
    '(0.0, 0.25): 0.25,'
    '(0.25, 0.5): 0.25,'
    '(0.5, 0.75): 0.25,'
    '(0.75, 1.0): 0.25,'
    '}',
    'Dictionary mapping bins of score ranges to the fraction of the sample that'
    ' should be drawn from each bin. The value should be a string representing'
    ' a literal Python dict of tuples (low score, high score) to a fraction'
    ' from 0 - 1.0. The fractions should sum to 1.0.',
)

flags.DEFINE_string(
    'filter_by_column',
    None,
    'If specified, the name of the column in the scores CSV file to use as a'
    ' filter. The column must contain binary values, either true/false or 0/1.'
    ' Rows with positive values in this column are then filtered out.',
)


def validate_score_bins(score_bins_to_sample_fraction):
  """Validates the score_bins_to_sample_fraction flag.

  Args:
    score_bins_to_sample_fraction: Dictionary mapping bins of score ranges to
      the fraction of the sample that should be drawn from each bin. The value
      should be a string representing a literal Python dict of tuples (low
      score, high score) to a fraction from 0 - 1.0. The fractions should sum to
      1.0.'.

  Returns:
    True if the score_bins_to_sample_fraction are valid.
  """
  score_bins_to_sample_fraction = ast.literal_eval(
      score_bins_to_sample_fraction
  )
  total_percentage = 0
  for bin_interval, percentage in score_bins_to_sample_fraction.items():
    total_percentage += percentage
    assert percentage >= 0, f'{percentage} should be non-negative.'
    assert (
        bin_interval[1] > bin_interval[0]
    ), f'{bin_interval[1]} should be greater than {bin_interval[0]}'

  assert total_percentage == 1.0, (
      'total percentage in score_bins_to_sample_fraction should be equal to'
      f' 1.0, got {total_percentage}'
  )
  return True


flags.register_validator(
    'score_bins_to_sample_fraction',
    validate_score_bins,
    '--score_bins_to_sample_fraction value is not valid.',
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
  score_bins_to_sample_fraction = ast.literal_eval(
      FLAGS.score_bins_to_sample_fraction
  )

  num_images = labeling.create_labeling_images(
      FLAGS.examples_pattern,
      FLAGS.max_images,
      FLAGS.allowed_example_ids_path,
      FLAGS.exclude_import_file_patterns,
      FLAGS.output_dir,
      FLAGS.use_multiprocessing,
      None,
      FLAGS.max_processes,
      FLAGS.buffered_sampling_radius,
      score_bins_to_sample_fraction,
      FLAGS.scores_path,
      FLAGS.filter_by_column,
  )

  if num_images == 0:
    logging.fatal('No labeling images found.')
    return

  logging.info('Wrote %d labeling images.', num_images)


if __name__ == '__main__':
  app.run(main)
