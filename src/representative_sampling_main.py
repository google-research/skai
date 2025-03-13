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

r"""Samples examples to be sent to labeling from a CSV file of zero-shot scores.

This script samples a specified number of examples to be sent to labelers from
a CSV containing zero-shot predictions based on pre-specified criteria. It will
sample to achieve geographic diversity, based on coordinates, and example
diversity, based on the zero-shot model's output score. It first samples for the
train set and then the test, based on the provided ratio. Specifically for the
train set, it also supplements likely positives by ranking and then taking a
set number of top-scoring examples.

Example invocation:

python representative_sampling_main.py \
  --predictions_file=gs://bucket/disaster/zero_shot_model/output.csv \
  --output_dir=gs://bucket/disaster/examples/labeling-images \
  --num_examples_to_sample_total=500 \
  --num_examples_to_take_from_top=100 \
  --train_ratio=0.7
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import pandas as pd
from skai import representative_sampling
import tensorflow as tf


FLAGS = flags.FLAGS
_PREDICTIONS_FILE = flags.DEFINE_string(
    'predictions_file',
    None,
    'CSV file containing zero-shot output scores.',
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Directory to write output CSV to.', required=True
)
_NUM_EXAMPLES_TO_SAMPLE_TOTAL = flags.DEFINE_integer(
    'num_examples_to_sample_total', 500, 'Number of examples to sample.'
)
_NUM_EXAMPLES_TO_TAKE_FROM_TOP = flags.DEFINE_integer(
    'num_examples_to_take_from_top',
    100,
    'Out of the total number of number of examples to sample, how many of them '
    'should be sampled from the top-scoring examples after ranking.'
)
_TRAIN_RATIO = flags.DEFINE_float(
    'train_ratio', 0.7, 'Ratio of examples to sample for the train set.'
)
_BUFFER_METERS = flags.DEFINE_integer(
    'buffer_meters',
    80,
    'Buffer distance between two examples to consider them overlapping.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  with tf.io.gfile.GFile(_PREDICTIONS_FILE.value, 'r') as f:
    scores_df = pd.read_csv(f)
  logging.info('Number of Buildings Total: %s', len(scores_df))
  train_df, test_df = representative_sampling.run_representative_sampling(
      scores_df,
      _NUM_EXAMPLES_TO_SAMPLE_TOTAL.value,
      _NUM_EXAMPLES_TO_TAKE_FROM_TOP.value,
      _TRAIN_RATIO.value,
      _BUFFER_METERS.value,
  )
  # Combine the train and test sets and save to one output CSV.
  sampled_df = pd.concat([train_df, test_df])
  sampled_df.to_csv(
      os.path.join(
          _OUTPUT_DIR.value,
          f'{_NUM_EXAMPLES_TO_SAMPLE_TOTAL.value}_sampled_examples.csv',
      ),
      index=False,
  )


if __name__ == '__main__':
  app.run(main)
