# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extracts non-image properties of examples to a CSV file.

Useful for performing analysis on the examples.
"""

import collections

from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf
import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string('examples_pattern', None, 'Examples pattern', required=True)
flags.DEFINE_string('output_path', None, 'Output path.', required=True)


def _single_parse_function(example):
  features = {
      'example_id': tf.io.FixedLenFeature(1, tf.string),
      'plus_code': tf.io.FixedLenFeature(1, tf.string),
      'pre_image_id': tf.io.FixedLenFeature(1, tf.string),
      'post_image_id': tf.io.FixedLenFeature(1, tf.string),
      'encoded_coordinates': tf.io.FixedLenFeature(1, tf.string),
      'string_label': tf.io.FixedLenFeature(1, tf.string),
      'coordinates': tf.io.FixedLenFeature((2,), tf.float32),
  }
  return tf.io.parse_single_example(example, features=features)


def read_tfrecords(pattern: str) -> pd.DataFrame:
  """Reads TFRecords and returns Pandas DataFrame with metadata.

  Args:
    pattern: File pattern for TFRecords.

  Returns:
    DataFrame with example metadata.
  """
  paths = tf.io.gfile.glob(pattern)
  ds = tf.data.Dataset.from_tensor_slices(paths)
  ds = ds.interleave(
      tf.data.TFRecordDataset,
      cycle_length=len(paths),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  ds = ds.map(_single_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  feature_values = collections.defaultdict(list)
  for x in tqdm.tqdm(ds.as_numpy_iterator(), smoothing=0):
    for feature, values_array in x.items():
      if feature == 'coordinates':
        feature_values['longitude'].append(values_array[0])
        feature_values['latitude'].append(values_array[1])
      else:
        feature_values[feature].append(values_array[0].decode())
  return pd.DataFrame(feature_values)


def main(_) -> None:
  df = read_tfrecords(FLAGS.examples_pattern)
  with tf.io.gfile.GFile(FLAGS.output_path, 'w') as f:
    df.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)
