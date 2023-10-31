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

from collections.abc import Sequence
import multiprocessing

from absl import app
from absl import flags
import pandas as pd
from skai import utils
import tensorflow as tf
import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string('examples_pattern', None, 'Examples pattern', required=True)
flags.DEFINE_string('output_path', None, 'Output path.', required=True)
flags.DEFINE_bool('parallel', False, 'Read TFRecords in parallel.')


def read_single_tfrecord(path: str) -> pd.DataFrame:
  """Reads example properties from a single TFRecord.

  Args:
    path: TFRecord path.

  Returns:
    DataFrame with example properties.
  """
  example_properties = []
  for record in tf.data.TFRecordDataset([path]).as_numpy_iterator():
    example = tf.train.Example()
    example.ParseFromString(record)
    longitude, latitude = utils.get_float_feature(example, 'coordinates')
    properties = {
        'longitude': longitude,
        'latitude': latitude,
    }
    for string_prop in [
        'example_id',
        'plus_code',
        'pre_image_id',
        'post_image_id',
        'encoded_coordinates',
        'string_label',
    ]:
      properties[string_prop] = utils.get_string_feature(example, string_prop)
    example_properties.append(properties)
  return pd.DataFrame(example_properties)


def read_tfrecords(paths: Sequence[str]) -> pd.DataFrame:
  if FLAGS.parallel:
    num_workers = min(multiprocessing.cpu_count(), len(paths))
    with multiprocessing.Pool(num_workers) as executor:
      results = tqdm.tqdm(
          executor.imap(read_single_tfrecord, paths), total=len(paths)
      )
  else:
    results = [read_single_tfrecord(p) for p in tqdm.tqdm(paths)]
  return pd.concat(results)


def main(_) -> None:
  df = read_tfrecords(tf.io.gfile.glob(FLAGS.examples_pattern))
  with tf.io.gfile.GFile(FLAGS.output_path, 'w') as f:
    df.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)
