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

"""A script that extracts GeoTIFF urls from Maxar Open Data pages."""

import json
import os
import shutil

from absl import app
from absl import flags
from bs4 import BeautifulSoup
import pandas as pd
import requests
import tensorflow as tf
import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('event_url', '', 'URL of Maxar Open Data event page.')
flags.DEFINE_string('output_dir', '', 'Output directory.')
flags.DEFINE_bool('post_only', False, 'Only download post-event images.')
flags.DEFINE_bool(
    'print_urls', False, 'Print image URLs instead of downloading.'
)
flags.DEFINE_bool('overwrite', False, 'Overwrite existing files.')


def parse_table(url: str) -> pd.DataFrame:
  """Extracts image urls and info from Maxar Open Data page.

  Args:
    url: Open Data page url.

  Returns:
    DataFrame with all image information.
  """
  page_html = requests.get(url).text
  bs = BeautifulSoup(page_html, features='html.parser')
  rows = json.loads(bs.find_all('maxar-table')[0][':rows'])
  df_rows = []
  for row in rows:
    row_bs = BeautifulSoup(row['download'], features='html.parser')
    url = row_bs.find_all('a')[0].get('href')
    df_rows.append({
        'date': row['date'],
        'catalog_id': row['cat_id'],
        'quad_key': row['quad_key'],
        'pre_post': row['filterKey'],
        'url': url
    })
  return pd.DataFrame(df_rows)


def download_image(url: str, output_path: str) -> None:
  if tf.io.gfile.exists(output_path) and not FLAGS.overwrite:
    return
  with requests.get(url, stream=True) as r:
    with tf.io.gfile.GFile(output_path, 'wb') as f:
      shutil.copyfileobj(r.raw, f)


def download_images(url_table: pd.DataFrame, output_dir: str) -> None:
  """Downloads images.

  Args:
    url_table: DataFrame containing download URLs and image metadata.
    output_dir: Output directory.
  """
  for _, row in tqdm.tqdm(url_table.iterrows(), total=len(url_table)):
    catalog_id = row['catalog_id']
    quad_key = row['quad_key']
    date = row['date']
    dir_name = os.path.join(output_dir, row['pre_post'], f'{catalog_id}-{date}')
    if not tf.io.gfile.exists(dir_name):
      tf.io.gfile.makedirs(dir_name)
    file_name = f'{catalog_id}-{quad_key}.tif'
    download_image(row['url'], os.path.join(dir_name, file_name))


def main(_) -> None:
  table = parse_table(FLAGS.event_url)
  if FLAGS.post_only:
    table = table[table['pre_post'] == 'post-event']
  print(f'Found {len(table)} images.')
  if FLAGS.print_urls:
    for _, row in table.iterrows():
      print(row['catalog_id'], row['date'], row['pre_post'], row['url'])
  else:
    download_images(table, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
