"""A script that extracts GeoTIFF urls from Maxar Open Data pages."""

import json
import shutil
from typing import Sequence
import urllib.parse

from absl import app
from absl import flags
from bs4 import BeautifulSoup
import requests
import tensorflow as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('event_url', '', 'URL of Maxar Open Data event page.')
flags.DEFINE_string('output_dir', '', 'Output directory.')
flags.DEFINE_bool(
    'print_urls', False, 'Print image URLs instead of downloading.'
)


def get_download_urls(url: str):
  page_html = requests.get(url).text
  bs = BeautifulSoup(page_html, features='html.parser')
  rows = json.loads(bs.find_all('maxar-table')[0][':rows'])
  for row in rows:
    row_bs = BeautifulSoup(row['download'], features='html.parser')
    yield row_bs.find_all('a')[0].get('href')


def download_image(url: str, output_path: str) -> None:
  with requests.get(url, stream=True) as r:
    with tf.io.gfile.GFile(output_path, 'wb') as f:
      shutil.copyfileobj(r.raw, f)


def make_download_path(url: str, output_dir: str) -> str:
  parsed_url = urllib.parse.urlparse(url)
  components = parsed_url.path.split('/')
  image_name = components[-1].split('.')[0]
  return f'{output_dir}/{image_name}-{components[-3]}.tif'


def download_images(urls: Sequence[str], output_dir: str) -> None:
  for url in tqdm(urls, total=len(urls)):
    download_image(url, make_download_path(url, output_dir))


def main(_) -> None:
  urls = get_download_urls(FLAGS.event_url)
  if FLAGS.print_urls:
    for url in urls:
      print(url)
  else:
    download_images(urls, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
