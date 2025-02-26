# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all,cellView
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% cellView="form"
# @title Install Libraries
# @markdown This will take approximately 1 minute to run. After completing, you
# @markdown may be prompted to restart the kernel. Select "Restart" and then
# @markdown proceed to run the next cell.
"""Notebook for running SKAI assessments."""

# pylint: disable=g-statement-before-imports
SKAI_REPO = 'https://github.com/google-research/skai.git'
SKAI_CODE_DIR = '/content/skai_src'


def install_requirements():
  """Installs necessary Python libraries."""
  # !rm -rf {SKAI_CODE_DIR}
  # !git clone {SKAI_REPO} {SKAI_CODE_DIR}
  # !pip install {SKAI_CODE_DIR}/src/.

  requirements = [
      'apache_beam[gcp]==2.54.0',
      'fiona',
      # https://github.com/apache/beam/issues/32169
      'google-cloud-storage>=2.18.2',
      'ml-collections',
      'openlocationcode',
      'rasterio',
      'rio-cogeo',
      'rtree',
      'tensorflow==2.14.0',
      'tensorflow_addons',
      'tensorflow_text',
      'xmanager',
  ]

  requirements_file = '/content/requirements.txt'
  with open(requirements_file, 'w') as f:
    f.write('\n'.join(requirements))
  # !pip install -r {requirements_file}

install_requirements()

# %% cellView="form"
# @title Configure Assessment Parameters

# pylint:disable=missing-module-docstring
# pylint:disable=g-bad-import-order
# pylint:disable=g-wrong-blank-lines
# pylint:disable=g-import-not-at-top

# @markdown You must re-run this cell every time you make a change.
import os
import ee
from google.colab import auth

GCP_PROJECT = ''  # @param {type:"string"}
GCP_LOCATION = ''  # @param {type:"string"}
GCP_BUCKET = ''  # @param {type:"string"}
GCP_SERVICE_ACCOUNT = ''  # @param {type:"string"}
# @markdown This is only needed if BUILDINGS_METHOD is set to "run_model":
BUILDING_SEGMENTATION_MODEL_PATH = ''  # @param {type:"string"}

# @markdown ---
ASSESSMENT_NAME = ''  # @param {type:"string"}
EVENT_DATE = ''  # @param {type:"date"}
OUTPUT_DIR = ''  # @param {type:"string"}
EXAMPLE_RESOLUTION = 0.5  # @param {type:"number"}

# @markdown ---
BEFORE_IMAGE_0 = ''  # @param {type:"string"}
BEFORE_IMAGE_1 = ''  # @param {type:"string"}
BEFORE_IMAGE_2 = ''  # @param {type:"string"}
BEFORE_IMAGE_3 = ''  # @param {type:"string"}
BEFORE_IMAGE_4 = ''  # @param {type:"string"}
BEFORE_IMAGE_5 = ''  # @param {type:"string"}
BEFORE_IMAGE_6 = ''  # @param {type:"string"}
BEFORE_IMAGE_7 = ''  # @param {type:"string"}
BEFORE_IMAGE_8 = ''  # @param {type:"string"}
BEFORE_IMAGE_9 = ''  # @param {type:"string"}
# @markdown ---
AFTER_IMAGE_0 = ''  # @param {type:"string"}
AFTER_IMAGE_1 = ''  # @param {type:"string"}
AFTER_IMAGE_2 = ''  # @param {type:"string"}
AFTER_IMAGE_3 = ''  # @param {type:"string"}
AFTER_IMAGE_4 = ''  # @param {type:"string"}
AFTER_IMAGE_5 = ''  # @param {type:"string"}
AFTER_IMAGE_6 = ''  # @param {type:"string"}
AFTER_IMAGE_7 = ''  # @param {type:"string"}
AFTER_IMAGE_8 = ''  # @param {type:"string"}
AFTER_IMAGE_9 = ''  # @param {type:"string"}

# Constants
SKAI_CODE_DIR = '/content/skai_src'
OPEN_BUILDINGS_FEATURE_COLLECTION = 'GOOGLE/Research/open-buildings/v3/polygons'
OSM_OVERPASS_URL = 'https://lz4.overpass-api.de/api/interpreter'
TRAIN_TFRECORD_NAME = 'labeled_examples_train.tfrecord'
TEST_TFRECORD_NAME = 'labeled_examples_test.tfrecord'
HIGH_RECALL = 0.7
HIGH_PRECISION = 0.7
INFERENCE_BATCH_SIZE = 8
INFERENCE_SUBDIR = 'inference'

# Derived variables
AOI_PATH = os.path.join(OUTPUT_DIR, 'aoi.geojson')
BUILDINGS_FILE_LOG = os.path.join(OUTPUT_DIR, 'buildings_file_log.txt')
EXAMPLE_GENERATION_CONFIG_PATH = os.path.join(
    OUTPUT_DIR, 'example_generation_config.json'
)
UNLABELED_TFRECORD_PATTERN = os.path.join(
    OUTPUT_DIR, 'examples', 'unlabeled-large', 'unlabeled-*-of-*.tfrecord'
)
UNLABELED_PARQUET_PATTERN = os.path.join(
    OUTPUT_DIR, 'examples', 'unlabeled-parquet', 'examples-*-of-*.parquet'
)
ZERO_SHOT_DIR = os.path.join(OUTPUT_DIR, 'zero_shot_model')
ZERO_SHOT_SCORES = os.path.join(ZERO_SHOT_DIR, 'dataset_0_output.csv')
LABELING_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'labeling_images')
LABELING_EXAMPLES_TFRECORD_PATTERN = os.path.join(
    LABELING_IMAGES_DIR, '*', 'labeling_examples.tfrecord'
)
LABELED_EXAMPLES_ROOT = os.path.join(OUTPUT_DIR, 'labeled_examples')


def process_image_entries(entries: list[str]) -> list[str]:
  image_ids = []
  for entry in entries:
    entry = entry.strip()
    if entry:
      image_ids.append(entry)
  return image_ids


BEFORE_IMAGES = process_image_entries([
    BEFORE_IMAGE_0,
    BEFORE_IMAGE_1,
    BEFORE_IMAGE_2,
    BEFORE_IMAGE_3,
    BEFORE_IMAGE_4,
    BEFORE_IMAGE_5,
    BEFORE_IMAGE_6,
    BEFORE_IMAGE_7,
    BEFORE_IMAGE_8,
    BEFORE_IMAGE_9,
])

AFTER_IMAGES = process_image_entries([
    AFTER_IMAGE_0,
    AFTER_IMAGE_1,
    AFTER_IMAGE_2,
    AFTER_IMAGE_3,
    AFTER_IMAGE_4,
    AFTER_IMAGE_5,
    AFTER_IMAGE_6,
    AFTER_IMAGE_7,
    AFTER_IMAGE_8,
    AFTER_IMAGE_9,
])


# %% [markdown]
# #Initialization

# %% cellView="form"
# @title Authenticate with Google Cloud
def authenticate():
  auth.authenticate_user()
  ee.Authenticate()
  ee.Initialize(project=GCP_PROJECT)

authenticate()

# %% cellView="form"
# @title Imports and Function Defs
# %load_ext tensorboard

import collections
import io
import json
import math
import shutil
import subprocess
import textwrap
import time
import warnings

import folium
import folium.plugins
import geopandas as gpd
from google.colab import data_table
from google.colab import files
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely.wkt
from skai import earth_engine as skai_ee
from skai import labeling
from skai import open_street_map
from skai.model import inference_lib
import sklearn.metrics
import tensorflow as tf
import tqdm.notebook

data_table.enable_dataframe_formatter()


def convert_wgs_to_utm(lon: float, lat: float):
  """Based on lat and lng, return best utm epsg-code."""
  utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
  if len(utm_band) == 1:
    utm_band = '0' + utm_band
  if lat >= 0:
    epsg_code = '326' + utm_band
  else:
    epsg_code = '327' + utm_band
  return f'EPSG:{epsg_code}'


def get_aoi_area_km2(aoi_path: str):
  with tf.io.gfile.GFile(aoi_path) as f:
    aoi = gpd.read_file(f)

  centroid = aoi.geometry.unary_union.centroid
  utm_crs = convert_wgs_to_utm(centroid.x, centroid.y)
  utm_aoi = aoi.to_crs(utm_crs)
  area_meters_squared = utm_aoi.geometry.unary_union.area
  area_km_squared = area_meters_squared / 1000000
  return area_km_squared


def show_inference_stats(
    aoi_path: str,
    inference_csv_path: str,
    threshold: float):
  """Prints out statistics on inference result."""
  with tf.io.gfile.GFile(inference_csv_path) as f:
    df = pd.read_csv(f)
  building_count = len(df)
  if 'damage_score' in df.columns:
    scores = df['damage_score']
  elif 'score' in df.columns:
    scores = df['score']
  else:
    raise ValueError(f'{inference_csv_path} does not contain a score column.')

  damaged = df.loc[scores > threshold]
  damaged_count = len(damaged)
  damaged_pct = 100 * damaged_count / building_count
  print('Area KM^2:', get_aoi_area_km2(aoi_path))
  print('Buildings assessed:', building_count)
  print('Damaged buildings:', damaged_count)
  print(f'Percentage damaged: {damaged_pct:0.3g}%')


def _open_file(path: str, mode: str):
  f = tf.io.gfile.GFile(path, mode)
  f.closed = False
  return f


def _file_exists(path: str) -> bool:
  return bool(tf.io.gfile.glob(path))


def _read_text_file(path: str) -> str:
  with tf.io.gfile.GFile(path, 'r') as f:
    return f.read()


def _make_map(longitude: float, latitude: float, zoom: float):
  """Creates a Folium map with common base layers.

  Args:
    longitude: Longitude of initial view.
    latitude: Latitude of initial view.
    zoom: Zoom level of initial view.

  Returns:
    Folium map.
  """
  base_maps = [
      folium.TileLayer(
          tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
          attr='Google',
          name='Google Maps',
          overlay=False,
          control=True,
      ),
  ]

  m = folium.Map(
      location=(latitude, longitude),
      max_zoom=24,
      zoom_start=zoom,
      tiles=None)
  for base_map in base_maps:
    base_map.add_to(m)
  return m


def show_assessment_heatmap(
    aoi_path: str,
    scores_path: str,
    threshold: float,
    is_zero_shot: bool):
  """Creates a Folium heatmap from inference scores."""
  with _open_file(scores_path, 'rb') as f:
    df = pd.read_csv(f)
  if is_zero_shot:
    damaged = df.loc[~df['is_cloudy'] & (df['damage_score'] >= threshold)]
  else:
    damaged = df.loc[df['score'] >= threshold]
  points = zip(damaged['latitude'].values, damaged['longitude'].values)
  centroid_x = np.mean(damaged['longitude'].values)
  centroid_y = np.mean(damaged['latitude'].values)
  folium_map = _make_map(centroid_x, centroid_y, 12)
  with _open_file(aoi_path, 'rb') as f:
    aoi_gdf = gpd.read_file(f)
  folium.GeoJson(
      aoi_gdf.to_json(),
      name='AOI',
      style_function=lambda _: {'fillOpacity': 0},
  ).add_to(folium_map)
  heatmap = folium.plugins.HeatMap(points)
  heatmap.add_to(folium_map)
  display(folium_map)


def make_download_button(path: str, file_name: str, caption: str):
  """Displays a button for downloading a file in the colab kernel."""
  def download(_):
    temp_path = f'/tmp/{file_name}'
    with _open_file(path, 'rb') as src:
      with open(temp_path, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    files.download(temp_path)

  button = widgets.Button(
      description=caption,
  )
  button.on_click(download)
  display(button)


def find_labeled_examples_dirs():
  """Returns directories containing labeled TFRecords."""
  dirs = tf.io.gfile.glob(os.path.join(LABELED_EXAMPLES_ROOT, '*'))
  valid_dirs = []
  for d in dirs:
    train_path = os.path.join(d, TRAIN_TFRECORD_NAME)
    test_path = os.path.join(d, TEST_TFRECORD_NAME)
    valid = True
    if not tf.io.gfile.exists(train_path):
      print(
          f'Warning: Train TFRecord does not exist in {d}, so not considering'
          ' this a valid labeled dataset.'
      )
      valid = False
    if not tf.io.gfile.exists(test_path):
      print(
          f'Warning: Test TFRecord does not exist in {d}, so not considering'
          ' this a valid labeled dataset.'
      )
      valid = False
    if not valid:
      continue
    valid_dirs.append(d)
  return sorted(valid_dirs, reverse=True)


def find_model_dirs():
  # Find all checkpoints dirs first. We only want model dirs that have at least
  # one checkpoint.
  checkpoint_dirs = tf.io.gfile.glob(
      os.path.join(LABELED_EXAMPLES_ROOT, '*/models/*/*/model/epoch-*-aucpr-*'))
  model_dirs = set(os.path.dirname(os.path.dirname(p)) for p in checkpoint_dirs)
  return sorted(model_dirs, reverse=True)


def get_best_checkpoint(model_dir: str):
  """Finds the checkpoint subdirectory with the highest AUPRC.

  Args:
    model_dir: Model directory.

  Returns:
    Checkpoint directory path.
  """
  checkpoint_dirs = tf.io.gfile.glob(os.path.join(model_dir, 'epoch-*-aucpr-*'))
  best_checkpoint = None
  best_aucpr = 0
  for checkpoint in checkpoint_dirs:
    aucpr = float(checkpoint.split('-')[-1])
    if aucpr > best_aucpr:
      best_checkpoint = checkpoint
      best_aucpr = aucpr
  return best_checkpoint


def find_inference_csvs():
  inference_csvs = []
  for model_dir in find_model_dirs():
    inference_csvs.extend(
        tf.io.gfile.glob(os.path.join(model_dir, INFERENCE_SUBDIR, '*.csv'))
    )
  return inference_csvs


def find_labeling_image_metadata_files(labeling_images_dir: str):
  return tf.io.gfile.glob(os.path.join(
      labeling_images_dir, '*', 'image_metadata.csv'))


def yes_no_text(value: bool) -> str:
  return '\x1b[32mYES\x1b[0m' if value else '\x1b[31mNO\x1b[0m'


def visualize_images(images: list[tuple[np.ndarray, np.ndarray]]):
  """Displays before and after images side-by-side."""
  num_rows = len(images)
  size_factor = 3
  fig_size = (2 * size_factor, num_rows * size_factor)
  fig, axes = plt.subplots(num_rows, 2, figsize=fig_size)
  for row, (pre_image, post_image) in enumerate(images):
    ax1 = axes[row, 0]
    ax2 = axes[row, 1]
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(pre_image)
    ax2.imshow(post_image)
  plt.show(fig)


def get_eeda_bearer_token(service_account: str) -> str:
  return subprocess.check_output(
      'gcloud auth print-access-token'
      f' --impersonate-service-account="{service_account}"',
      shell=True,
  ).decode()


def get_timestamp() -> str:
  return time.strftime('%Y%m%d_%H%M%S', time.localtime())


# %% [markdown]
# # Check Assessment Status
#
# Run the following cell to check which steps of the assessment have already
# been completed.

# %% cellView="form"
# @title Check assessment status
def check_assessment_status():
  """Shows which steps of the assessment have been completed."""
  print('AOI uploaded:', yes_no_text(_file_exists(AOI_PATH)))

  if _file_exists(BUILDINGS_FILE_LOG):
    buildings_file = _read_text_file(BUILDINGS_FILE_LOG).strip()
    print('Building footprints generated:', yes_no_text(True))
    print(f'  Building footprints file: {buildings_file}')
  else:
    print('Building footprints generated:', yes_no_text(False))

  print(
      'Example generation config file exists:',
      yes_no_text(_file_exists(EXAMPLE_GENERATION_CONFIG_PATH)),
  )
  print(
      'Unlabeled tfrecord files generated:',
      yes_no_text(_file_exists(UNLABELED_TFRECORD_PATTERN)),
  )
  print(
      'Unlabeled parquet files generated:',
      yes_no_text(_file_exists(UNLABELED_PARQUET_PATTERN)),
  )
  print(
      'Zero-shot assessment generated:',
      yes_no_text(_file_exists(ZERO_SHOT_SCORES)),
  )
  labeling_metadata_files = find_labeling_image_metadata_files(
      LABELING_IMAGES_DIR
  )
  print(
      'Labeling images generated:', yes_no_text(bool(labeling_metadata_files))
  )
  for p in labeling_metadata_files:
    print(f'  {p}')

  labeled_examples_dirs = find_labeled_examples_dirs()
  print(
      'Labeled examples generated:', yes_no_text(bool(labeled_examples_dirs)))
  if labeled_examples_dirs:
    print('\n'.join([f'  {d}' for d in labeled_examples_dirs]))
  trained_model_dirs = find_model_dirs()
  print('Fine-tuned model trained:', yes_no_text(bool(trained_model_dirs)))
  if trained_model_dirs:
    print('\n'.join([f'  {d}' for d in trained_model_dirs]))

  inference_csvs = find_inference_csvs()
  print('Inference CSVs generated:', yes_no_text(bool(inference_csvs)))
  if inference_csvs:
    print('\n'.join([f'  {f}' for f in inference_csvs]))


check_assessment_status()


# %% [markdown]
# # Example Generation

# %% cellView="form"
# @title Upload AOI file
def upload_aoi():
  """Shows button for user to upload AOI to the assessment directory."""
  if _file_exists(AOI_PATH):
    print(f'AOI file {AOI_PATH} already exists.')
    answer = input('Do you want to overwrite (y/n)? ')
    if answer.lower() not in ['y', 'yes']:
      print('AOI file not uploaded.')
      return

  uploaded = files.upload()

  file_names = list(uploaded.keys())
  if len(file_names) != 1:
    print('You must choose exactly one GeoJSON file to upload.')
    print('Upload NOT successful.')
    return

  if not file_names[0].endswith('.geojson'):
    print('AOI file must be in GeoJSON format and have extension ".geojson".')
    print('Upload NOT successful.')
    return

  with _open_file(AOI_PATH, 'wb') as f:
    f.write(uploaded[file_names[0]])

upload_aoi()

# %% cellView="form"
# @title Get building footprints

# pylint:disable=line-too-long
BUILDINGS_METHOD = 'open_buildings'  # @param ["open_buildings","open_street_map","run_model","file"]
# pylint:enable=line-too-long
# @markdown This is only needed if BUILDINGS_METHOD is set to "file":
USER_BUILDINGS_FILE = ''  # @param {type:"string"}


def download_open_buildings(aoi_path: str, output_dir: str) -> str:
  path = os.path.join(output_dir, 'open_buildings.parquet')
  with _open_file(aoi_path, 'r') as f:
    gdf = gpd.read_file(f)
  aoi = gdf.unary_union
  skai_ee.get_open_buildings(
      [aoi], OPEN_BUILDINGS_FEATURE_COLLECTION, 0.5, False, path)
  return path


def download_open_street_map(aoi_path: str, output_dir: str) -> str:
  path = os.path.join(output_dir, 'open_street_map_buildings.parquet')
  with _open_file(aoi_path, 'r') as f:
    gdf = gpd.read_file(f)
  aoi = gdf.unary_union
  open_street_map.get_building_centroids_in_regions(
      [aoi], OSM_OVERPASS_URL, path
  )
  return path


def run_building_detection_model(
    aoi_path: str,
    output_dir: str):
  """Runs building detection model."""
  image_paths = ','.join(BEFORE_IMAGES)
  child_dir = os.path.join(output_dir, 'buildings')
  if any('EEDAI:' in image for image in BEFORE_IMAGES):
    token = get_eeda_bearer_token(GCP_SERVICE_ACCOUNT)
    eeda_bearer_env = f'export EEDA_BEARER="{token}"'
  else:
    eeda_bearer_env = ''

  script = textwrap.dedent(f'''
    export PYTHONPATH={SKAI_CODE_DIR}/src:$PYTHONPATH
    export GOOGLE_CLOUD_PROJECT={GCP_PROJECT}
    {eeda_bearer_env}
    cd {SKAI_CODE_DIR}/src
    python detect_buildings_main.py \
      --cloud_project='{GCP_PROJECT}' \
      --cloud_region='{GCP_LOCATION}' \
      --worker_service_account='{GCP_SERVICE_ACCOUNT}' \
      --use_dataflow \
      --output_dir='{output_dir}' \
      --image_paths='{image_paths}' \
      --aoi_path='{aoi_path}' \
      --model_path='{BUILDING_SEGMENTATION_MODEL_PATH}'
  ''')
  script_path = '/content/run_building_detection.sh'
  with open(script_path, 'w') as f:
    f.write(script)
  # !bash {script_path}

  buildings_file = os.path.join(child_dir, 'dedup_buildings.parquet')
  return buildings_file


def _display_building_footprints(buildings_gdf: gpd.GeoDataFrame):
  """Visualizes building footprints in a folium map."""
  centroid = buildings_gdf.centroid.unary_union.centroid

  folium_map = _make_map(centroid.x, centroid.y, 13)
  if len(buildings_gdf) > 100000:
    print('Too many building footprints to display. Displaying random sample.')
    buildings_gdf = buildings_gdf.sample_points(100000)
  folium.GeoJson(
      buildings_gdf.to_json(),
      name='buildings',
      marker=folium.CircleMarker(
          radius=3, weight=0, fill_color='#FF0000', fill_opacity=1
      ),
  ).add_to(folium_map)
  display(folium_map)


def download_buildings(aoi_path: str, output_dir: str) -> None:
  """Downloads buildings to assessment directory."""
  if BUILDINGS_METHOD == 'open_buildings':
    path = download_open_buildings(aoi_path, output_dir)
  elif BUILDINGS_METHOD == 'open_street_map':
    path = download_open_street_map(aoi_path, output_dir)
  elif BUILDINGS_METHOD == 'run_model':
    path = run_building_detection_model(aoi_path, output_dir)
  elif BUILDINGS_METHOD == 'file':
    path = USER_BUILDINGS_FILE
  else:
    raise ValueError(f'Unknown BUILDINGS_METHOD {BUILDINGS_METHOD}')

  with _open_file(BUILDINGS_FILE_LOG, 'w') as f:
    f.write(f'{path}\n')

  with _open_file(path, 'rb') as f:
    if path.endswith('.csv'):
      df = pd.read_csv(f)
      if 'wkt' in df.columns:
        df['geometry'] = df['wkt'].apply(shapely.wkt.loads)
        gdf = gpd.GeoDataFrame(df.drop(columns=['wkt']), crs='EPSG:4326')
      elif 'longitude' in df.columns and 'latitude' in df.columns:
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(
            df.drop(columns=['longitude', 'latitude']), crs='EPSG:4326'
        )
    elif path.endswith('.parquet'):
      gdf = gpd.read_parquet(f)
    else:
      gdf = gpd.read_file(f)
  print(f'Found {len(gdf)} buildings.')
  print(f'Saved buildings to {path}')
  _display_building_footprints(gdf)


with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  download_buildings(AOI_PATH, OUTPUT_DIR)


# %% cellView="form"
# @title Write Example Generation Config File
def write_example_generation_config(path: str) -> None:
  """Writes example generation config file to assessment directory."""
  dataset_name = ASSESSMENT_NAME.lower().replace('_', '-')
  with _open_file(BUILDINGS_FILE_LOG, 'r') as f:
    buildings_file = f.read().strip()

  config_dict = {
      'dataset_name': dataset_name,
      'aoi_path': AOI_PATH,
      'output_dir': OUTPUT_DIR,
      'buildings_method': 'file',
      'buildings_file': buildings_file,
      'resolution': EXAMPLE_RESOLUTION,
      'use_dataflow': True,
      'cloud_project': GCP_PROJECT,
      'cloud_region': GCP_LOCATION,
      'worker_service_account': GCP_SERVICE_ACCOUNT,
      'max_dataflow_workers': 100,
      'output_shards': 100,
      'output_parquet': True,
      'output_metadata_file': True,
      'before_image_patterns': BEFORE_IMAGES,
      'after_image_patterns': AFTER_IMAGES,
  }

  valid_config = True
  for key, value in config_dict.items():
    if not value:
      if key == 'buildings_file' and config_dict['buildings_method'] != 'file':
        continue
      print(f'Field {key} cannot be empty')
      valid_config = False
  if not valid_config:
    return

  config_string = json.dumps(config_dict, indent=2)
  print(f'Example Generation configuration written to {path}:')
  print()
  print(config_string)
  with tf.io.gfile.GFile(path, 'w') as f:
    f.write(config_string)

write_example_generation_config(EXAMPLE_GENERATION_CONFIG_PATH)


# %% cellView="form"
# @title Run Example Generation Job
def run_example_generation(config_file_path: str):
  """Runs example generation pipeline."""
  if any('EEDAI:' in image for image in BEFORE_IMAGES):
    token = get_eeda_bearer_token(GCP_SERVICE_ACCOUNT)
    eeda_bearer_env = f'export EEDA_BEARER="{token}"'
  else:
    eeda_bearer_env = ''

  script = textwrap.dedent(f'''
    cd {SKAI_CODE_DIR}/src
    {eeda_bearer_env}
    python generate_examples_main.py \
      --configuration_path={config_file_path} \
      --output_metadata_file
  ''')

  script_path = '/content/example_generation.sh'
  with open(script_path, 'w') as f:
    f.write(script)
  # !bash {script_path}

run_example_generation(EXAMPLE_GENERATION_CONFIG_PATH)


# %% cellView="form"
# @title Visualize Generated Examples
def visualize_generated_examples(pattern: str, num: int):
  images = []
  paths = tf.io.gfile.glob(pattern)
  for record in tf.data.TFRecordDataset([paths[0]]).take(num):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    pre_image = plt.imread(io.BytesIO(
        example.features.feature['pre_image_png_large'].bytes_list.value[0]))
    post_image = plt.imread(io.BytesIO(
        example.features.feature['post_image_png_large'].bytes_list.value[0]))
    images.append((pre_image, post_image))
  visualize_images(images)

visualize_generated_examples(UNLABELED_TFRECORD_PATTERN, 3)


# %% cellView="form"
# @title Run Zero Shot Model
def run_zero_shot_model():
  """Runs zero-shot model inference."""
  script = textwrap.dedent(f'''
    export PYTHONPATH={SKAI_CODE_DIR}/src:$PYTHONPATH
    export GOOGLE_CLOUD_PROJECT={GCP_PROJECT}
    export GOOGLE_CLOUD_BUCKET_NAME={GCP_BUCKET}
    cd {SKAI_CODE_DIR}/src

    xmanager launch skai/model/xm_vlm_zero_shot_vertex.py -- \
      --example_patterns={UNLABELED_TFRECORD_PATTERN} \
      --output_dir={ZERO_SHOT_DIR} \
      --cloud_location={GCP_LOCATION}
    ''')

  print(
      'Starting zero shot model inference. Scores will be written to'
      f' {ZERO_SHOT_SCORES}'
  )
  script_path = '/content/zero_shot_model.sh'
  with open(script_path, 'w') as f:
    f.write(script)
  # !bash {script_path}

run_zero_shot_model()

# %% cellView="form"
# @title View Zero Shot Assessment
DAMAGE_SCORE_THRESHOLD = 0.5  # @param {type:"number"}

make_download_button(
    ZERO_SHOT_SCORES,
    f'{ASSESSMENT_NAME}_zero_shot_assessment.csv',
    'Download CSV')
show_inference_stats(AOI_PATH, ZERO_SHOT_SCORES, DAMAGE_SCORE_THRESHOLD)
show_assessment_heatmap(
    AOI_PATH, ZERO_SHOT_SCORES, DAMAGE_SCORE_THRESHOLD, True
)

# %% [markdown]
# # Labeling

# %% cellView="form"
# @title Create Labeling Images
MAX_LABELING_IMAGES = 1000  # @param {"type":"integer"}


def visualize_labeling_images(images_dir: str, num: int):
  """Displays a small sample of labeling images."""
  pre_image_paths = sorted(
      tf.io.gfile.glob(os.path.join(images_dir, '*_pre.png'))
  )
  post_image_paths = sorted(
      tf.io.gfile.glob(os.path.join(images_dir, '*_post.png'))
  )
  assert len(pre_image_paths) == len(post_image_paths), (
      f'Number of pre images ({len(pre_image_paths)}) does not match number of'
      f' post images ({len(post_image_paths)}).'
  )
  images = []
  for pre_image_path, post_image_path in list(
      zip(pre_image_paths, post_image_paths)
  )[:num]:
    with _open_file(pre_image_path, 'rb') as f:
      pre_image = plt.imread(f)
    with _open_file(post_image_path, 'rb') as f:
      post_image = plt.imread(f)
    images.append((pre_image, post_image))
  visualize_images(images)


def create_labeling_images(
    tfrecord_pattern: str,
    parquet_pattern: str,
    scores_file: str,
    output_dir: str,
    max_images: int,
):
  """Creates labeling images."""

  # Prefer using Parquet dataset over TFRecords.
  if tf.io.gfile.glob(parquet_pattern):
    examples_pattern = parquet_pattern
  else:
    examples_pattern = tfrecord_pattern

  if not tf.io.gfile.glob(examples_pattern):
    print(
        f'No files match "{examples_pattern}". Please run example generation'
        ' first.'
    )
    return

  existing_metadata_files = find_labeling_image_metadata_files(output_dir)
  if existing_metadata_files:
    print(
        'The following labeling image metadata files have already been'
        ' generated:'
    )
    print('\n'.join(f'  {p}' for p in existing_metadata_files))
    response = input(
        'Do you want to generate a new set of labeling images (y/n)? '
    )
    if response.lower() not in ['y', 'yes']:
      return

  timestamp = get_timestamp()
  images_dir = os.path.join(output_dir, timestamp)
  metadata_csv = os.path.join(images_dir, 'image_metadata.csv')

  num_images = labeling.create_labeling_images(
      examples_pattern,
      max_images,
      set(),
      set(),
      images_dir,
      True,
      None,
      4,
      70.0,
      {
          (0, 0.25): 0.25,
          (0.25, 0.5): 0.25,
          (0.5, 0.75): 0.25,
          (0.75, 1.0): 0.25,
      },
      scores_path=scores_file,
      filter_by_column='is_cloudy',
  )
  print('Number of labeling images:', num_images)
  print(
      'Please create a new project in the SKAI labeling tool with the following'
      ' metadata CSV:'
  )
  print(metadata_csv)
  visualize_labeling_images(images_dir, 3)

create_labeling_images(
    UNLABELED_TFRECORD_PATTERN,
    UNLABELED_PARQUET_PATTERN,
    ZERO_SHOT_SCORES,
    LABELING_IMAGES_DIR,
    MAX_LABELING_IMAGES,
)

# %% [markdown]
# When the labeling project is complete, download the CSV from the labeling tool
# and upload them here to create labeled examples.
#
# You may upload multiple CSV files at once, in case you wish to combine labels
# from multiple rounds of labeling.

# %% cellView="form"
# @title Create Labeled Examples
TEST_PERCENTAGE = 20  # @param {"type":"integer"}
MINOR_IS_0 = True  # @param {"type":"boolean"}


def upload_label_csvs(output_path: str) -> bool:
  """Lets the user upload the labeling CSV file from their computer."""

  print('Choose labels CSV to create labeled examples from.')
  uploaded = files.upload()
  if not uploaded:
    print('Upload cancelled')
    return False

  dfs = []
  for filename in uploaded.keys():
    f = io.BytesIO(uploaded[filename])
    df = pd.read_csv(f)
    if 'example_id' not in df.columns:
      print('"example_id" column not found in {filename}')
      return False
    if 'string_label' not in df.columns:
      print('"string_label" column not found in {filename}')
      return False
    dfs.append(df)
    print(f'Read {len(df)} rows from {filename}')

  combined = pd.concat(dfs, ignore_index=True)

  with tf.io.gfile.GFile(output_path, 'wb') as f:
    f.closed = False
    combined.to_csv(f, index=False)
  print(f'Wrote {len(combined)} labels to {output_path}')
  return True


def create_labeled_examples(
    examples_pattern: str,
    test_percent: int,
    minor_is_0: bool,
    labeled_examples_dir: str):
  """Creates labeled train and test TFRecords files."""

  assert test_percent < 100, 'Test percentage must be less than 100%.'
  assert test_percent >= 1, 'Test percentage must be at least 1%.'
  train_percent = 100 - test_percent
  timestamp = get_timestamp()
  output_dir = os.path.join(
      labeled_examples_dir,
      f'{timestamp}_{train_percent:02d}_{test_percent:02d}_minor{0 if minor_is_0 else 1}',
  )
  labels_path = os.path.join(output_dir, 'labels.csv')
  if not upload_label_csvs(labels_path):
    return

  train_path = os.path.join(output_dir, TRAIN_TFRECORD_NAME)
  test_path = os.path.join(output_dir, TEST_TFRECORD_NAME)
  minor_damage_float_label = (0 if minor_is_0 else 1)
  label_mapping = [
      'bad_example=0',
      'no_damage=0',
      f'minor_damage={minor_damage_float_label}',
      'major_damage=1',
      'destroyed=1',
  ]

  print('Creating labeled examples. This may take a while.')
  labeling.create_labeled_examples(
      label_file_paths=[labels_path],
      string_to_numeric_labels=label_mapping,
      example_patterns=[examples_pattern],
      test_fraction=test_percent / 100,
      train_output_path=train_path,
      test_output_path=test_path,
      connecting_distance_meters=70.0,
      use_multiprocessing=False,
      multiprocessing_context=None,
      max_processes=1,
  )
  print(f'Train TFRecord: {train_path}')
  print(f'Test TFRecord: {test_path}')

create_labeled_examples(
    LABELING_EXAMPLES_TFRECORD_PATTERN,
    TEST_PERCENTAGE,
    MINOR_IS_0,
    LABELED_EXAMPLES_ROOT)


# %% cellView="form"
# @title Show Label Stats
def _load_examples_into_df(
    train_tfrecords: str,
    test_tfrecords: str,
) -> pd.DataFrame:
  """Loads examples from TFRecords into a DataFrame.
  """
  feature_config = {
      'example_id': tf.io.FixedLenFeature([], tf.string),
      'coordinates': tf.io.FixedLenFeature([2], tf.float32),
      'string_label': tf.io.FixedLenFeature([], tf.string, 'unlabeled'),
      'label': tf.io.FixedLenFeature([], tf.float32),
  }

  def _parse_examples(record_bytes):
    return tf.io.parse_single_example(record_bytes, feature_config)

  columns = collections.defaultdict(list)
  longitudes = []
  latitudes = []
  for path in [train_tfrecords, test_tfrecords]:
    for features in tqdm.notebook.tqdm(
        tf.data.TFRecordDataset([path])
        .map(_parse_examples, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator(),
        desc=path,
    ):
      longitudes.append(features['coordinates'][0])
      latitudes.append(features['coordinates'][1])
      columns['example_id'].append(features['example_id'].decode())
      columns['string_label'].append(features['string_label'].decode())
      columns['label'].append(features['label'])
      columns['source_path'].append(path)

  return pd.DataFrame(columns)


def _format_counts_table(df: pd.DataFrame):
  for column in df.columns:
    if column != 'All':
      df[column] = [
          f'{x}  ({x/t * 100:0.2f}%)' for x, t in zip(df[column], df['All'])
      ]


def show_label_stats(train_tfrecord: str, test_tfrecord: str):
  """Displays tables showing label count stats."""
  df = _load_examples_into_df(train_tfrecord, test_tfrecord)
  counts = df.pivot_table(
      index='source_path',
      columns='string_label',
      aggfunc='count',
      values='example_id',
      margins=True,
      fill_value=0)
  _format_counts_table(counts)

  print('String Label Counts')
  display(data_table.DataTable(counts))

  float_counts = df.pivot_table(
      index='source_path',
      columns='label',
      aggfunc='count',
      values='example_id',
      margins=True,
      fill_value=0.0)
  _format_counts_table(float_counts)
  print('Float Label Counts')
  display(data_table.DataTable(float_counts))


def choose_dataset_show_label_stats():
  """Allows user to choose a labeled dataset and shows stats about it."""
  labeled_example_dirs = find_labeled_examples_dirs()
  dir_select = widgets.Dropdown(
      options=labeled_example_dirs,
      description='Choose a labeled examples dir:',
      layout={'width': 'initial'},
  )
  dir_select.style.description_width = 'initial'

  show_stats_button = widgets.Button(description='Show Stats')
  def show_stats(_):
    show_stats_button.disabled = True
    train_path = os.path.join(dir_select.value, TRAIN_TFRECORD_NAME)
    test_path = os.path.join(dir_select.value, TEST_TFRECORD_NAME)
    show_label_stats(train_path, test_path)

  show_stats_button.on_click(show_stats)
  display(dir_select)
  display(show_stats_button)


choose_dataset_show_label_stats()

# %% [markdown]
# # Fine Tuning

# %% cellView="form"
# @title Train model

NUM_EPOCHS = 20  # @param {type:"integer"}


def run_training(
    experiment_name: str,
    train_path: str,
    test_path: str,
    output_dir: str,
    num_epochs: int):
  """Runs training job."""
  if not tf.io.gfile.exists(train_path):
    raise ValueError(
        f'Train TFRecord {train_path} does not exist. Did you run the "Create'
        ' Labeled Examples" cell?'
    )
  if not tf.io.gfile.exists(test_path):
    raise ValueError(
        f'Test TFRecord {test_path} does not exist. Did you run the "Create'
        ' Labeled Examples" cell?'
    )

  print(f'Train data: {train_path}')
  print(f'Test data: {test_path}')
  print(f'Model dir: {output_dir}')
  job_args = {
      'config': 'src/skai/model/configs/skai_two_tower_config.py',
      'config.data.tfds_dataset_name': 'skai_dataset',
      'config.data.adhoc_config_name': 'adhoc_dataset',
      'config.data.labeled_train_pattern': train_path,
      'config.data.validation_pattern': test_path,
      'config.output_dir': output_dir,
      'config.training.num_epochs': num_epochs,
      'accelerator': 'V100',
      'experiment_name': experiment_name,
  }
  job_arg_str = ' '.join(f'--{f}={v}' for f, v in job_args.items())
  sh = textwrap.dedent(f'''
    export GOOGLE_CLOUD_PROJECT={GCP_PROJECT}
    export GOOGLE_CLOUD_BUCKET_NAME={GCP_BUCKET}
    export PYTHONPATH={SKAI_CODE_DIR}/src
    export LOCATION={GCP_LOCATION}

    cd {SKAI_CODE_DIR}

    xmanager launch src/skai/model/xm_launch_single_model_vertex.py -- \
    --xm_wrap_late_bindings \
    --xm_upgrade_db=True \
    --cloud_location=$LOCATION \
    --accelerator_count=1 {job_arg_str}''')

  with open('script.sh', 'w') as file:
    file.write(sh)

  # !bash script.sh


def choose_dataset_run_training():
  """Allows user to choose a labeled dataset and trains a model using it."""
  labeled_example_dirs = find_labeled_examples_dirs()
  dir_select = widgets.Dropdown(
      options=labeled_example_dirs,
      description='Choose a labeled examples dir:',
      layout={'width': 'initial'},
  )
  dir_select.style.description_width = 'initial'

  start_button = widgets.Button(description='Start Training')
  def start_training(_):
    start_button.disabled = True
    train_path = os.path.join(dir_select.value, TRAIN_TFRECORD_NAME)
    test_path = os.path.join(dir_select.value, TEST_TFRECORD_NAME)
    model_dir = os.path.join(dir_select.value, 'models')
    run_training(ASSESSMENT_NAME, train_path, test_path, model_dir, NUM_EPOCHS)

  start_button.on_click(start_training)
  display(dir_select)
  display(start_button)

choose_dataset_run_training()


# %% cellView="form"
# @title View Tensorboard
def start_tensorboard():
  """Shows Tensorboard visualization."""
  model_dirs = find_model_dirs()
  tensorboard_dirs = [
      tb
      for d in model_dirs
      if tf.io.gfile.isdir(tb := os.path.join(d, 'tensorboard'))
  ]
  if not tensorboard_dirs:
    print(
        'No Tensorboard directories found. Either you have not trained a model'
        ' yet or a running job has not written any tensorboard log events yet.'
    )
    return

  dir_selection_widget = widgets.Dropdown(
      options=tensorboard_dirs,
      description='Choose a tensorboard dir:',
      layout={'width': 'initial'},
  )
  dir_selection_widget.style.description_width = 'initial'

  start_button = widgets.Button(description='Start')
  def run_tensorboard(_):
    # pylint:disable=unused-variable
    start_button.disabled = True
    tensorboard_dir = dir_selection_widget.value
    # %tensorboard --load_fast=false --logdir $tensorboard_dir
    # pylint:enable=unused-variable

  start_button.on_click(run_tensorboard)

  display(dir_selection_widget)
  display(start_button)

start_tensorboard()

# %% cellView="form"
# @title Evaluate Fine-Tuned Model

def plot_precision_recall(labels: np.ndarray, scores: np.ndarray) -> None:
  """Plots distinct precision and recall curves in a single graph.

  The X-axis of the graph is the threshold value. This graph shows the
  trade-off between precision and recall for a specific threshold value more
  clearly than the usual PR curve.

  Args:
    labels: True labels array.
    scores: Model scores array.
  """
  sklearn.metrics.PrecisionRecallDisplay.from_predictions(labels, scores)
  plt.title('Precision and Recall vs. Threshold')
  plt.grid()
  plt.show()

  precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
      labels, scores)
  x = pd.DataFrame({
      'threshold': thresholds,
      'precision': precision[:-1],
      'recall': recall[:-1],
  })
  sns.lineplot(data=x.set_index('threshold'))
  plt.title('Precision/Recall vs. Threshold')
  plt.grid()
  plt.show()


def get_recall_at_precision(
    thresholds: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    min_precision: float) -> tuple[float, float, float]:
  """Finds threshold that maximizes recall with a minimum precision value.

  Args:
    thresholds: List of threshold values returned by
      sklearn.metrics.precision_recall_curve. Length N.
    precisions: List of precision values returned by
      sklearn.metrics.precision_recall_curve. Length N + 1.
    recalls: List of recall values returned by
      sklearn.metrics.precision_recall_curve. Length N + 1.
    min_precision: Minimum precision value to maintain.

  Returns:
    Tuple of (threshold, precision, recall).
  """
  precisions = precisions[:-1]
  recalls = recalls[:-1]
  eligible = (precisions > min_precision)
  if not any(eligible):
    # If precision never exceeds the minimum value desired, return the threshold
    # where it is highest.
    eligible = (precisions == np.max(precisions))
  i = np.argmax(recalls[eligible])
  return thresholds[eligible][i], precisions[eligible][i], recalls[eligible][i]


def get_precision_at_recall(
    thresholds: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    min_recall: float) -> tuple[float, float, float]:
  """Finds threshold that maximizes precision with a minimum recall value.

  Args:
    thresholds: List of threshold values returned by
      sklearn.metrics.precision_recall_curve. Length N.
    precisions: List of precision values returned by
      sklearn.metrics.precision_recall_curve. Length N + 1.
    recalls: List of recall values returned by
      sklearn.metrics.precision_recall_curve. Length N + 1.
    min_recall: Minimum recall value to maintain.

  Returns:
    Tuple of (threshold, precision, recall).
  """
  precisions = precisions[:-1]
  recalls = recalls[:-1]
  eligible = (recalls > min_recall)
  if not any(eligible):
    # If recall never exceeds the minimum value desired, return the threshold
    # where it is highest.
    eligible = (recalls == np.max(recalls))
  i = np.argmax(precisions[eligible])
  return thresholds[eligible][i], precisions[eligible][i], recalls[eligible][i]


def get_max_f1_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
  """Finds the threshold that maximizes F1 score.

  Args:
    scores: Prediction scores assigned by the model.
    labels: True labels.

  Returns:
    Tuple of best threshold and F1-score, Precision, Recall at that threshold.
  """
  best_f1 = 0
  best_threshold = 0
  best_precision = 0
  best_recall = 0
  for threshold in scores:
    predictions = (scores >= threshold)
    if (f1 := sklearn.metrics.f1_score(labels, predictions)) > best_f1:
      best_f1 = f1
      best_threshold = threshold
      best_precision = sklearn.metrics.precision_score(labels, predictions)
      best_recall = sklearn.metrics.recall_score(labels, predictions)
  return best_threshold, best_f1, best_precision, best_recall


def plot_score_distribution(labels: np.ndarray, scores: np.ndarray) -> None:
  df = {'score': scores, 'label': labels}
  sns.displot(data=df, x='score', col='label')
  plt.show()


def print_model_metrics(scores: np.ndarray, labels: np.ndarray) -> None:
  """Prints evaluation metrics."""
  precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(
      labels, scores
  )
  auprc = sklearn.metrics.auc(recalls, precisions)
  auroc = sklearn.metrics.roc_auc_score(labels, scores)
  print(f'AUPRC:     {auprc:.4g}')
  print(f'AUROC:     {auroc:.4g}')

  threshold, f1, precision, recall = get_max_f1_threshold(scores, labels)
  print('\nFor maximum F1-score')
  print(f'  Threshold: {threshold}')
  print(f'  F1-score: {f1}')
  print(f'  Precision: {precision}')
  print(f'  Recall: {recall}')

  threshold, precision, recall = get_precision_at_recall(
      thresholds, precisions, recalls, HIGH_RECALL
  )
  print(f'\nFor recall >= {HIGH_RECALL}')
  print(f'  Threshold: {threshold}')
  print(f'  Precision: {precision}')
  print(f'  Recall: {recall}')

  threshold, precision, recall = get_recall_at_precision(
      thresholds, precisions, recalls, HIGH_PRECISION
  )
  print(f'\nFor precision >= {HIGH_PRECISION}')
  print(f'  Threshold: {threshold}')
  print(f'  Precision: {precision}')
  print(f'  Recall: {recall}')

  plot_precision_recall(labels, scores)
  plot_score_distribution(labels, scores)


def _read_examples(path: str) -> list[tf.train.Example]:
  examples = []
  for record in tf.data.TFRecordDataset([path]):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    examples.append(example)
  return examples


def _get_label(example: tf.train.Example) -> float:
  return example.features.feature['label'].float_list.value[0]


def _evaluate_model(model_dir: str, examples_path: str) -> None:
  """Evaluates model on examples and prints metrics."""

  print('Reading examples ...')
  examples = _read_examples(examples_path)
  print('Done reading examples')
  if not examples:
    raise ValueError('No examples')

  print('Loading model ...')
  model = inference_lib.TF2InferenceModel(
      model_dir,
      224,
      False,
      inference_lib.ModelType.CLASSIFICATION,
  )
  model.prepare_model()
  print('Done loading model')

  print('Running inference ...')
  scores = []
  labels = []
  for batch_start in tqdm.notebook.tqdm(
      range(0, len(examples), INFERENCE_BATCH_SIZE)
  ):
    batch = examples[batch_start:batch_start+INFERENCE_BATCH_SIZE]
    scores.extend(model.predict_scores(batch).numpy())
    labels.extend(_get_label(e) for e in batch)
  scores = np.array(scores)
  labels = np.array(labels)
  print_model_metrics(scores, labels)


def evaluate_model_on_test_examples():
  """Lets user evaluate a model on chosen trained model and test examples.
  """
  labeled_example_dirs = find_labeled_examples_dirs()
  examples_select = widgets.Dropdown(
      options=labeled_example_dirs,
      description='Choose a labeled examples dir:',
      layout={'width': 'initial'},
  )
  examples_select.style.description_width = 'initial'

  model_dirs = find_model_dirs()
  if not model_dirs:
    print('No trained model directories found. Please train a model first.')
    return

  model_select = widgets.Dropdown(
      options=model_dirs,
      description='Choose a model:',
      layout={'width': 'initial'},
  )
  model_select.style.description_width = 'initial'
  run_button = widgets.Button(description='Run')

  def run_button_clicked(_):
    run_button.disabled = True
    test_path = os.path.join(examples_select.value, TEST_TFRECORD_NAME)
    model_dir = os.path.join(model_select.value, 'model')
    checkpoint = get_best_checkpoint(model_dir)
    if not checkpoint:
      print('Model directory does not contain a valid checkpoint directory.')
      return
    _evaluate_model(checkpoint, test_path)

  run_button.on_click(run_button_clicked)

  display(model_select)
  display(examples_select)
  display(run_button)


evaluate_model_on_test_examples()

# %% cellView="form"
# @title Run inference
# @markdown These should be changed to the thresholds chosen in the eval cell.
DEFAULT_THRESHOLD = 0.5  # @param {"type":"number"}
HIGH_PRECISION_THRESHOLD = 0.6  # @param {"type":"number"}
HIGH_RECALL_THRESHOLD = 0.4  # @param {"type":"number"}
INFERENCE_FILE_NAME = 'inference_test.csv'  # @param {"type":"string"}


def run_inference(
    examples_pattern: str,
    checkpoint_dir: str,
    default_threshold: float,
    high_precision_threshold: float,
    high_recall_threshold: float,
    output_dir: str,
    cloud_project: str,
    cloud_region: str,
    service_account: str) -> None:
  """Starts model inference job."""

  output_path = os.path.join(output_dir, INFERENCE_FILE_NAME)
  if tf.io.gfile.exists(output_path):
    if input(f'File {output_path} exists. Overwrite? (y/n)').lower() not in (
        'y',
        'yes',
    ):
      print('Cancelled')
      return

  temp_dir = os.path.join(output_dir, 'inference_temp')
  print(
      f'Running inference with model checkpoint "{checkpoint_dir}" on examples'
      f' matching "{examples_pattern}"'
  )
  print(f'Output will be written to {output_path}')

  accelerator_flags = ' '.join([
      '--worker_machine_type=n1-highmem-8',
      '--accelerator=nvidia-tesla-t4',
      '--accelerator_count=1'])

  script = textwrap.dedent(f'''
    cd {SKAI_CODE_DIR}/src
    export PYTHONPATH={SKAI_CODE_DIR}/src:$PYTHONPATH
    export GOOGLE_CLOUD_PROJECT={cloud_project}
    python skai/model/inference.py \
      --examples_pattern='{examples_pattern}' \
      --image_model_dir='{checkpoint_dir}' \
      --output_path='{output_path}' \
      --use_dataflow \
      --cloud_project='{cloud_project}' \
      --cloud_region='{cloud_region}' \
      --dataflow_temp_dir='{temp_dir}' \
      --worker_service_account='{service_account}' \
      --threshold={default_threshold} \
      --high_precision_threshold={high_precision_threshold} \
      --high_recall_threshold={high_recall_threshold} \
      --max_dataflow_workers=4 {accelerator_flags}
  ''')

  script_path = '/content/inference_script.sh'
  with open(script_path, 'w') as f:
    f.write(script)
  # !bash {script_path}


def do_inference():
  """Runs model inference."""
  model_dirs = find_model_dirs()
  if not model_dirs:
    print('No trained model directories found. Please train a model first.')
    return

  model_selection_widget = widgets.Dropdown(
      options=model_dirs,
      description='Choose a model:',
      layout={'width': 'initial'},
  )
  model_selection_widget.style.description_width = 'initial'
  start_button = widgets.Button(description='Start')

  def start_clicked(_):
    start_button.disabled = True
    model_dir = os.path.join(model_selection_widget.value, 'model')
    checkpoint_dir = get_best_checkpoint(model_dir)
    output_dir = os.path.join(model_selection_widget.value, INFERENCE_SUBDIR)
    if not checkpoint_dir:
      print('Model directory does not contain a valid checkpoint directory.')
      return
    run_inference(
        UNLABELED_TFRECORD_PATTERN,
        checkpoint_dir,
        DEFAULT_THRESHOLD,
        HIGH_PRECISION_THRESHOLD,
        HIGH_RECALL_THRESHOLD,
        output_dir,
        GCP_PROJECT,
        GCP_LOCATION,
        GCP_SERVICE_ACCOUNT,
    )

  start_button.on_click(start_clicked)

  display(model_selection_widget)
  display(start_button)


do_inference()

# %% cellView="form"
# @title Get assessment stats
DAMAGE_SCORE_THRESHOLD = 0.5  # @param {type:"number"}


def get_assessment_stats():
  """Get assessment statistics."""
  inference_csvs = find_inference_csvs()
  if not inference_csvs:
    print('No inference CSVs found.')
    return

  selection_widget = widgets.Dropdown(
      options=inference_csvs,
      description='Choose an inference CSV:',
      layout={'width': 'initial'},
  )
  selection_widget.style.description_width = 'initial'

  def stats_clicked(_):
    stats_button.disabled = True
    inference_csv = selection_widget.value
    show_inference_stats(AOI_PATH, inference_csv, DAMAGE_SCORE_THRESHOLD)
    show_assessment_heatmap(
        AOI_PATH, inference_csv, DAMAGE_SCORE_THRESHOLD, False
    )

  def download_clicked(_):
    inference_csv = selection_widget.value
    file_name = os.path.basename(inference_csv)
    temp_path = f'/tmp/{file_name}'
    with _open_file(inference_csv, 'rb') as src:
      with open(temp_path, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    files.download(temp_path)

  stats_button = widgets.Button(description='Get Stats')
  stats_button.on_click(stats_clicked)
  download_button = widgets.Button(description='Download')
  download_button.on_click(download_clicked)

  display(selection_widget)
  display(stats_button)
  display(download_button)


get_assessment_stats()
