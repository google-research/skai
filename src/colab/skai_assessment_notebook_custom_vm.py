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
# @title Configure Assessment Parameters

# pylint:disable=missing-module-docstring
# pylint:disable=g-bad-import-order
# pylint:disable=g-wrong-blank-lines
# pylint:disable=g-import-not-at-top

# @markdown You must re-run this cell every time you make a change.
import os
import textwrap
import ee

GCP_PROJECT = ''  # @param {type:"string"}
GCP_LOCATION = ''  # @param {type:"string"}
GCP_BUCKET = ''  # @param {type:"string"}
GCP_SERVICE_ACCOUNT = ''  # @param {type:"string"}
SERVICE_ACCOUNT_KEY = ''  # @param {type:"string"}
# @markdown This is only needed if BUILDINGS_METHOD is set to "run_model":
BUILDING_SEGMENTATION_MODEL_PATH = ''  # @param {type:"string"}

# @markdown ---
ASSESSMENT_NAME = ''  # @param {type:"string"}
EVENT_DATE = ''  # @param {type:"date"}
OUTPUT_DIR = ''  # @param {type:"string"}

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
SKAI_REPO = 'https://github.com/google-research/skai.git'
OPEN_BUILDINGS_FEATURE_COLLECTION = 'GOOGLE/Research/open-buildings/v3/polygons'
OSM_OVERPASS_URL = 'https://lz4.overpass-api.de/api/interpreter'

# Derived variables
SKAI_CODE_DIR = '/content/skai_src'
AOI_PATH = os.path.join(OUTPUT_DIR, 'aoi.geojson')
BUILDINGS_FILE_LOG = os.path.join(OUTPUT_DIR, 'buildings_file_log.txt')
EXAMPLE_GENERATION_CONFIG_PATH = os.path.join(
    OUTPUT_DIR, 'example_generation_config.json'
)
UNLABELED_TFRECORD_PATTERN = os.path.join(
    OUTPUT_DIR, 'examples', 'unlabeled-large', 'unlabeled-*-of-*.tfrecord'
)
ZERO_SHOT_DIR = os.path.join(OUTPUT_DIR, 'zero_shot_model')
ZERO_SHOT_SCORES = os.path.join(ZERO_SHOT_DIR, 'dataset_0_output.csv')
LABELING_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'labeling_images')
LABELING_EXAMPLES_TFRECORD_PATTERN = os.path.join(
    LABELING_IMAGES_DIR, '*', 'labeling_examples.tfrecord'
)
LABELS_CSV = os.path.join(OUTPUT_DIR, 'labels.csv')
TRAIN_TFRECORD = os.path.join(OUTPUT_DIR, 'labeled_examples_train.tfrecord')
TEST_TFRECORD = os.path.join(OUTPUT_DIR, 'labeled_examples_test.tfrecord')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
INFERENCE_CSV = os.path.join(OUTPUT_DIR, 'inference_scores.csv')


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

if os.path.exists(SERVICE_ACCOUNT_KEY):
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_KEY
else:
  print(f'Service account key not found: "{SERVICE_ACCOUNT_KEY}"')

# %% [markdown]
# #Initialization

# %% cellView="form"
# @title Imports and Function Defs
# %load_ext tensorboard

import collections
import io
import json
import math
import shutil
import subprocess
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
import shapely.wkt
from skai import earth_engine as skai_ee
from skai import labeling
from skai import open_street_map
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


def find_model_dirs(model_root: str):
  # Find all checkpoints dirs first. We only want model dirs that have at least
  # one checkpoint.
  checkpoint_dirs = tf.io.gfile.glob(
      os.path.join(model_root, '*/*/model/epoch-*-aucpr-*'))
  model_dirs = set(os.path.dirname(os.path.dirname(p)) for p in checkpoint_dirs)
  return model_dirs


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


# %% cellView="form"
# @title Authenticate with Earth Engine

def auth():
  credentials = ee.ServiceAccountCredentials(
      GCP_SERVICE_ACCOUNT, SERVICE_ACCOUNT_KEY)
  ee.Initialize(credentials)

auth()


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
      'Unlabeled examples generated:',
      yes_no_text(_file_exists(UNLABELED_TFRECORD_PATTERN)),
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
  print('Label CSV uploaded:', yes_no_text(_file_exists(LABELS_CSV)))
  print(
      'Labeled examples generated:',
      yes_no_text(_file_exists(TRAIN_TFRECORD) and _file_exists(TEST_TFRECORD)),
  )
  trained_models = find_model_dirs(MODEL_DIR)
  print('Fine-tuned model trained:', yes_no_text(bool(trained_models)))
  for model_dir in trained_models:
    print(f'  {model_dir}')

  print(
      'Fine-tuned inference generated:',
      yes_no_text(_file_exists(INFERENCE_CSV)),
  )

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
      df['geometry'] = df['wkt'].apply(shapely.wkt.loads)
      gdf = gpd.GeoDataFrame(df.drop(columns=['wkt']), crs='EPSG:4326')
    elif path.endswith('.parquet'):
      gdf = gpd.read_parquet(f)
    else:
      gdf = gpd.read_file(f)
  print(f'Found {len(gdf)} buildings.')
  print(f'Saved buildings to {path}')
  if len(gdf) < 500000:
    _display_building_footprints(gdf)
  else:
    print('Too many buildings to visualize. Use QGIS instead.')

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
      'resolution': 0.5,
      'use_dataflow': True,
      'cloud_project': GCP_PROJECT,
      'cloud_region': GCP_LOCATION,
      'worker_service_account': GCP_SERVICE_ACCOUNT,
      'max_dataflow_workers': 100,
      'output_shards': 100,
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

  script = textwrap.dedent(f'''
    cd {SKAI_CODE_DIR}/src
    export GOOGLE_APPLICATION_CREDENTIALS={SERVICE_ACCOUNT_KEY}
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
    export GOOGLE_APPLICATION_CREDENTIALS={SERVICE_ACCOUNT_KEY}
    cd {SKAI_CODE_DIR}/src

    xmanager launch skai/model/xm_vlm_zero_shot_vertex.py -- \
      --example_patterns={UNLABELED_TFRECORD_PATTERN} \
      --output_dir={ZERO_SHOT_DIR}
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
    examples_pattern: str,
    scores_file: str,
    output_dir: str,
    max_images: int,
):
  """Creates labeling images."""
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

  timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
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
    ZERO_SHOT_SCORES,
    LABELING_IMAGES_DIR,
    MAX_LABELING_IMAGES,
)


# %% [markdown]
# When the labeling project is complete, download the CSV from the labeling tool
# and upload it to your assessment directory using the following cell.
#
# You may upload multiple CSV files at once, in case you wish to combine labels
# from multiple rounds of labeling.

# %% cellView="form"
# @title Upload Label CSV
def upload_label_csvs(output_path: str):
  """Lets the user upload the labeling CSV file from their computer."""
  uploaded = files.upload()
  dfs = []
  for filename in uploaded.keys():
    f = io.BytesIO(uploaded[filename])
    df = pd.read_csv(f)
    if 'example_id' not in df.columns:
      print('"example_id" column not found in {filename}')
      return
    if 'string_label' not in df.columns:
      print('"string_label" column not found in {filename}')
      return
    dfs.append(df)
    print(f'Read {len(df)} rows from {filename}')

  combined = pd.concat(dfs, ignore_index=True)

  with tf.io.gfile.GFile(output_path, 'wb') as f:
    f.closed = False
    combined.to_csv(f, index=False)

upload_label_csvs(LABELS_CSV)

# %% cellView="form"
# @title Create Labeled Examples
TEST_FRACTION = 0.2  # @param {"type":"number"}
MINOR_IS_0 = False  # @param {"type":"boolean"}


def create_labeled_examples(
    examples_pattern: str,
    labels_csv: str,
    test_fraction: float,
    train_path: str,
    test_path: str,
    minor_is_0: bool):
  """Creates labeled train and test TFRecords files."""

  minor_damage_float_label = (0 if minor_is_0 else 1)
  label_mapping = [
      'bad_example=0',
      'no_damage=0',
      f'minor_damage={minor_damage_float_label}',
      'major_damage=1',
      'destroyed=1',
  ]

  labeling.create_labeled_examples(
      label_file_paths=[labels_csv],
      string_to_numeric_labels=label_mapping,
      example_patterns=[examples_pattern],
      test_fraction=test_fraction,
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
    LABELS_CSV,
    TEST_FRACTION,
    TRAIN_TFRECORD,
    TEST_TFRECORD,
    MINOR_IS_0)


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


show_label_stats(TRAIN_TFRECORD, TEST_TFRECORD)

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
    export GOOGLE_APPLICATION_CREDENTIALS={SERVICE_ACCOUNT_KEY}
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

run_training(
    ASSESSMENT_NAME,
    TRAIN_TFRECORD,
    TEST_TFRECORD,
    MODEL_DIR,
    NUM_EPOCHS)


# %% cellView="form"
# @title View Tensorboard
def start_tensorboard(model_root: str):
  """Shows Tensorboard visualization."""
  tensorboard_dirs = tf.io.gfile.glob(
      os.path.join(model_root, '*/*/tensorboard')
  )
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

  def run_tensorboard(_):
    # pylint:disable=unused-variable
    tensorboard_dir = dir_selection_widget.value
    # %tensorboard --load_fast=false --logdir $tensorboard_dir
    # pylint:enable=unused-variable

  start_button = widgets.Button(
      description='Start',
  )
  start_button.on_click(run_tensorboard)

  display(dir_selection_widget)
  display(start_button)

start_tensorboard(MODEL_DIR)


# %% cellView="form"
# @title Run inference
def get_best_checkpoint(model_dir: str):
  checkpoint_dirs = tf.io.gfile.glob(os.path.join(model_dir, 'epoch-*-aucpr-*'))
  best_checkpoint = None
  best_aucpr = 0
  for checkpoint in checkpoint_dirs:
    aucpr = float(checkpoint.split('-')[-1])
    if aucpr > best_aucpr:
      best_checkpoint = checkpoint
      best_aucpr = aucpr
  return best_checkpoint


def run_inference(
    examples_pattern: str,
    model_dir: str,
    output_dir: str,
    output_path: str,
    cloud_project: str,
    cloud_region: str,
    service_account: str) -> None:
  """Starts model inference job."""
  temp_dir = os.path.join(output_dir, 'inference_temp')
  print(
      f'Running inference with model checkpoint "{model_dir}" on examples'
      f' matching "{examples_pattern}"'
  )
  print(f'Output will be written to {output_path}')

  # accelerator_flags = ' '.join([
  #     '--worker_machine_type=n1-highmem-8',
  #     '--accelerator=nvidia-tesla-t4',
  #     '--accelerator_count=1'])

  # Currently, Colab only supports Python 3.10. However, the docker images we
  # need for GPU acceleration are based on Tensorflow 2.14.0 images, which are
  # based on Python 3.11. If we try to launch an inference job with GPU
  # acceleration, Dataflow will complain about a Python version mismatch.
  # Therefore, we can only use CPU inference until Colab upgrades to Python 3.11
  # (which should be sometime within 2024).
  accelerator_flags = ''

  script = textwrap.dedent(f'''
    cd {SKAI_CODE_DIR}/src
    export GOOGLE_CLOUD_PROJECT={cloud_project}
    export GOOGLE_APPLICATION_CREDENTIALS={SERVICE_ACCOUNT_KEY}
    python skai/model/inference.py \
      --examples_pattern='{examples_pattern}' \
      --image_model_dir='{model_dir}' \
      --output_path='{output_path}' \
      --use_dataflow \
      --cloud_project='{cloud_project}' \
      --cloud_region='{cloud_region}' \
      --dataflow_temp_dir='{temp_dir}' \
      --worker_service_account='{service_account}' \
      --threshold=0.5 \
      --high_precision_threshold=0.75 \
      --high_recall_threshold=0.4 \
      --max_dataflow_workers=4 {accelerator_flags}
  ''')

  script_path = '/content/inference_script.sh'
  with open(script_path, 'w') as f:
    f.write(script)
  # !bash {script_path}


def do_inference(model_root: str):
  """Runs model inference."""
  model_dirs = find_model_dirs(model_root)
  if not model_dirs:
    print(
        f'No models found in directory {model_root}. Please train a model'
        ' first.'
    )
    return

  model_selection_widget = widgets.Dropdown(
      options=model_dirs,
      description='Choose a model:',
      layout={'width': 'initial'},
  )
  model_selection_widget.style.description_width = 'initial'

  def start_clicked(_):
    model_dir = os.path.join(model_selection_widget.value, 'model')
    checkpoint = get_best_checkpoint(model_dir)
    if not checkpoint:
      print('Model directory does not contain a valid checkpoint directory.')
      return
    run_inference(
        UNLABELED_TFRECORD_PATTERN,
        checkpoint,
        OUTPUT_DIR,
        INFERENCE_CSV,
        GCP_PROJECT,
        GCP_LOCATION,
        GCP_SERVICE_ACCOUNT,
    )

  start_button = widgets.Button(
      description='Start',
  )
  start_button.on_click(start_clicked)

  display(model_selection_widget)
  display(start_button)

do_inference(MODEL_DIR)

# %% cellView="form"
# @title Get assessment stats
DAMAGE_SCORE_THRESHOLD = 0.5  # @param {type:"number"}

make_download_button(
    INFERENCE_CSV,
    f'{ASSESSMENT_NAME}_assessment.csv',
    'Download CSV')
show_inference_stats(AOI_PATH, INFERENCE_CSV, DAMAGE_SCORE_THRESHOLD)
show_assessment_heatmap(AOI_PATH, INFERENCE_CSV, DAMAGE_SCORE_THRESHOLD, False)
