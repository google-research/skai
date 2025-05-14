# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all,cellView
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% cellView="form"
# @title Install Libraries
# @markdown This will take approximately 3 minutes to run. After completing, you
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
      'fiona==1.10.1',
      # https://github.com/apache/beam/issues/32169
      'google-cloud-storage==2.19.0',
      'ml-collections==1.0.0',
      'numpy==1.24.4',
      'openlocationcode==1.0.1',
      'rasterio==1.3.9',
      'rio-cogeo==5.4.1',
      'rtree==1.3.0',
      'tensorflow==2.14.0',
      'tensorflow_addons==0.23.0',
      'tensorflow_text==2.14.0',
      'xmanager==0.7.0',
  ]

  requirements_file = '/content/requirements.txt'
  with open(requirements_file, 'w') as f:
    f.write('\n'.join(requirements))
  # !pip install -r {requirements_file}
  # !pip uninstall -y jax  # this is causing errors with our version of numpy


install_requirements()

# %% cellView="form"
# @title Configure Assessment Parameters

# pylint:disable=missing-module-docstring
# pylint:disable=g-bad-import-order
# pylint:disable=g-wrong-blank-lines
# pylint:disable=g-import-not-at-top

# @markdown See _documentation link_ for explanation of each parameter.

# @markdown **You must re-run this cell every time you make a change.**
import os
import ee
import tensorflow as tf
from google.colab import auth

GCP_PROJECT = ''  # @param {type:"string"}
GCP_LOCATION = ''  # @param {type:"string"}
GCP_BUCKET = ''  # @param {type:"string"}
GCP_SERVICE_ACCOUNT = ''  # @param {type:"string"}
CLOUD_RUN_PROJECT = ''  # @param {type:"string"}
CLOUD_RUN_LOCATION = ''  # @param {type:"string"}

# @markdown ---
ASSESSMENT_NAME = ''  # @param {type:"string"}
OUTPUT_DIR = ''  # @param {type:"string"}
IMAGE_RESOLUTION = 0.5  # @param {type:"number"}

# @markdown ---
# @markdown **Note**: Order of images should reflect layer order in ArcGIS.
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
# @markdown **Note**: Order of images should reflect layer order in ArcGIS.
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
BUILDINGS_DIR = os.path.join(OUTPUT_DIR, 'buildings')
EXAMPLE_GENERATION_CONFIG_PATH = os.path.join(
    OUTPUT_DIR, 'example_generation_config.json'
)
EXAMPLE_METADATA_PATTERN = os.path.join(
    OUTPUT_DIR, 'examples', 'metadata', 'metadata-*-of-*.parquet'
)
UNLABELED_TFRECORD_PATTERN = os.path.join(
    OUTPUT_DIR, 'examples', 'unlabeled-large', 'unlabeled-*-of-*.tfrecord'
)
UNLABELED_PARQUET_PATTERN = os.path.join(
    OUTPUT_DIR, 'examples', 'unlabeled-parquet', 'examples-*-of-*.parquet'
)
ZERO_SHOT_DIR = os.path.join(OUTPUT_DIR, 'zero_shot_model')
ZERO_SHOT_SCORES = os.path.join(ZERO_SHOT_DIR, 'dataset_0_output.csv')
ZERO_SHOT_DEDUPED_SCORES = os.path.join(ZERO_SHOT_DIR, 'dataset_0_deduped.csv')
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
  """Authenticates user with Cloud and checks read/write access to bucket."""
  auth.authenticate_user()
  ee.Authenticate()
  ee.Initialize(project=GCP_PROJECT)

  # Check to make sure that user has read and write access to the output dir.
  test_file_path = os.path.join(OUTPUT_DIR, 'skai_read_write_test.txt')
  value = 'skai'
  with tf.io.gfile.GFile(test_file_path, 'w') as f:
    f.write(value)
  with tf.io.gfile.GFile(test_file_path, 'r') as f:
    if f.read() != value:
      raise ValueError(f'Failed to read or write to {test_file_path}')
  tf.io.gfile.remove(test_file_path)

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

import geopandas as gpd
from google.colab import data_table
from google.colab import files
import IPython.display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely.wkt
from skai import earth_engine as skai_ee
from skai import labeling
from skai import open_street_map
from skai import representative_sampling
from skai.model import inference_lib
import sklearn.metrics
import tensorboard.notebook
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

  centroid = aoi.union_all().centroid
  utm_crs = convert_wgs_to_utm(centroid.x, centroid.y)
  utm_aoi = aoi.to_crs(utm_crs)
  area_meters_squared = utm_aoi.union_all().area
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
  IPython.display.display(button)


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
  return sorted(valid_dirs)


def find_model_dirs():
  # Find all checkpoints dirs first. We only want model dirs that have at least
  # one checkpoint.
  checkpoint_dirs = tf.io.gfile.glob(
      os.path.join(LABELED_EXAMPLES_ROOT, '*/models/*/*/model/epoch-*-aucpr-*'))
  model_dirs = set(os.path.dirname(os.path.dirname(p)) for p in checkpoint_dirs)
  return sorted(model_dirs)


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
  return sorted(inference_csvs)


def find_labeling_image_metadata_files(labeling_images_dir: str):
  return sorted(tf.io.gfile.glob(os.path.join(
      labeling_images_dir, '*', 'image_metadata.csv')))


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


def choose_option(options: list[str], prompt: str) -> str:
  """Make user choose one option from many.

  Args:
    options: List of options to choose from.
    prompt: Prompt to show to user.

  Returns:
    Chosen option.
  """
  if not options:
    return None
  elif len(options) == 1:
    return options[0]
  else:
    print(f'\n{prompt}:')
    print('\n'.join(f'[{i+1}] {o}' for i, o in enumerate(options)))
    min_choice = 1
    max_choice = len(options)
    while True:
      try:
        choice = int(input(f'Choice ({min_choice}-{max_choice}): '))
        if choice < min_choice or choice > max_choice:
          raise ValueError()
        print()
        return options[choice - 1]
      except ValueError:
        print(f'Invalid choice. Must be between {min_choice}-{max_choice}')


# %% cellView="form"
# @title Check assessment status (Optional)

# @markdown Run this cell to check what steps of the assessment have already
# @markdown been completed.


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
# @title Upload AOI file (Optional)
# @markdown If you want to limit the analysis to a region smaller than the image
# @markdown extents, please upload a polygon GeoJSON file here.


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

  gdf = gpd.read_file(file_names[0]).to_crs(4326)
  gdf.to_file(AOI_PATH, driver='GeoJSON')

upload_aoi()

# %% cellView="form"
# @title Get building footprints

# pylint:disable=line-too-long
BUILDINGS_METHOD = 'open_buildings'  # @param ["open_buildings","open_street_map","run_model","file"]
# @markdown This is only needed if BUILDINGS_METHOD is set to "file":
USER_BUILDINGS_FILE = ''  # @param {type:"string"}
# pylint:enable=line-too-long


def download_open_buildings(aoi_path: str, output_dir: str) -> str:
  path = os.path.join(output_dir, 'open_buildings.parquet')
  with _open_file(aoi_path, 'r') as f:
    gdf = gpd.read_file(f)
  aoi = gdf.union_all()
  skai_ee.get_open_buildings(
      [aoi], OPEN_BUILDINGS_FEATURE_COLLECTION, 0.5, False, path)
  return path


def download_open_street_map(aoi_path: str, output_dir: str) -> str:
  path = os.path.join(output_dir, 'open_street_map_buildings.parquet')
  with _open_file(aoi_path, 'r') as f:
    gdf = gpd.read_file(f)
  aoi = gdf.union_all()
  open_street_map.get_building_centroids_in_regions(
      [aoi], OSM_OVERPASS_URL, path
  )
  return path


def run_detect_buildings_cloud_run_job() -> str:
  """Runs building detection Cloud Run job."""
  images_str = ','.join(BEFORE_IMAGES)
  joined = ':::'.join([
      f'OUTPUT_DIR={BUILDINGS_DIR}',
      f'AOI={AOI_PATH}',
      f'IMAGES={images_str}',
      'CONFIDENCE_THRESHOLD=0.65'
  ])
  env_vars = f'^:::^{joined}'
  gcloud_args = [
      'gcloud',
      'run',
      'jobs',
      'execute',
      'detect-buildings',
      '--wait',
      f'--project={CLOUD_RUN_PROJECT}',
      f'--region={CLOUD_RUN_LOCATION}',
      f'--update-env-vars={env_vars}',
  ]
  cloud_run_dashboard_url = f'https://console.cloud.google.com/run/jobs/details/{CLOUD_RUN_LOCATION}/detect-buildings/executions?project={CLOUD_RUN_PROJECT}'
  print(f'Starting building detection Cloud Run job. {cloud_run_dashboard_url}')
  try:
    output = subprocess.check_output(gcloud_args)
  except subprocess.CalledProcessError as e:
    print(f'Cloud Run failed with status {e.returncode}. Output: "{e.output}"')
    return None
  print(output.decode())
  return os.path.join(BUILDINGS_DIR, 'dedup_buildings.parquet')


def download_buildings(aoi_path: str, output_dir: str) -> None:
  """Downloads buildings to assessment directory."""
  if BUILDINGS_METHOD == 'open_buildings':
    path = download_open_buildings(aoi_path, output_dir)
    print(f'Saved buildings to {path}')
  elif BUILDINGS_METHOD == 'open_street_map':
    path = download_open_street_map(aoi_path, output_dir)
    print(f'Saved buildings to {path}')
  elif BUILDINGS_METHOD == 'run_model':
    path = run_detect_buildings_cloud_run_job()
    print(f'Saved buildings to {path}')
  elif BUILDINGS_METHOD == 'file':
    path = USER_BUILDINGS_FILE
  else:
    raise ValueError(f'Unknown BUILDINGS_METHOD {BUILDINGS_METHOD}')

  with _open_file(BUILDINGS_FILE_LOG, 'w') as f:
    f.write(f'{path}\n')

  if not _file_exists(path):
    print(f'Buildings file {path} does not exist yet. Wait for the building '
          'detection job to finish before proceeding.')
    return

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
      else:
        raise ValueError(
            f'Buildings file {path} must have either a "wkt" column or'
            ' "longitude" and "latitude" columns.'
        )
    elif path.endswith('.parquet'):
      gdf = gpd.read_parquet(f)
    else:
      gdf = gpd.read_file(f)
  print(f'Found {len(gdf)} buildings.')


with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  download_buildings(AOI_PATH, BUILDINGS_DIR)

# %% cellView="form"
# @title Run Example Generation Job


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
      'resolution': IMAGE_RESOLUTION,
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


write_example_generation_config(EXAMPLE_GENERATION_CONFIG_PATH)
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

visualize_generated_examples(UNLABELED_TFRECORD_PATTERN, 10)


# %% cellView="form"
# @title Run Zero Shot Model
def run_zero_shot_model():
  """Runs zero-shot model inference."""
  image = 'gcr.io/disaster-assessment/skai-ml-siglip-tpu:20250423-155604_575647'
  script = textwrap.dedent(f'''
    export GOOGLE_CLOUD_PROJECT={GCP_PROJECT}
    export GOOGLE_CLOUD_BUCKET_NAME={GCP_BUCKET}
    cd {SKAI_CODE_DIR}/src

    xmanager launch skai/model/xm_vlm_zero_shot_vertex.py -- \
      --example_patterns={UNLABELED_TFRECORD_PATTERN} \
      --output_dir={ZERO_SHOT_DIR} \
      --cloud_location={GCP_LOCATION} \
      --siglip_docker_image='{image}' \
      --cloud_bucket_name={GCP_BUCKET} \
      --cloud_project={GCP_PROJECT}
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

def visualize_high_zero_shot_score_examples(
    tfrecord_pattern: str,
    zero_shot_scores_path: str,
    score_threshold: float,
    num_examples: int):
  """Visualizes examples with zero-shot scores above a threshold."""
  with _open_file(zero_shot_scores_path, 'r') as f:
    scores = pd.read_csv(f)
  scores = scores.loc[scores['is_cloudy'] == 0]
  scores.set_index('example_id', inplace=True)
  scores.drop_duplicates(inplace=True)
  chosen = []
  for record in (
      tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_pattern))
      .prefetch(5)
      .as_numpy_iterator()
  ):
    example = tf.train.Example()
    example.ParseFromString(record)
    example_id = (
        example.features.feature['example_id'].bytes_list.value[0].decode()
    )
    try:
      row = scores.loc[example_id]
    except KeyError:
      continue
    if row['damage_score'] > score_threshold:
      pre_image = tf.image.decode_png(
          example.features.feature['pre_image_png_large'].bytes_list.value[0])
      post_image = tf.image.decode_png(
          example.features.feature['post_image_png_large'].bytes_list.value[0])
      chosen.append((pre_image, post_image))
      if len(chosen) > num_examples:
        break

  print(f'Examples of buildings with a zero-shot score >= {score_threshold}')
  visualize_images(chosen)


def show_high_damage_example_counts(zero_shot_scores_path: str):
  with _open_file(zero_shot_scores_path, 'r') as f:
    scores = pd.read_csv(f)
  for score in (0.7, 0.8, 0.9, 0.95):
    num = sum(scores['damage_score'] > score)
    fraction = 100 * num / len(scores)
    print(f'Number of examples with score > {score}: {num} ({fraction:.1f}%)')


make_download_button(
    ZERO_SHOT_DEDUPED_SCORES,
    f'{ASSESSMENT_NAME}_zero_shot_assessment.csv',
    'Download CSV')

show_high_damage_example_counts(ZERO_SHOT_DEDUPED_SCORES)

# visualize_high_zero_shot_score_examples(
#     UNLABELED_TFRECORD_PATTERN,
#     ZERO_SHOT_SCORES,
#     DAMAGE_SCORE_THRESHOLD,
#     5,
# )


# %% [markdown]
# # Labeling

# %% cellView="form"
# @title Create Labeling Images
SAMPLING_METHOD = 'uniform'  # @param ['representative', 'uniform']
MAX_LABELING_IMAGES = 500  # @param {'type': 'integer'}
TEST_PERCENTAGE = 20  # @param {'type': 'integer'}


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

def run_representative_sampling(
    scores_path: str,
    max_images: int,
    output_dir: str) -> str:
  """Runs the representative sampling algorithm."""
  with tf.io.gfile.GFile(scores_path, 'r') as f:
    scores_df = pd.read_csv(f)
  train_ratio = (100 - TEST_PERCENTAGE) / 100
  train_df, test_df = representative_sampling.run_representative_sampling(
      scores_df=scores_df,
      num_examples_to_sample_total=max_images,
      num_examples_to_take_from_top=100,
      grid_rows=4,
      grid_cols=4,
      train_ratio=train_ratio,
      buffer_meters=70,
  )
  # Combine the train and test sets and save to one output CSV.
  sampled_df = pd.concat([train_df, test_df])
  output_path = os.path.join(output_dir, 'samples.csv')
  with tf.io.gfile.GFile(output_path, 'w') as f:
    sampled_df.to_csv(f, index=False)
  return output_path


def create_labeling_images(
    example_metadata_pattern: str,
    tfrecord_pattern: str,
    parquet_pattern: str,
    scores_file: str,
    output_dir: str,
    max_images: int,
    sampling_method: str,
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

  if sampling_method == 'representative':
    allowed_example_ids_path = run_representative_sampling(
        scores_file,
        max_images,
        images_dir)
    scores_path = None
  else:
    allowed_example_ids_path = None
    scores_path = scores_file

  num_images = labeling.create_labeling_images(
      example_metadata_pattern,
      None,
      examples_pattern,
      max_images,
      allowed_example_ids_path,
      [],
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
      scores_path=scores_path,
      filter_by_column='is_cloudy',
  )
  metadata_csv = os.path.join(images_dir, 'image_metadata.csv')
  assert tf.io.gfile.exists(
      metadata_csv
  ), f'Metdata CSV not found at "{metadata_csv}"'
  print('Number of labeling images:', num_images)
  print(
      'Please create a new project in the SKAI labeling tool with the following'
      ' metadata CSV:'
  )
  print(metadata_csv)

  visualize_labeling_images(images_dir, 3)

create_labeling_images(
    EXAMPLE_METADATA_PATTERN,
    UNLABELED_TFRECORD_PATTERN,
    UNLABELED_PARQUET_PATTERN,
    ZERO_SHOT_SCORES,
    LABELING_IMAGES_DIR,
    MAX_LABELING_IMAGES,
    SAMPLING_METHOD,
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
MINOR_IS_DAMAGED = False  # @param {"type":"boolean"}

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
  for split, path in [('train', train_tfrecords), ('test', test_tfrecords)]:
    for features in tqdm.notebook.tqdm(
        tf.data.TFRecordDataset([path])
        .map(_parse_examples, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator(),
        desc=f'Reading {split} examples',
    ):
      longitudes.append(features['coordinates'][0])
      latitudes.append(features['coordinates'][1])
      columns['example_id'].append(features['example_id'].decode())
      columns['string_label'].append(features['string_label'].decode())
      columns['label'].append(features['label'])
      columns['file'].append(os.path.basename(path))

  return pd.DataFrame(columns)


def _format_counts_table(df: pd.DataFrame):
  df_copy = df.copy()
  for column in df_copy.columns:
    if column != 'All':
      df_copy[column] = [
          f'{int(x)} ({x/t * 100:0.2f}%)'
          for x, t in zip(df_copy[column], df_copy['All'])
      ]
  return df_copy


def _check_counts(float_counts: pd.DataFrame):
  for _, row in float_counts.iterrows():
    if row.name == 'All':
      continue
    if row[0.0] == 0:
      print(f'WARNING: {row.name} has no negative examples.')
    if row[1.0] == 0:
      print(f'WARNING: {row.name} has no positive examples.')


def show_label_stats(train_tfrecord: str, test_tfrecord: str):
  """Displays tables showing label count stats."""
  df = _load_examples_into_df(train_tfrecord, test_tfrecord)
  counts = df.pivot_table(
      index='file',
      columns='string_label',
      aggfunc='count',
      values='example_id',
      margins=True,
      fill_value=0)

  print('String Label Counts')
  IPython.display.display(data_table.DataTable(_format_counts_table(counts)))

  float_counts = df.pivot_table(
      index='file',
      columns='label',
      aggfunc='count',
      values='example_id',
      margins=True,
      fill_value=0.0)
  print('Float Label Counts')
  IPython.display.display(
      data_table.DataTable(_format_counts_table(float_counts))
  )

  _check_counts(float_counts)


def upload_label_csvs(output_path: str) -> bool:
  """Lets the user upload the labeling CSV file from their computer."""

  print('Upload labels CSV')
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

  combined = pd.concat(dfs, ignore_index=True)

  with _open_file(output_path, 'w') as f:
    combined.to_csv(f, index=False)
  print(f'Wrote {len(combined)} labels to {output_path}')
  return True


def create_labeled_examples(
    test_percent: int,
    minor_is_0: bool,
    labeling_images_root: str,
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

  labeling_images_dirs = [
      os.path.join(labeling_images_root, d)
      for d in tf.io.gfile.listdir(labeling_images_root)]
  labeling_images_dir = choose_option(
      labeling_images_dirs, 'Choose labeling images'
  )
  tfrecord_path = os.path.join(
      labeling_images_dir, 'labeling_examples.tfrecord'
  )
  if not tf.io.gfile.exists(tfrecord_path):
    print(f'File {tfrecord_path} does not exist. Cannot proceed.')
    return

  samples_path = os.path.join(labeling_images_dir, 'samples.csv')
  if tf.io.gfile.exists(samples_path):
    print(f'Using samples file {samples_path}')
    print(
        'Train/test split determined by samples file. Ignoring TEST_PERCENTAGE.'
    )
  else:
    samples_path = None

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
      example_patterns=[tfrecord_path],
      splits_path=samples_path,
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

  show_label_stats(train_path, test_path)

create_labeled_examples(
    TEST_PERCENTAGE,
    not MINOR_IS_DAMAGED,
    LABELING_IMAGES_DIR,
    LABELED_EXAMPLES_ROOT)

# %% [markdown]
# # Fine Tuning

# %% cellView="form"
# @title Train model

NUM_TRAINING_EPOCHS = 100  # @param {type:"integer"}


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
  labeled_example_dir = choose_option(
      find_labeled_examples_dirs(),
      'Choose a labeled examples dir')
  if labeled_example_dir is None:
    return
  train_path = os.path.join(labeled_example_dir, TRAIN_TFRECORD_NAME)
  test_path = os.path.join(labeled_example_dir, TEST_TFRECORD_NAME)
  model_dir = os.path.join(labeled_example_dir, 'models')
  run_training(
      ASSESSMENT_NAME, train_path, test_path, model_dir, NUM_TRAINING_EPOCHS)

choose_dataset_run_training()


# %% cellView="form"
# @title View Tensorboard (Optional)
# @markdown If you want to keep track of model training progress, you can run
# @markdown this cell and look at the Tensorboard graphs.<br>
# @markdown **Note:** It may take a while before the first Tensorboard log is
# @markdown generated.

def _kill_old_tensorboards():
  for tb in tensorboard.notebook.manager.get_all():
    print(f'Stopping old tensorboard instance (pid={tb.pid}, port={tb.port})')
    subprocess.check_call(['kill', str(tb.pid)])

def start_tensorboard():
  """Shows Tensorboard visualization."""
  _kill_old_tensorboards()

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

  tensorboard_dir = choose_option(
      tensorboard_dirs, 'Choose the model directory to show tensorboard for')
  tensorboard.notebook.start(f'--load_fast=false --logdir {tensorboard_dir}')

start_tensorboard()

# %% cellView="form"
# @title Evaluate Fine-Tuned Model
SHOW_DETAILED_GRAPHS = False  # @param {"type":"boolean"}

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


def true_positive_rate(labels: np.ndarray, predictions: np.ndarray) -> float:
  predicted_pos = np.sum((predictions == 1.0) & (labels == 1.0))
  true_pos = np.sum(labels == 1.0)
  return predicted_pos / true_pos


def true_negative_rate(labels: np.ndarray, predictions: np.ndarray) -> float:
  predicted_neg = np.sum((predictions == 0.0) & (labels == 0.0))
  true_neg = np.sum(labels == 0.0)
  return predicted_neg / true_neg


def false_positive_rate(labels: np.ndarray, predictions: np.ndarray) -> float:
  false_pos = np.sum((predictions == 1.0) & (labels == 0.0))
  true_neg = np.sum((predictions == 0.0) & (labels == 0.0))
  return false_pos / (false_pos + true_neg)


def false_negative_rate(labels: np.ndarray, predictions: np.ndarray) -> float:
  false_neg = np.sum((predictions == 0.0) & (labels == 1.0))
  true_pos = np.sum((predictions == 1.0) & (labels == 1.0))
  return false_neg / (false_neg + true_pos)


def get_max_f1_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
  """Finds the threshold that maximizes F1 score.

  Args:
    scores: Prediction scores assigned by the model.
    labels: True labels.

  Returns:
    Tuple of best threshold and F1-score, Precision, Recall at that threshold.
  """
  best_f1 = 0
  best_threshold = 0
  for threshold in scores:
    predictions = (scores >= threshold)
    if (f1 := sklearn.metrics.f1_score(labels, predictions)) > best_f1:
      best_f1 = f1
      best_threshold = threshold
  return best_threshold, best_f1


def plot_score_distribution(labels: np.ndarray, scores: np.ndarray) -> None:
  df = {'score': scores, 'label': labels}
  sns.displot(data=df, x='score', col='label')
  plt.show()


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
  precisions, recalls, _ = sklearn.metrics.precision_recall_curve(
      labels, scores
  )
  auprc = sklearn.metrics.auc(recalls, precisions)
  threshold, f1 = get_max_f1_threshold(scores, labels)
  predictions = (scores >= threshold)
  accuracy = sklearn.metrics.accuracy_score(labels, predictions)
  precision = sklearn.metrics.precision_score(labels, predictions)
  recall = sklearn.metrics.recall_score(labels, predictions)
  tpr = true_positive_rate(labels, predictions)
  tnr = true_negative_rate(labels, predictions)
  fpr = false_positive_rate(labels, predictions)
  fnr = false_negative_rate(labels, predictions)

  metrics_df = pd.DataFrame({
      'AUPRC': [auprc],
      'Threshold': [threshold],
      'F1-score': [f1],
      'Accuracy': [accuracy],
      'True Positive Rate': [tpr],
      'True Negative Rate': [tnr],
      'False Positive Rate': [fpr],
      'False Negative Rate': [fnr],
      'Precision': [precision],
      'Recall': [recall],
  })
  metrics_log = os.path.join(model_dir, 'metrics.csv')
  with _open_file(metrics_log, 'w') as f:
    metrics_df.to_csv(f, index=False)
  IPython.display.display(metrics_df.transpose())
  if SHOW_DETAILED_GRAPHS:
    plot_precision_recall(labels, scores)
    plot_score_distribution(labels, scores)


def evaluate_model_on_test_examples():
  """Lets user evaluate a model on chosen trained model and test examples.
  """
  labeled_example_dirs = find_labeled_examples_dirs()
  if not labeled_example_dirs:
    print(
        'No labeled example datasets found. Please create a labeled dataset'
        ' first.'
    )
    return

  model_dirs = find_model_dirs()
  if not model_dirs:
    print('No trained model directories found. Please train a model first.')
    return

  labeled_example_dir = choose_option(
      labeled_example_dirs,
      'Choose a labeled examples directory')

  model_dir = choose_option(model_dirs, 'Choose a model')

  test_path = os.path.join(labeled_example_dir, TEST_TFRECORD_NAME)
  model_dir = os.path.join(model_dir, 'model')
  checkpoint = get_best_checkpoint(model_dir)
  if not checkpoint:
    print('Model directory does not contain a valid checkpoint directory.')
    return
  print(f'Evaluating checkpoint {checkpoint} on examples {test_path}')
  _evaluate_model(checkpoint, test_path)

evaluate_model_on_test_examples()

# %% cellView="form"
# @title Run inference
# @markdown Leave INFERENCE_TFRECORD_PATTERN blank to run on default unlabeled
# @markdown examples.
INFERENCE_TFRECORD_PATTERN = ''  # @param {"type":"string"}
INFERENCE_FILE_NAME = 'inference_output.csv'  # @param {"type":"string"}


def run_inference(
    examples_pattern: str,
    checkpoint_dir: str,
    default_threshold: float,
    output_dir: str,
    cloud_project: str,
    cloud_region: str,
    service_account: str) -> None:
  """Starts model inference job."""

  output_path = os.path.join(output_dir, INFERENCE_FILE_NAME)
  if tf.io.gfile.exists(output_path):
    if input(f'File {output_path} exists. Overwrite? (y/n)').lower() not in [
        'y',
        'yes',
    ]:
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
    export GOOGLE_CLOUD_PROJECT={cloud_project}
    export PYTHONPATH={SKAI_CODE_DIR}/src:$PYTHONPATH
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

  model_dir = choose_option(model_dirs, 'Choose a model')
  checkpoint_dir = get_best_checkpoint(os.path.join(model_dir, 'model'))
  output_dir = os.path.join(model_dir, INFERENCE_SUBDIR)
  if not checkpoint_dir:
    print('Model directory does not contain a valid checkpoint directory.')
    return
  pattern = INFERENCE_TFRECORD_PATTERN or UNLABELED_TFRECORD_PATTERN

  metrics_csv_path = os.path.join(checkpoint_dir, 'metrics.csv')
  if not tf.io.gfile.exists(metrics_csv_path):
    print(
        'Model directory does not contain a valid metrics.csv file. Please'
        ' run the "Evaluate Fine-Tuned Model" cell.'
    )
    return
  with _open_file(metrics_csv_path, 'r') as f:
    metrics_df = pd.read_csv(f)
  threshold = metrics_df['Threshold'].iloc[0]
  print(f'Running inference with threshold {threshold:.4g}')

  run_inference(
      pattern,
      checkpoint_dir,
      threshold,
      output_dir,
      GCP_PROJECT,
      GCP_LOCATION,
      GCP_SERVICE_ACCOUNT,
  )

do_inference()

# %% cellView="form"
# @title Get assessment stats (Optional)

DAMAGE_SCORE_THRESHOLD = 0.9  # @param {type:"number"}


def get_assessment_stats():
  """Get assessment statistics."""
  inference_csvs = find_inference_csvs()
  if not inference_csvs:
    print('No inference CSVs found.')
    return

  inference_csv = choose_option(inference_csvs, 'Choose an inference CSV')
  show_inference_stats(AOI_PATH, inference_csv, DAMAGE_SCORE_THRESHOLD)

  make_download_button(
      inference_csv, os.path.basename(inference_csv), 'Download'
  )


get_assessment_stats()
