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
r"""Runs example generation pipeline.

Example invocation to run on workstation:

python generate_examples_main.py \
  --before_image_patterns=/path/to/before_*_1.tif,/path/to/before_*_2.tif \
  --after_image_patterns=/path/to/after_*_1.tif,/path/to/after_*_2.tif \
  --aoi_path=/path/to/aoi.geojson \
  --buildings_method=open_street_map \
  --output_dir=/path/to/output \


Example invocation to run on Cloud DataFlow:

python generate_examples_main.py \
  --before_image_patterns=gs://bucket-name/before_image.tif \
  --after_image_patterns=gs://bucket-name/after_image.tif \
  --aoi_path=/path/to/aoi.geojson \
  --buildings_method=open_street_map \
  --output_dir=gs://bucket-name/disaster-name \
  --use_dataflow \
  --cloud_project=disaster-assessment \
  --cloud_region=us-west1
"""

import os
import time
from typing import List

from absl import app
from absl import flags
from absl import logging
import shapely.geometry
from skai import buildings
from skai import generate_examples
from skai import read_raster
import tensorflow as tf

FLAGS = flags.FLAGS

# See generate_examples.ExampleGenerationConfig for default values.
# General GCP flags.
flags.DEFINE_string('cloud_project', None, 'GCP project name.')
flags.DEFINE_string('cloud_region', None, 'GCP region, e.g. us-central1.')
flags.DEFINE_bool('use_dataflow', None, 'If true, run pipeline on Dataflow.')
flags.DEFINE_string(
    'worker_service_account', None,
    'Service account that will launch Dataflow workers. If unset, workers will '
    'run with the project\'s default Compute Engine service account.')
flags.DEFINE_integer(
    'max_dataflow_workers', None, 'Maximum number of dataflow workers'
)

# Example generation flags.
flags.DEFINE_string(
    'dataset_name', None, 'Identifier for the generated dataset.'
)
flags.DEFINE_list(
    'before_image_patterns',
    None,
    'Comma-separated list of path patterns of pre-disaster GeoTIFFs.',
)
flags.DEFINE_list(
    'after_image_patterns',
    None,
    'Comma-separated list of paths of post-disaster GeoTIFFs.',
)
flags.DEFINE_string('before_image_config', None,
                    'Before image config file path.')
flags.DEFINE_string('after_image_config', None, 'After image config file path.')
flags.DEFINE_string(
    'aoi_path', None, 'Path to file containing area of interest')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_integer('example_patch_size', None, 'Image patch size.')
flags.DEFINE_integer(
    'large_patch_size', None, 'Patch size used for labeling and alignment.'
)
flags.DEFINE_float(
    'resolution', 0.5,
    'The desired resolution (in m/pixel) of the image patches. If this is '
    'different from the image\'s native resolution, patches will be upsampled '
    'or downsampled.')

flags.DEFINE_integer('output_shards', None, 'Number of output shards.')
flags.DEFINE_list(
    'gdal_env',
    None,
    (
        'Environment configuration for GDAL. Comma delimited list '
        'where each element has the form "var=value".'
    ),
)
flags.DEFINE_bool(
    'output_metadata_file',
    False,
    'Enable true to generate a file of example metadata, or disable to skip'
    ' this step.',
)

# Building discovery flags.
flags.DEFINE_enum(
    'buildings_method',
    None,
    ['file', 'open_street_map', 'open_buildings', 'none'],
    'Building detection method.',
)
flags.DEFINE_string(
    'buildings_file', None, 'Path to file containing building locations. '
    'Supports CSV, shapefile, and GeoJSON.')
flags.DEFINE_string('overpass_url', None, 'OpenStreetMap Overpass server URL.')
flags.DEFINE_string(
    'open_buildings_feature_collection',
    None,
    (
        'Name of Earth Engine feature collection containing Open Buildings '
        'footprints.'
    ),
)

# Flags controlling Earth Engine authentication.
flags.DEFINE_string(
    'earth_engine_service_account',
    None,
    (
        'Service account to use for authenticating with Earth Engine. If empty,'
        ' authenticate as user.'
    ),
)
flags.DEFINE_string(
    'earth_engine_private_key',
    None,
    (
        'Private key for Earth Engine service account. Not needed if'
        ' authenticating as user.'
    ),
)

# Flags controlling the ingestion of user-provided labels.
flags.DEFINE_string(
    'labels_file', None, 'If specified, read labels for dataset from this file.'
)
flags.DEFINE_string(
    'label_property', None, 'Property to use as label, e.g. "string_label".'
)
flags.DEFINE_list(
    'labels_to_classes',
    None,
    'Mapping of label from dataset and class for model.',
)
flags.DEFINE_integer(
    'num_keep_labeled_examples',
    None,
    'Number of labeled examples to keep (keeps all if None or 0).',
)
flags.DEFINE_string(
    'configuration_path', None, 'A path to a json configuration file'
)
flags.DEFINE_bool(
    'wait_for_dataflow',
    True,
    'Wait for dataflow job to finish before finishing execution.',
)

Polygon = shapely.geometry.polygon.Polygon

BUILDINGS_FILE_NAME = 'processed_buildings.parquet'


def _read_image_config(path: str) -> List[str]:
  with tf.io.gfile.GFile(path, 'r') as f:
    return [line.strip() for line in f.readlines()]


def main(args):
  del args  # unused

  config = None
  if FLAGS.configuration_path:
    config = generate_examples.ExamplesGenerationConfig.init_from_json_path(
        FLAGS.configuration_path
    )
  else:
    config = generate_examples.ExamplesGenerationConfig.init_from_flags(FLAGS)

  timestamp = time.strftime('%Y%m%d-%H%M%S')
  timestamped_dataset = f'{config.dataset_name}-{timestamp}'

  gdal_env = generate_examples.parse_gdal_env(config.gdal_env)
  if config.before_image_patterns:
    before_image_patterns = config.before_image_patterns
  elif config.before_image_config:
    before_image_patterns = _read_image_config(config.before_image_config)
  else:
    before_image_patterns = []

  if config.after_image_patterns:
    after_image_patterns = config.after_image_patterns
  elif config.after_image_config:
    after_image_patterns = _read_image_config(config.after_image_config)
  else:
    after_image_patterns = []

  # Validate before_image_patterns and after_image_patterns
  generate_examples.validate_image_patterns(before_image_patterns, False)
  generate_examples.validate_image_patterns(after_image_patterns, True)

  if not config.labels_file and config.buildings_method == 'none':
    raise ValueError('At least labels_file (for labeled examples extraction) '
                     'or buildings_method != none (for unlabeled data) should '
                     'be specified.')

  buildings_path = os.path.join(config.output_dir, BUILDINGS_FILE_NAME)
  if config.labels_file:
    generate_examples.read_labels_file(
        config.labels_file,
        config.label_property,
        config.labels_to_classes,
        config.num_keep_labeled_examples,
        buildings_path
    )
    buildings_labeled = True
  else:
    if config.aoi_path:
      aois = buildings.read_aois(config.aoi_path)
    else:
      aois = [read_raster.get_raster_bounds(path, gdal_env)
              for path in after_image_patterns]
    try:
      generate_examples.download_building_footprints(
          config, aois, buildings_path
      )
    except generate_examples.NotInitializedEarthEngineError:
      logging.fatal('Could not initialize Earth Engine.', exc_info=True)
    except generate_examples.NoBuildingFoundError:
      logging.fatal('No building is found.', exc_info=True)
    buildings_labeled = False

  generate_examples.generate_examples_pipeline(
      before_image_patterns,
      after_image_patterns,
      config.large_patch_size,
      config.example_patch_size,
      config.resolution,
      config.output_dir,
      config.output_shards,
      buildings_path,
      buildings_labeled,
      config.use_dataflow,
      gdal_env,
      timestamped_dataset,
      config.cloud_project,
      config.cloud_region,
      config.worker_service_account,
      config.max_dataflow_workers,
      FLAGS.wait_for_dataflow,
      config.cloud_detector_model_path,
      config.output_metadata_file
  )


if __name__ == '__main__':
  app.run(main)
