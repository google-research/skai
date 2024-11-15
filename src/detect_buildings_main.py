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

r"""Uses ML model to detect buildings in  input GeoTIFF.


Example invocation to run on workstation:

python detect_buildings_main.py \
  --input_path=/path/to/image.tif \
  --model_path=gs://my-bucket/model \
  --output_prefix=/path/to/buildings


Example invocation to run on Cloud Dataflow.

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

IMAGE=gcr.io/disaster-assessment/dataflow_image_3.9_image:latest
python detect_buildings_main.py \
  --input_path=gs://my-bucket/image.tif \
  --model_path=gs://my-bucket/model \
  --output_prefix=gs://my-bucket/buildings \
  --runner=DataflowRunner \
  --project=my-project \
  --region=us-west1 \
  --worker_service_account='sa@my-project.iam.gserviceaccount.com' \
  --experiment=use_runner_v2 \
  --sdk_container_image=$IMAGE \
  --setup_file "$(pwd)/setup.py"
"""

import datetime
import logging
import os

from absl import app
from absl import flags

import apache_beam as beam
import geopandas as gpd
from skai import beam_utils
from skai import detect_buildings
from skai import extract_tiles
from skai import read_raster
from skai import utils

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('cloud_project', None, 'GCP project name.')
flags.DEFINE_string('cloud_region', None, 'GCP region, e.g. us-central1.')
flags.DEFINE_bool('use_dataflow', None, 'If true, run pipeline on Dataflow.')
flags.DEFINE_string(
    'worker_service_account', None,
    'Service account that will launch Dataflow workers. If unset, workers will '
    'run with the project\'s default Compute Engine service account.')
flags.DEFINE_integer(
    'min_dataflow_workers', 10, 'Minimum number of dataflow workers'
)
flags.DEFINE_integer(
    'max_dataflow_workers', None, 'Maximum number of dataflow workers'
)
flags.DEFINE_string('worker_machine_type', 'n1-standard-8', 'Worker type.')
flags.DEFINE_list('image_paths', None, 'Paths of input images.', required=True)
flags.DEFINE_bool(
    'mosaic_images',
    False,
    'If true, mosaic all images into a single image using a VRT.',
)
flags.DEFINE_string('aoi_path', None, 'Path of AOI GeoJSON.', required=True)
flags.DEFINE_string(
    'model_path',
    None,
    "Path to building segmentation model's SavedModel directory.",
)
flags.DEFINE_string('output_dir', None, 'Output directory.', required=True)
flags.DEFINE_integer('output_shards', 20, 'Number of output shards.')
# By default, use a tile size of 540 and a margin size of 50 so that each full
# tile is size 640 x 640, a common input size for building segmentation
# models.
flags.DEFINE_integer('tile_size', 540, 'Tile size in pixels.')
flags.DEFINE_integer('margin', 50, 'Margin size in pixels.')
flags.DEFINE_float(
    'detection_confidence_threshold',
    0.2,
    'Confidence threshold for building detection. All instances below this'
    ' detection score will be dropped.',
)
flags.DEFINE_list(
    'gdal_env',
    None,
    (
        'Environment configuration for GDAL. Comma delimited list '
        'where each element has the form "var=value".'
    ),
)

PipelineOptions = beam.options.pipeline_options.PipelineOptions


def main(args):
  del args  # unused

  logging.getLogger().setLevel(logging.INFO)

  temp_dir = os.path.join(FLAGS.output_dir, 'temp')
  timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  dataflow_job_name = f'detect-buildings-{timestamp}'

  pipeline_options = beam_utils.get_pipeline_options(
      FLAGS.use_dataflow,
      dataflow_job_name,
      FLAGS.cloud_project,
      FLAGS.cloud_region,
      temp_dir,
      FLAGS.min_dataflow_workers,
      FLAGS.max_dataflow_workers,
      FLAGS.worker_service_account,
      machine_type=FLAGS.worker_machine_type,
      accelerator=None,
      accelerator_count=0,
  )

  with tf.io.gfile.GFile(FLAGS.aoi_path, mode='rb') as f:
    gdf = gpd.read_file(f)
  aoi = gdf.geometry.values[0]
  gdal_env = read_raster.parse_gdal_env(FLAGS.gdal_env)
  image_paths = utils.expand_file_patterns(FLAGS.image_paths)
  for image_path in image_paths:
    if not read_raster.raster_is_tiled(image_path):
      raise ValueError(f'Raster "{image_path}" is not tiled.')

  vrt_paths = read_raster.build_vrts(
      image_paths, os.path.join(temp_dir, 'image'), 0.5, FLAGS.mosaic_images
  )

  tiles = []
  for path in vrt_paths:
    tiles.extend(
        extract_tiles.get_tiles_for_aoi(
            path, aoi, FLAGS.tile_size, FLAGS.margin, gdal_env
        )
    )
  print(f'Extracting {len(tiles)} tiles total')

  with beam.Pipeline(options=pipeline_options) as pipeline:
    buildings = (
        pipeline
        | 'CreateTiles' >> beam.Create(tiles)
        | 'ExtractTiles'
        >> beam.ParDo(extract_tiles.ExtractTilesAsExamplesFn({}))
        | 'DetectBuildings'
        >> beam.ParDo(
            detect_buildings.DetectBuildingsFn(
                FLAGS.model_path, FLAGS.detection_confidence_threshold
            )
        )
    )

    detect_buildings.write_tfrecords(
        buildings,
        os.path.join(FLAGS.output_dir, 'buildings'),
        FLAGS.output_shards,
        'Buildings',
    )
    deduplicated_buildings = detect_buildings.deduplicate_buildings(buildings)
    detect_buildings.write_tfrecords(
        deduplicated_buildings,
        os.path.join(FLAGS.output_dir, 'dedup_buildings'),
        FLAGS.output_shards,
        'DedupedBuildings',
    )
    detect_buildings.write_csv(
        deduplicated_buildings,
        os.path.join(FLAGS.output_dir, 'dedup_buildings.csv'),
    )

  detect_buildings.combine_csvs(
      os.path.join(FLAGS.output_dir, 'dedup_buildings.csv-*-of-*'),
      os.path.join(FLAGS.output_dir, 'dedup_buildings'),
  )

if __name__ == '__main__':
  app.run(main)
