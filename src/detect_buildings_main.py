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
import argparse
import logging
import apache_beam as beam
import geopandas as gpd

from skai import extract_tiles
from skai import detect_buildings

PipelineOptions = beam.options.pipeline_options.PipelineOptions

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_path', required=True, help='Path of input GeoTIFF.')
  parser.add_argument('--aoi_path', required=True, help='Path to AOI geojson.')
  parser.add_argument(
      '--output_prefix',
      required=True,
      help='Path prefix for output TFRecords.')
  parser.add_argument(
      '--output_shards', type=int, default=20, help='Number of output shards.')

  # By default, use a tile size of 540 and a margin size of 50 so that each full
  # tile is size 640 x 640, a common input size for building segmentation
  # models.
  parser.add_argument(
      '--tile_size', type=int, default=540, help='Tile size in pixels.')
  parser.add_argument(
      '--margin', type=int, default=50, help='Margin size in pixels.')
  parser.add_argument(
      '--model_path',
      type=str,
      default='',
      help='Path to building segmentation model\'s SavedModel directory.')

  args, pipeline_args = parser.parse_known_args()

  pipeline_options = PipelineOptions(pipeline_args)

  gdf = gpd.read_file(args.aoi_path)
  aoi = gdf.geometry.values[0]
  tiles = list(extract_tiles.get_tiles_for_aoi(
      args.input_path, aoi, args.tile_size, args.margin, {}))
  print(f'Extracting {len(tiles)} tiles total')

  with beam.Pipeline(options=pipeline_options) as pipeline:
    buildings = (
        pipeline
        | 'CreateTiles' >> beam.Create(tiles)
        | 'ExtractTiles' >> beam.ParDo(
            extract_tiles.ExtractTilesAsExamplesFn(args.input_path, {}))
        | 'DetectBuildings' >> beam.ParDo(
            detect_buildings.DetectBuildingsFn(args.model_path)))

    detect_buildings.write_buildings(
        buildings, args.output_prefix, args.output_shards, 'Buildings')
    deduplicated_buildings = detect_buildings.deduplicate_buildings(buildings)
    detect_buildings.write_buildings(
        deduplicated_buildings, args.output_prefix + '_dedup',
        args.output_shards, 'DedupedBuildings')
    detect_buildings.write_centroids_csv(
        deduplicated_buildings, args.output_prefix + '_dedup_centroids.csv')
