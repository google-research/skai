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

"""Runs model inference."""

import time

from absl import app
from absl import flags
from skai import beam_utils
from skai.model import inference_lib


ModelType = inference_lib.ModelType
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'examples_pattern', None, 'File pattern for input TFRecords.', required=True
)
flags.DEFINE_string(
    'image_model_dir', None, 'Saved image model directory.', required=True
)
flags.DEFINE_string(
    'output_prefix', None, 'Path prefix for output files.', required=True
)
flags.DEFINE_integer('image_size', 224, 'Expected image width and height.')
flags.DEFINE_bool('post_images_only', False, 'Model expects only post images')
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_string('cloud_project', None, 'GCP project name.')
flags.DEFINE_string('cloud_region', None, 'GCP region, e.g. us-central1.')
flags.DEFINE_bool('use_dataflow', True, 'If true, run pipeline on Dataflow.')
flags.DEFINE_string(
    'worker_service_account',
    None,
    'Service account that will launch Dataflow workers. If unset, workers will '
    "run with the project's default Compute Engine service account.",
)
flags.DEFINE_string('dataflow_temp_dir', '', 'Temp dir.')
flags.DEFINE_integer(
    'max_dataflow_workers', None, 'Maximum number of dataflow workers'
)
flags.DEFINE_string('worker_type', 'c3-standard-8', 'Dataflow worker type.')
flags.DEFINE_string(
    'worker_machine_type', 'n1-highmem-8', 'worker machine type'
)
flags.DEFINE_string('accelerator', None, 'Accelerator to use.')

# Setting experiments flags to no_use_multiple_sdk_containers is important when
# using GPU, Because it will save memory
flags.DEFINE_string(
    'experiments',
    'use_runner_v2',
    'Enable pre-GA Dataflow features. Setting experiments flags to'
    ' no_use_multiple_sdk_containers is important when using GPU, Because it'
    ' will save memory',
)
flags.DEFINE_integer('accelerator_count', 1, 'Number of accelerators to use.')
flags.DEFINE_float('threshold', 0.5, 'Damaged score threshold.')
flags.DEFINE_float(
    'high_precision_threshold',
    0.5,
    'Damaged score threshold for high precision.',
)
flags.DEFINE_float(
    'high_recall_threshold', 0.5, 'Damaged score threshold for high recall.'
)
flags.DEFINE_bool('deduplicate', True, 'Merge examples of the same building.')
flags.DEFINE_bool('generate_embeddings', False, 'Generate embeddings.')
flags.DEFINE_list(
    'post_image_order',
    [],
    'List of post-disaster image ids in descending priority order.',
)
flags.DEFINE_bool(
    'wait_for_dataflow',
    False,
    'If true, wait for Dataflow job to finish before exiting.',
)


def main(_) -> None:
  timestamp = time.strftime('%Y%m%d-%H%M%S')
  job_name = f'skai-inference-{timestamp}'

  pipeline_options = beam_utils.get_pipeline_options(
      FLAGS.use_dataflow,
      job_name,
      FLAGS.cloud_project,
      FLAGS.cloud_region,
      FLAGS.dataflow_temp_dir,
      min_workers=min(FLAGS.max_dataflow_workers, 10),
      max_workers=FLAGS.max_dataflow_workers,
      worker_service_account=FLAGS.worker_service_account,
      machine_type=FLAGS.worker_machine_type,
      accelerator=FLAGS.accelerator,
      accelerator_count=FLAGS.accelerator_count,
      experiments=FLAGS.experiments,
  )

  inference_lib.run_tf2_inference_with_csv_output(
      FLAGS.examples_pattern,
      FLAGS.image_model_dir,
      FLAGS.output_prefix,
      FLAGS.image_size,
      FLAGS.post_images_only,
      FLAGS.batch_size,
      FLAGS.threshold,
      FLAGS.high_precision_threshold,
      FLAGS.high_recall_threshold,
      FLAGS.deduplicate,
      FLAGS.post_image_order,
      FLAGS.generate_embeddings,
      FLAGS.wait_for_dataflow,
      pipeline_options,
  )


if __name__ == '__main__':
  app.run(main)
