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

# pylint: disable=line-too-long

r"""Creates a labeling task on Vertex AI.

This script converts a subset of unlabeled TF examples into labeling image
format (before and after images of each building juxtaposed side-by-side),
uploads them to Vertex AI as a dataset, and creates a labeling job for it. [1]
It will assign the job to the labeler pool that you specify. You can create
labeler pools using the REST API documented at [2].

Example invocation:

python create_cloud_labeling_task.py \
  --cloud_project=my-cloud-project \
  --cloud_location=us-central1 \
  --dataset_name=some_dataset_name \
  --examples_pattern=gs://bucket/disaster/examples/unlabeled-large/*.tfrecord \
  --images_dir=gs://bucket/disaster/examples/labeling-images \
  --max_images=1000 \
  --cloud_labeler_emails=user1@gmail.com,user2@gmail.com

[1] https://cloud.google.com/vertex-ai/docs/datasets/data-labeling-job
[2] https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.specialistPools
"""
# pylint: enable=line-too-long


import time

from absl import app
from absl import flags
from absl import logging

from skai import cloud_labeling

FLAGS = flags.FLAGS
flags.DEFINE_string('cloud_project', None, 'GCP project name.', required=True)
flags.DEFINE_string('cloud_location', None, 'Project location.', required=True)
flags.DEFINE_string(
    'examples_pattern', None, 'Pattern matching TFRecords.', required=True
)
flags.DEFINE_string(
    'images_dir', None, 'Directory to write images to.', required=True
)
flags.DEFINE_string(
    'import_file_path', None,
    'If specified, use this import file directly instead of generating new '
    'images. Assumes that all images referenced by this import file exist.'
)
flags.DEFINE_list('exclude_import_file_patterns', [],
                  'File patterns for import files listing images not to '
                  'generate.')
flags.DEFINE_integer('max_images', 1000, 'Maximum number of images to label.')
flags.DEFINE_bool('randomize', True, 'If true, randomly sample images.')
flags.DEFINE_string('dataset_name', None, 'Dataset name')
flags.DEFINE_list(
    'label_classes',
    ['no_damage', 'minor_damage', 'major_damage', 'destroyed', 'bad_example'],
    'Label classes.')
flags.DEFINE_bool('use_google_managed_labelers', False,
                  'If true, assign the task to Google managed labeling pool.')
flags.DEFINE_string(
    'cloud_labeler_pool', None, 'Labeler pool. Format is '
    'projects/<project>/locations/us-central1/specialistPools/<id>.')
flags.DEFINE_list(
    'cloud_labeler_emails', None, 'Emails of workers of new labeler pool. '
    'First email will become the manager.')
flags.DEFINE_string('labeler_instructions_uri',
                    'gs://skai-public/labeling_instructions_v2.pdf',
                    'URI for instructions.')
flags.DEFINE_bool('generate_images_only', False,
                  'If true, script will only create labeling images and import '
                  'file, but will not upload the dataset to VertexAI or create '
                  'a labeling task.')
flags.DEFINE_string(
    'allowed_example_ids_path',
    None,
    'If specified, only allow example ids found in this text file.')
flags.DEFINE_bool(
    'use_multiprocessing',
    True,
    'If true, starts multiple processes to run task.',
)
flags.DEFINE_float(
    'buffered_sampling_radius',
    70.0,
    'The minimum distance between two examples for the two examples to be in'
    ' the labeling task.',
)


def _get_labeling_dataset_region(project_region: str) -> str:
  """Choose where to host a labeling dataset.

  As of November 2021, labeling datasets can only be created in "us-central1"
  and "europe-west4" regions. See

  https://cloud.google.com/vertex-ai/docs/general/locations#available-regions

  Args:
    project_region: The region of the project.

  Returns:
    Supported region for hosting the labeling dataset.
  """
  if project_region.startswith('europe-'):
    return 'europe-west4'
  return 'us-central1'


def main(unused_argv):
  timestamp = time.strftime('%Y%m%d_%H%M%S')
  timestamped_dataset = f'{FLAGS.dataset_name}_{timestamp}'

  if FLAGS.import_file_path:
    import_file_path = FLAGS.import_file_path
  else:
    num_images, import_file_path = cloud_labeling.create_labeling_images(
        FLAGS.examples_pattern,
        FLAGS.max_images,
        FLAGS.allowed_example_ids_path,
        FLAGS.exclude_import_file_patterns,
        FLAGS.images_dir,
        FLAGS.use_multiprocessing,
        FLAGS.buffered_sampling_radius)

    if num_images == 0:
      logging.fatal('No labeling images found.')
      return

    logging.info('Wrote %d labeling images.', num_images)
    logging.info('Wrote import file %s.', import_file_path)
    if FLAGS.generate_images_only:
      return

  if FLAGS.use_google_managed_labelers:
    labeler_pool = None
  elif FLAGS.cloud_labeler_pool is not None:
    labeler_pool = FLAGS.cloud_labeler_pool
  else:
    # Create a new labeler pool.
    if not FLAGS.cloud_labeler_emails:
      raise ValueError('Must provide at least one labeler email.')

    pool_display_name = f'{timestamped_dataset}_pool'
    labeler_pool = cloud_labeling.create_specialist_pool(
        FLAGS.cloud_project, FLAGS.cloud_location, pool_display_name,
        FLAGS.cloud_labeler_emails[:1], FLAGS.cloud_labeler_emails)
    logging.info('Created labeler pool: %s', labeler_pool)

  cloud_labeling.create_cloud_labeling_job(
      FLAGS.cloud_project,
      _get_labeling_dataset_region(FLAGS.cloud_location),
      timestamped_dataset,
      FLAGS.label_classes,
      import_file_path,
      FLAGS.labeler_instructions_uri,
      labeler_pool)

  if labeler_pool:
    pool_id = labeler_pool.split('/')[-1]
    pool_location = labeler_pool.split('/')[-3].replace('-', '_')
    labeling_url = f'https://datacompute.google.com/w/cloudml_data_specialists_{pool_location}_{pool_id}'
    logging.info('Labeling URL: %s', labeling_url)


if __name__ == '__main__':
  app.run(main)
