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

"""Functions for performing data labeling in GCP Vertex AI."""

import json
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

from absl import logging

from google.cloud import aiplatform
from google.cloud import aiplatform_v1
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from skai import utils
import tensorflow as tf

from google.protobuf import struct_pb2
from google.protobuf import json_format


# Gap to add between before and after images.
BEFORE_AFTER_GAP = 10

# Margin around caption text.
CAPTION_MARGIN = 32

# Size of the reticule that identifies the building being labeled.
RETICULE_HALF_LEN = 32

# Name of generated import file.
IMPORT_FILE_NAME = 'import_file.jsonl'

# Schema for image classification tasks.
LABEL_SCHEMA_URI = ('gs://google-cloud-aiplatform/schema/datalabelingjob/'
                    'inputs/image_classification_1.0.0.yaml')

Example = tf.train.Example
Image = PIL.Image.Image


def _get_api_endpoint(cloud_location: str) -> str:
  return f'{cloud_location}-aiplatform.googleapis.com'


def _annotate_image(image: Image, caption: str) -> Image:
  """Adds center square and caption to image.

  Args:
    image: Input image.
    caption: Caption text to add.

  Returns:
    A copy of the input image with annotations added.
  """
  # Copy image into bigger frame to have room for caption.
  annotated_image = PIL.Image.new('RGB',
                                  (image.width, image.height + CAPTION_MARGIN),
                                  (225, 225, 225))
  annotated_image.paste(image, (0, 0))

  # Draw center rectangle.
  cx = image.width // 2
  cy = image.height // 2
  coords = [(cx - RETICULE_HALF_LEN, cy - RETICULE_HALF_LEN),
            (cx + RETICULE_HALF_LEN, cy + RETICULE_HALF_LEN)]
  annotations = PIL.ImageDraw.Draw(annotated_image)
  annotations.rectangle(coords, outline=(255, 0, 0), width=1)

  # Add caption.
  caption_xy = (cx, image.height + 5)
  annotations.text(caption_xy, caption, fill='black', anchor='mt')
  return annotated_image


def create_labeling_image(before_image: Image, after_image: Image) -> Image:
  """Creates an image used for labeling.

  The image is composed of the before and after images from the input example
  side-by-side.

  Args:
    before_image: Before image.
    after_image: After image.

  Returns:
    Combined image.

  """
  before_annotated = _annotate_image(before_image, 'BEFORE')
  after_annotated = _annotate_image(after_image, 'AFTER')
  width = before_annotated.width + after_annotated.width + 3 * BEFORE_AFTER_GAP
  height = before_annotated.height + 2 * BEFORE_AFTER_GAP
  combined = PIL.Image.new('RGB', (width, height), (225, 225, 225))
  combined.paste(before_annotated, (BEFORE_AFTER_GAP, BEFORE_AFTER_GAP))
  combined.paste(after_annotated,
                 (before_annotated.width + 2 * BEFORE_AFTER_GAP,
                  BEFORE_AFTER_GAP))
  return combined


def create_labeling_images(
    examples_pattern: str,
    max_images: int,
    output_dir: str) -> Tuple[int, str]:
  """Creates PNGs used for labeling from TFRecords.

  Also writes an import file in CSV format that is used to upload the images
  into the VertexAI labeling tool.

  Args:
    examples_pattern: File pattern for input TFRecords.
    max_images: Maximum number of images to create.
    output_dir: Output directory.

  Returns:
    Tuple of number of images written and file path for the import file.
  """
  example_files = tf.io.gfile.glob(examples_pattern)
  image_paths = []
  for record in tf.data.TFRecordDataset(example_files):
    example = Example()
    example.ParseFromString(record.numpy())
    example_id = (
        example.features.feature['encoded_coordinates'].bytes_list.value[0]
        .decode())
    before_image = utils.deserialize_image(
        example.features.feature['pre_image_png_large'].bytes_list.value[0],
        'png')
    after_image = utils.deserialize_image(
        example.features.feature['post_image_png_large'].bytes_list.value[0],
        'png')
    labeling_image = create_labeling_image(before_image, after_image)
    labeling_image_bytes = utils.serialize_image(labeling_image, 'png')
    path = os.path.join(output_dir, f'{example_id}.png')
    with tf.io.gfile.GFile(path, 'w') as writer:
      writer.write(labeling_image_bytes)
    image_paths.append(path)
    if len(image_paths) >= max_images:
      break

  import_file_path = os.path.join(output_dir, 'import_file.csv')
  with tf.io.gfile.GFile(import_file_path, 'w') as f:
    f.write('\n'.join(image_paths))
  return len(image_paths), import_file_path


def write_import_file(
    images_dir: str,
    max_images: int,
    randomize: bool,
    output_path: str) -> None:
  """Writes import file.

  This file tells Vertex AI how to import the generated dataset.

  Args:
    images_dir: Directory containing images.
    max_images: Maximum number of images to include in import file.
    randomize: If true, randomly sample images. Otherwise just take the
        first ones by file sort order.
    output_path: Path to write import file to.
  """
  images_pattern = os.path.join(images_dir, '*.png')
  image_files = sorted(tf.io.gfile.glob(images_pattern))
  if not image_files:
    raise ValueError(f'Pattern "{images_pattern}" did not match any images.')
  if randomize:
    image_files = random.sample(image_files, min(max_images, len(image_files)))
  else:
    image_files = image_files[:max_images]
  with tf.io.gfile.GFile(output_path, 'w') as f:
    f.write('\n'.join(image_files))


def create_cloud_labeling_job(
    project: str,
    location: str,
    dataset_name: str,
    label_classes: List[str],
    import_file_uri: str,
    instruction_uri: str,
    specialist_pool: Optional[str]) -> None:
  """Creates Cloud dataset and labeling job.

  Args:
    project: Cloud project.
    location: Cloud location, e.g. us-central1.
    dataset_name: Name for created dataset and labeling job.
    label_classes: Class names used in labeling task.
    import_file_uri: URI to import file for all example images.
    instruction_uri: URI to labeling instructions PDF.
    specialist_pool: Name of labeler pool to use. Has the form
      projects/<project_id>/locations/us-central1/specialistPools/<pool_id>. If
      None, the Google managed pool will be used.
  """
  aiplatform.init(project=project, location=location)
  import_schema_uri = (
      aiplatform.schema.dataset.ioformat.image.single_label_classification)
  dataset = aiplatform.ImageDataset.create(
      display_name=dataset_name,
      gcs_source=import_file_uri,
      import_schema_uri=import_schema_uri,
      sync=True)

  logging.info('Waiting for dataset to be created.')
  dataset.wait()
  # TODO(jzxu): Add error checking for dataset creation.
  logging.info('Dataset created: %s', dataset.resource_name)

  # Wait a while before using this dataset to create a labeling task. Otherwise
  # the labeling task creation request may fail with an "invalid argument"
  # error.
  #
  # If we find that this is still unreliable, we can resort to retrying the
  # labeling task creation request multiple times with sleeps between each
  # retry.
  time.sleep(15)

  # Create labeling job for newly created dataset.
  client_options = {'api_endpoint': _get_api_endpoint(location)}
  client = aiplatform.gapic.JobServiceClient(client_options=client_options)
  inputs_dict = {'annotation_specs': label_classes}
  inputs = json_format.ParseDict(inputs_dict, struct_pb2.Value())

  data_labeling_job_config = {
      'display_name': dataset_name,
      'datasets': [dataset.resource_name],
      'labeler_count': 1,
      'instruction_uri': instruction_uri,
      'inputs_schema_uri': LABEL_SCHEMA_URI,
      'inputs': inputs
  }

  if specialist_pool:
    data_labeling_job_config['annotation_labels'] = {
        'aiplatform.googleapis.com/annotation_set_name':
            'data_labeling_job_specialist_pool'
    }
    data_labeling_job_config['specialist_pools'] = [specialist_pool]
  else:
    data_labeling_job_config['annotation_labels'] = {
        'aiplatform.googleapis.com/annotation_set_name':
            'data_labeling_job_google_managed_pool'
    }

  logging.log(logging.DEBUG, 'Data labeling job config: "%s"',
              repr(data_labeling_job_config))
  logging.info('Creating labeling job.')
  data_labeling_job = client.create_data_labeling_job(
      parent=f'projects/{project}/locations/{location}',
      data_labeling_job=data_labeling_job_config
  )
  if data_labeling_job.error.code != 0:
    logging.error('Data labeling job error: "%s"', data_labeling_job.error)
  else:
    logging.info('Data labeling job created: "%s"', data_labeling_job.name)


def create_specialist_pool(project: str,
                           location: str,
                           display_name: str,
                           manager_emails: List[str],
                           worker_emails: List[str]) -> str:
  """Creates a specialist labeling pool in Vertex AI.

  Args:
    project: Cloud project name.
    location: Cloud project location.
    display_name: Display name for the specialist pool.
    manager_emails: Emails of managers.
    worker_emails: Emails of workers.

  Returns:
    Name of the specialist pool. Has the form
    projects/<project_id>/locations/us-central1/specialistPools/<pool_id>.
  """
  client_options = {'api_endpoint': _get_api_endpoint(location)}
  client = aiplatform_v1.SpecialistPoolServiceClient(
      client_options=client_options)

  pool = aiplatform_v1.SpecialistPool()
  pool.display_name = display_name
  pool.specialist_manager_emails.extend(set(manager_emails))
  pool.specialist_worker_emails.extend(set(worker_emails))

  request = aiplatform_v1.CreateSpecialistPoolRequest(
      parent=f'projects/{project}/locations/{location}',
      specialist_pool=pool,
  )

  logging.log(logging.DEBUG, 'Create specialist pool request: "%s"',
              repr(request))
  operation = client.create_specialist_pool(request=request)
  response = operation.result()
  return response.name


def _read_label_annotations_file(path: str) -> Dict[str, str]:
  """Reads labels out of annotations file.

  For details on annotation file format, see

  # pylint: disable=line-too-long
  https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets#understanding_your_datasets_format
  # pylint: enable=line-too-long

  Args:
    path: Path to annotations file.

  Returns:
    Dictionary of example ids to float label.
  """
  labels = {}
  with tf.io.gfile.GFile(path, 'r') as f:
    for line in f:
      record = json.loads(line.strip())
      if 'classificationAnnotation' not in record:
        continue
      # Assumes that the image path stem is the example id.
      # e.g gs://bucket/path/to/image/0D61BAC242D5F141.png
      example_id = os.path.basename(record['imageGcsUri'])[:-4]
      label = record['classificationAnnotation']['displayName']
      labels[example_id] = label
  return labels


def _write_tfrecord(examples: Iterable[Example], path: str) -> None:
  """Writes a list of examples to a TFRecord file."""
  output_dir = os.path.dirname(path)
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  with tf.io.TFRecordWriter(path) as writer:
    for example in examples:
      writer.write(example.SerializeToString())


def _string_to_float_label(label: str) -> float:
  """Converts string labels supplied by labelers to binary float labels."""
  # TODO(jzxu): Make the mapping from string to float values configurable.
  if label in ['damaged_destroyed']:
    return 1.0
  return 0.0


def _split_examples(
    examples: List[Example],
    test_fraction: float
) -> Tuple[List[Example], List[Example]]:
  """Splits a list of examples into training and test sets.

  Args:
    examples: Input examples.
    test_fraction: Fraction of examples to use for testing.

  Returns:
    Tuple of (training examples, test examples).
  """
  shuffled = examples[:]
  random.shuffle(shuffled)
  num_test = int(len(shuffled) * test_fraction)
  return shuffled[num_test:], shuffled[:num_test]


def _merge_examples_and_labels(examples_pattern: str,
                               labels: Dict[str, str],
                               test_fraction: float,
                               train_output_path: str,
                               test_output_path: str) -> None:
  """Merges examples with labels into train and test TFRecords.

  Args:
    examples_pattern: File pattern for input examples.
    labels: Dictionary of example ids to labels.
    test_fraction: Fraction of examples to write to test output.
    train_output_path: Path to training examples TFRecord output.
    test_output_path: Path to test examples TFRecord output.
  """
  example_files = tf.io.gfile.glob(examples_pattern)
  labeled_examples = []
  for example_file in example_files:
    logging.info('Processing unlabeled example file "%s".', example_file)
    for record in tf.data.TFRecordDataset([example_file]):
      example = Example()
      example.ParseFromString(record.numpy())
      example_id = (
          example.features.feature['encoded_coordinates'].bytes_list.value[0]
          .decode())
      if example_id in labels:
        label = _string_to_float_label(labels[example_id])
        label_feature = example.features.feature['label'].float_list
        if not label_feature.value:
          label_feature.value.append(label)
        else:
          label_feature.value[0] = label
        labeled_examples.append(example)
    if len(labeled_examples) == len(labels):
      break

  train_examples, test_examples = _split_examples(labeled_examples,
                                                  test_fraction)
  _write_tfrecord(train_examples, train_output_path)
  _write_tfrecord(test_examples, test_output_path)


def _get_labels(
    project: str,
    location: str,
    dataset_id: str,
    export_dir: str) -> Dict[str, str]:
  """Reads labels from completed cloud labeling job.

  Args:
    project: Cloud project name.
    location: Dataset location, e.g. us-central1.
    dataset_id: Numeric id of the dataset to export.
    export_dir: GCS directory to export annotations to.

  Returns:
    Dictionary of example ids to string labels.
  """

  aiplatform.init(project=project, location=location)
  dataset = aiplatform.ImageDataset(dataset_name=dataset_id)
  exported_file_paths = dataset.export_data(export_dir)

  # There will be multiple exported jsonl files. The one in the subdirectory
  # that starts with "data_labeling_job" contains the class annotations that the
  # labelers provided.
  labels = {}
  for path in exported_file_paths:
    parent_dir = path.split('/')[-2]
    if (parent_dir.startswith('data_labeling_job') and
        path.endswith('jsonl')):
      logging.info('Reading annotation file "%s"', path)
      labels.update(_read_label_annotations_file(path))

  logging.info('Read %d labels total.', len(labels))
  return labels


def create_labeled_examples(
    project: str,
    location: str,
    dataset_id: str,
    export_dir: str,
    examples_pattern: str,
    test_fraction: float,
    train_output_path: str,
    test_output_path: str) -> None:
  """Creates a labeled dataset by merging cloud labels and unlabeled examples.

  Args:
    project: Cloud project name.
    location: Dataset location, e.g. us-central1.
    dataset_id: Numeric id of the dataset to export.
    export_dir: GCS directory to export annotations to.
    examples_pattern: Pattern for unlabeled examples.
    test_fraction: Fraction of examples to write to test output.
    train_output_path: Path to training examples TFRecord output.
    test_output_path: Path to test examples TFRecord output.
  """

  labels = _get_labels(project, location, dataset_id, export_dir)
  _merge_examples_and_labels(examples_pattern, labels, test_fraction,
                             train_output_path, test_output_path)
