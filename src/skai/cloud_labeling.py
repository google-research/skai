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

import collections
import functools
import json
import multiprocessing
import os
import queue
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple, Set

from absl import logging

from google.cloud import aiplatform
from google.cloud import aiplatform_v1
import pandas as pd
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


def _read_example_ids_from_import_file(path: str) -> Iterable[str]:
  with tf.io.gfile.GFile(path, 'r') as import_file:
    for line in import_file:
      # Each line has the form gs://bucket/a/b/<example_id>.png.
      yield os.path.splitext(os.path.basename(line.strip()))[0]


def create_labeling_image(
    before_image: Image, after_image: Image, example_id: str, plus_code: str
) -> Image:
  """Creates an image used for labeling.

  The image is composed of the before and after images from the input example
  side-by-side.

  Args:
    before_image: Before image.
    after_image: After image.
    example_id: Example id.
    plus_code: Plus code.

  Returns:
    Annotated and combined image.

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
  caption = PIL.ImageDraw.Draw(combined)
  bottom_text = f'Example id: {example_id}   Plus code: {plus_code}'
  caption.text(
      (10, combined.height - 10),
      bottom_text,
      fill='black',
      anchor='lb',
  )
  return combined


def create_labeling_images(
    examples_pattern: str,
    max_images: int,
    allowed_example_ids_path: str,
    excluded_import_file_patterns: List[str],
    output_dir: str,
    use_multiprocessing: bool) -> Tuple[int, Optional[str]]:
  """Creates PNGs used for labeling from TFRecords.

  Also writes an import file in CSV format that is used to upload the images
  into the VertexAI labeling tool.

  Args:
    examples_pattern: File pattern for input TFRecords.
    max_images: Maximum number of images to create.
    allowed_example_ids_path: Path of file containing example ids that are
      allowed to be in the labeling set. The file should have one example id per
      line.
    excluded_import_file_patterns: List of import file patterns containing
      images to exclude.
    output_dir: Output directory.
    use_multiprocessing: If true, create multiple processes to create labeling
      images.

  Returns:
    Tuple of number of images written and file path for the import file.
  """
  example_files = tf.io.gfile.glob(examples_pattern)
  if not example_files:
    raise ValueError(
        f'Example pattern {examples_pattern} did not match any files.')

  excluded_example_ids = set()
  if excluded_import_file_patterns:
    for pattern in excluded_import_file_patterns:
      for path in tf.io.gfile.glob(pattern):
        logging.info('Excluding example ids from "%s"', path)
        excluded_example_ids.update(_read_example_ids_from_import_file(path))
    logging.info('Excluding %d example ids', len(excluded_example_ids))

  allowed_example_ids = None
  if allowed_example_ids_path:
    with tf.io.gfile.GFile(allowed_example_ids_path, 'r') as f:
      allowed_example_ids = set(line.strip() for line in f)
    logging.info('Allowing %d example ids', len(allowed_example_ids))

  if use_multiprocessing:
    image_paths_queue = multiprocessing.Manager().Queue(maxsize=max_images)
    num_workers = min(multiprocessing.cpu_count(), len(example_files))

    arg_list = []
    for example_file in example_files:
      arg_list.append((
          example_file,
          output_dir,
          allowed_example_ids,
          excluded_example_ids,
          image_paths_queue,
      ))

    with multiprocessing.Pool(num_workers) as pool_executor:
      _ = pool_executor.starmap(
          _create_labeling_images_from_example_file,
          arg_list,
      )
  else:
    image_paths_queue = queue.Queue(maxsize=max_images)
    for example_file in example_files:
      _create_labeling_images_from_example_file(
          example_file,
          output_dir,
          allowed_example_ids,
          excluded_example_ids,
          image_paths_queue,
      )

  if image_paths_queue.empty():
    return 0, None

  import_file_path = os.path.join(output_dir, 'import_file.csv')
  num_images = image_paths_queue.qsize()
  with tf.io.gfile.GFile(import_file_path, 'w') as f:
    while not image_paths_queue.empty():
      f.write(image_paths_queue.get() + '\n')

  return num_images, import_file_path


def _create_labeling_images_from_example_file(
    example_file: str,
    output_dir: str,
    allowed_example_ids: Set[str],
    excluded_example_ids: Set[str],
    image_paths_queue: queue.Queue[str],
) -> None:
  """Creates PNGs used for labeling from TFRecords for a single example_file.

  Also writes an import file in CSV format that is used to upload the images
  into the VertexAI labeling tool.

  Args:
    example_file: Path to file containing TF records.
    output_dir: Output directory.
    allowed_example_ids: Set of example_id from which a subset will be used in
      creating labeling task.
    excluded_example_ids: Set of example_id to be excluded.
    image_paths_queue: List of paths to images created for labeling task.
  """
  for record in tf.data.TFRecordDataset([example_file]):
    example = Example()
    example.ParseFromString(record.numpy())
    if 'example_id' in example.features.feature:
      example_id = (
          example.features.feature['example_id'].bytes_list.value[0].decode()
      )
    else:
      # If the example doesn't have an "example_id" feature, fall back on using
      # "encoded_coordinates". This maintains backwards compatibility with
      # older datasets.
      # TODO(jzxu): Remove this branch when backward compatibility is no longer
      # needed.
      example_id = (
          example.features.feature['encoded_coordinates'].bytes_list.value[0]
          .decode())

    if 'plus_code' in example.features.feature:
      plus_code = (
          example.features.feature['plus_code'].bytes_list.value[0].decode()
      )
    else:
      plus_code = 'unknown'

    if (allowed_example_ids is not None
        and example_id not in allowed_example_ids):
      continue

    if example_id in excluded_example_ids:
      logging.info('"%s" excluded', example_id)
      continue

    before_image = utils.deserialize_image(
        example.features.feature['pre_image_png_large'].bytes_list.value[0],
        'png')
    after_image = utils.deserialize_image(
        example.features.feature['post_image_png_large'].bytes_list.value[0],
        'png')
    labeling_image = create_labeling_image(
        before_image, after_image, example_id, plus_code)
    labeling_image_bytes = utils.serialize_image(labeling_image, 'png')
    path = os.path.join(output_dir, f'{example_id}.png')

    try:
      _ = image_paths_queue.put_nowait(path)
      with tf.io.gfile.GFile(path, 'w') as writer:
        writer.write(labeling_image_bytes)

    except queue.Full:
      break


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


def create_specialist_pool(
    project: str,
    location: str,
    display_name: str,
    manager_emails: List[str],
    worker_emails: List[str],
) -> str:
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


def _split_examples(
    examples: List[Example],
    test_fraction: float
) -> Tuple[List[Example], List[Example]]:
  """Splits a list of examples into training and test sets.

  Examples with the same encoded coordinates will always end up in the same
  split to prevent leaking information between training and test sets.

  Args:
    examples: Input examples.
    test_fraction: Fraction of examples to use for testing.

  Returns:
    Tuple of (training examples, test examples).
  """
  coordinates_to_examples = collections.defaultdict(list)
  for example in examples:
    c = example.features.feature['encoded_coordinates'].bytes_list.value[0]
    coordinates_to_examples[c].append(example)

  shuffled = random.sample(sorted(coordinates_to_examples.keys()),
                           len(coordinates_to_examples))
  num_test = int(len(shuffled) * test_fraction)
  test_examples = []
  for coordinate in shuffled[:num_test]:
    test_examples.extend(coordinates_to_examples[coordinate])

  train_examples = []
  for coordinate in shuffled[num_test:]:
    train_examples.extend(coordinates_to_examples[coordinate])
  return train_examples, test_examples


def _merge_single_example_file_and_labels(
    example_file: str, labels: Dict[str, List[Tuple[str, float, str]]]
) -> List[Example]:
  """Merges TF records from single example_file with corresponding labels.

  Args:
    example_file: Path to file containing TF records.
    labels: Dictionary of example id to a list of tuples
        (string label, numeric label, source dataset id).

  Returns:
    List of TF examples merged with labels for a single example_file.
  """
  labeled_examples = []
  for record in tf.data.TFRecordDataset([example_file]):
    example = Example()
    example.ParseFromString(record.numpy())
    if 'example_id' in example.features.feature:
      example_id = (
          example.features.feature['example_id'].bytes_list.value[0].decode()
      )
    else:
      # If the example doesn't have an "example_id" feature, fall back on
      # using "encoded_coordinates". This maintains backwards compatibility
      # with older datasets.
      # TODO(jzxu): Remove this branch when backward compatibility is no
      # longer needed.
      example_id = (
          example.features.feature['encoded_coordinates']
          .bytes_list.value[0]
          .decode()
      )

    label_tuples = labels.get(example_id, [])
    for string_label, numeric_label, dataset_id in label_tuples:
      labeled_example = Example()
      labeled_example.CopyFrom(example)
      features = labeled_example.features
      features.feature['string_label'].bytes_list.value[:] = [
          string_label.encode()
      ]
      features.feature['label_dataset_id'].bytes_list.value.append(
          dataset_id.encode())
      label_feature = features.feature['label'].float_list
      if not label_feature.value:
        label_feature.value.append(numeric_label)
      else:
        label_feature.value[0] = numeric_label
      labeled_examples.append(labeled_example)

  return labeled_examples


def _merge_examples_and_labels(
    examples_pattern: str,
    labels: Dict[str, List[Tuple[str, float, str]]],
    test_fraction: float,
    train_output_path: str,
    test_output_path: str,
    use_multiprocessing: bool,
) -> None:
  """Merges examples with labels into train and test TFRecords.

  Args:
    examples_pattern: File pattern for input examples.
    labels: Dictionary of example ids to a list of tuples
        (string label, numeric label, source dataset id).
    test_fraction: Fraction of examples to write to test output.
    train_output_path: Path to training examples TFRecord output.
    test_output_path: Path to test examples TFRecord output.
    use_multiprocessing: If true, create multiple processes to create labeled
      examples.
  """
  example_files = tf.io.gfile.glob(examples_pattern)

  if not example_files:
    raise ValueError(f'File pattern {examples_pattern} did not match anything')
  if not labels:
    raise ValueError(
        'Dictionary of labels is empty. Ensure that the dictionary of'
        ' labels is not empty'
    )

  if use_multiprocessing:
    num_workers = min(multiprocessing.cpu_count(), len(example_files))
    with multiprocessing.Pool(num_workers) as pool_executor:
      logging.info('Using multiprocessing with %d processes.', num_workers)
      results = pool_executor.map(
          functools.partial(
              _merge_single_example_file_and_labels, labels=labels
          ),
          example_files,
      )
  else:
    logging.info('Not using multiprocessing.')
    results = [
        _merge_single_example_file_and_labels(example_file, labels)
        for example_file in example_files
    ]

  all_labeled_examples = []
  for result in results:
    all_labeled_examples.extend(result)

  train_examples, test_examples = _split_examples(
      all_labeled_examples, test_fraction
  )

  _write_tfrecord(train_examples, train_output_path)
  _write_tfrecord(test_examples, test_output_path)


def _get_labels_from_dataset(
    project: str,
    location: str,
    dataset_id: str,
    export_dir: str) -> List[Tuple[str, str, str]]:
  """Reads labels from completed cloud labeling job.

  Args:
    project: Cloud project name.
    location: Dataset location, e.g. us-central1.
    dataset_id: Numeric id of the dataset to export.
    export_dir: GCS directory to export annotations to.

  Returns:
    List of (example id, string label) tuples.
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
    tf.io.gfile.remove(path)

  logging.info('Read %d labels total.', len(labels))
  return [
      (example_id, label, dataset_id) for example_id, label in labels.items()
  ]


def _read_label_file(path: str) -> List[Tuple[str, str, str]]:
  """Reads a label file.

  The file should be a CSV containing at least an "example_id" column
  and a "string_label" column. In the future example_ids will also be supported.

  Args:
    path: Path to file.

  Returns:
    List of (example id, string label) tuples.
  """
  with tf.io.gfile.GFile(path) as f:
    df = pd.read_csv(f)

  if 'example_id' not in df.columns:
    raise ValueError('Label file must contain "example_id" column.')
  if 'string_label' not in df.columns:
    raise ValueError('Label file must contain "string_label" column.')

  return [(row.example_id, row.string_label, path) for _, row in df.iterrows()]


def create_labeled_examples(
    project: str,
    location: str,
    dataset_ids: List[str],
    label_file_paths: List[str],
    string_to_numeric_labels: List[str],
    export_dir: str,
    examples_pattern: str,
    test_fraction: float,
    train_output_path: str,
    test_output_path: str,
    use_multiprocessing: bool) -> None:
  """Creates a labeled dataset by merging cloud labels and unlabeled examples.

  Args:
    project: Cloud project name.
    location: Dataset location, e.g. us-central1.
    dataset_ids: List of numeric dataset ids to export.
    label_file_paths: Paths to files to read labels from.
    string_to_numeric_labels: List of strings in the form
      "<string label>=<numeric label>", e.g. "no_damage=0"
    export_dir: GCS directory to export annotations to.
    examples_pattern: Pattern for unlabeled examples.
    test_fraction: Fraction of examples to write to test output.
    train_output_path: Path to training examples TFRecord output.
    test_output_path: Path to test examples TFRecord output.
    use_multiprocessing: If true, create multiple processes to create labeled
      examples.
  """
  string_to_numeric_map = {}
  for string_to_numeric_label in string_to_numeric_labels:
    if '=' not in string_to_numeric_label:
      raise ValueError(
          f'Invalid label mapping "{string_to_numeric_label}", should have '
          'form "label=0 or 1".')
    label, numeric_label = string_to_numeric_label.split('=')
    try:
      string_to_numeric_map[label] = float(numeric_label)
    except TypeError:
      logging.error('Class %s is not numeric.', numeric_label)
      raise

  labels = []
  for dataset_id in dataset_ids:
    labels.extend(
        _get_labels_from_dataset(project, location, dataset_id, export_dir)
    )

  for path in label_file_paths:
    labels.extend(_read_label_file(path))

  ids_to_labels = collections.defaultdict(list)
  for example_id, string_label, dataset_id in labels:
    example_labels = ids_to_labels[example_id]
    if string_label in [l[0] for l in example_labels]:
      # Don't add multiple labels with the same value for a single example.
      continue
    numeric_label = string_to_numeric_map.get(string_label, None)
    if numeric_label is None:
      raise ValueError(f'Label "{string_label}" has no numeric mapping.')
    example_labels.append((string_label, numeric_label, dataset_id))

  _merge_examples_and_labels(
      examples_pattern,
      ids_to_labels,
      test_fraction,
      train_output_path,
      test_output_path,
      use_multiprocessing,
  )
