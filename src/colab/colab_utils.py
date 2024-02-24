# Copyright 2023 Google LLC
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
"""Utility functions for skai colab notebook."""

import base64
import collections
import io
import json
import os
import re
import subprocess
import tempfile
import time

import ee
import folium
from folium.plugins import HeatMap
from google.cloud import monitoring_v3
import ipyplot
from IPython.display import display, HTML, Javascript
import pandas as pd
import pexpect
from PIL import Image, ImageDraw, ImageFont
import pyproj
import requests
import tensorflow as tf


def launch_pexpect_process(script,
                           arguments,
                           dir_args,
                           use_pexpect,
                           sleep=None):
  """Build and run the shell command."""
  if not isinstance(script, list):
    script = [script]
    arguments = [arguments]

  if sleep is None and len(script) > 1:
    sleep = [0] * (len(script) - 1)
  elif len(script) == 1:
    sleep = None

  flags_str = [
      ' '.join(f"--{f}='{v}'"
               for f, v in argument.items())
      for argument in arguments
  ]
  commands = '; '.join([
      'set -e', f'source {dir_args["python_env"]}',
      f'export GOOGLE_APPLICATION_CREDENTIALS={dir_args["path_cred"]}',
      f'export PYTHONPATH={dir_args["path_skai"]}/src',
      f'python {dir_args["path_skai"]}/src/{script[0]} {flags_str[0]}'
  ])

  if sleep is not None:
    commands_bis = ' '.join([
        '; '.join([
            f'& sleep {sleep[i-1]} ',
            f'python {dir_args["path_skai"]}/src/{script[i]} {flags_str[i]}'
        ]) for i in range(1, len(script))
    ])

    commands = ' '.join([commands, commands_bis])

  sh_command = f'bash -c "{commands}" | tee /tmp/output.txt'
  print(sh_command, '\n')

  if use_pexpect:
    return pexpect.spawn(sh_command)
  else:
    with open('/tmp/shell_command.sh', 'w') as f:
      f.write(commands)
    return subprocess.call(
        '/tmp/shell_command.sh | tee /tmp/output.txt', shell=True)


def make_gcp_http_request(url):
  """Run GET GCP API request using url provided."""
  token = subprocess.check_output(
      'gcloud auth print-access-token'.split()).decode().rstrip('.\r\n')
  response = requests.get(
      url=url, headers={"Authorization": "Bearer {token}".format(token=token)})
  if not response.ok:
    response.raise_for_status()
  return response.json()


def bucket_exists(project, bucket_name):
  """Check if bucket is alredy existing."""
  url = f'https://storage.googleapis.com/storage/v1/b?project={project}'
  data = make_gcp_http_request(url)
  buckets = [
      item['name'] for item in data['items'] if item['kind'] == 'storage#bucket'
  ]
  return (bucket_name in buckets)


def create_bucket(project, location, bucket_name):
  """Create bucket in given project."""
  os.system(
      f"""gsutil mb -p {project} -l {location} -b on gs://{bucket_name}""")


def get_project_id(project):
  """Return the project id for given project name."""
  url = 'https://cloudresourcemanager.googleapis.com/v1/projects/{}'.format(
      project)
  data = make_gcp_http_request(url)
  return int(data['projectNumber'])


def create_folium_map_with_images(pathgcp_before, pathgcp_after):
  """Display images before and after if there are all TIFF files."""
  # Load before image and get latitude/longitude of map center.
  # TODO(jzxu): Clean up this code and merge with function "create_folium_map".
  no_cog_file = ''
  error_message = []
  for image_path in pathgcp_before.split(',') + pathgcp_after.split(','):
    try:
      ee.Image.loadGeoTIFF(image_path).getMapId()
    except Exception as e:
      if re.search('The GeoTIFF is invalid or is not cloud optimized:', str(e)):
        no_cog_file = f'{no_cog_file}{image_path}\n'
        error_message.append(e)

  if error_message:
    print(
        'The following TIFF image(s) need to be Cloud Optimzed GeoTIFF file(s) '
        'in order to be visualized in the map using EarthEngine:'
        f'\n{no_cog_file}\n{str(error_message[0])}')
    return

  before_image_path = pathgcp_before.split(',')[0]
  before_map = ee.Image.loadGeoTIFF(before_image_path)
  x = before_map.getInfo()['bands'][0]['crs_transform'][2]
  y = before_map.getInfo()['bands'][0]['crs_transform'][-1]
  dim_x, dim_y = before_map.getInfo()['bands'][0]['dimensions']
  crs = before_map.getInfo()['bands'][0]['crs'].split(':')[-1]
  proj = pyproj.Transformer.from_crs(int(crs), 4326, always_xy=True)
  lon, lat = proj.transform(x + int(dim_x / 4), y - int(dim_y / 4))

  # Create a folium map object. Location is latitude, longitude.
  my_map = folium.Map(location=[lat, lon], zoom_start=12, max_zoom=25)

  for image_path in pathgcp_before.split(','):
    map_id_dict = ee.Image.loadGeoTIFF(image_path).getMapId()
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='COG',
        name=f'Pre-Disaster Imagery {os.path.basename(image_path)}',
        overlay=True,
        control=True,
        max_zoom=25,
    ).add_to(my_map)

  for image_path in pathgcp_after.split(','):
    map_id_dict = ee.Image.loadGeoTIFF(image_path).getMapId()
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='COG',
        name=f'Post-Disaster Imagery {os.path.basename(image_path)}',
        overlay=True,
        control=True,
        max_zoom=25,
    ).add_to(my_map)

  my_map.add_child(folium.LayerControl())
  display(my_map)


class DataflowMetricFetcher:
  """Read and parse log of generate_examples_main.py command to track progress
     of examples processed.
  """

  def __init__(self, project_id: str, job_name: str, metric_name: str):
    self._client = monitoring_v3.MetricServiceClient()
    self._project_id = project_id
    self._job_name = job_name
    self._metric_name = metric_name
    self._filter = self.make_filter()

  def make_filter(self):
    conditions = [
        'resource.type = "dataflow_job"',
        f'resource.labels.project_id = "{self._project_id}"',
        f'resource.labels.job_name = "{self._job_name}"',
        'metric.name = "dataflow.googleapis.com/job/user_counter"',
        f'metric.labels.metric_name = "{self._metric_name}"',
    ]
    return '({})'.format(' AND '.join(conditions))

  def get_latest_value(self):
    end_seconds = int(time.time())
    start_seconds = 1
    interval = monitoring_v3.TimeInterval({
        'start_time': {
            'seconds': start_seconds
        },
        'end_time': {
            'seconds': end_seconds
        }
    })
    request = {
        'name': f'projects/{self._project_id}',
        'filter': self._filter,
        'interval': interval,
        'view': monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
    }
    results = self._client.list_time_series(request)
    if (len(results._response.time_series) == 0 or
        len(results._response.time_series[0].points) == 0):
      return None, None

    latest_point = results._response.time_series[0].points[0]
    return latest_point.interval.end_time, latest_point.value.double_value


class ProgressBar:
  """Configuration of the progress bar for generate_examples_main.py command
     with progress tracking of examples processed.
  """

  def __init__(self, metrics, job_type=None, display_message=None):
    self._job_type = job_type
    self._display = display(
        self.get_html(metrics, display_message), display_id=True)

  def format_example_metrics(self, args):
    return 'Num generated examples: {value}/{max}'.format(**args)

  def format_training_metrics(self, args):
    return '''Metrics (updated as training progresses {timestamp}):<br>
    Labeled Training Set, Epoch {train_epoch}<br>
    Accuracy: {train_label_acc}% | AUC: {train_label_auc}<br>
    Test Set, Epoch {test_epoch}<br>
    Accuracy: {test_acc}% | AUC: {test_auc}'''.format(**args)

  def get_html(self, metrics, display_message=None):
    style = f'''<progress value="{metrics["value"]}" max="{metrics["max"]}"
    style="width: 100%">{metrics["value"]}</progress>'''

    if display_message is not None:
      formated_message = display_message
    else:
      if self._job_type == None:
        formated_message = ''
      elif self._job_type == 'example_progress':
        formated_message = self.format_example_metrics(metrics)
      elif self._job_type == 'training_progress':
        formated_message = self.format_training_metrics(metrics)

    return HTML(formated_message + style)

  def update(self, metrics, display_message=None):
    self._display.update(self.get_html(metrics, display_message))


def parse_dataflow_job_creation_params(param_str: str):
  """Parse dataflow job log to build dictionnary with job parameters."""
  params_dict = {}
  lines = [line.strip() for line in param_str.split('\r\n')]
  for line in lines:
    if not line:
      continue
    pieces = line.strip().split(':')
    key = pieces[0].strip()
    value = pieces[1].strip().strip("'")
    params_dict[key] = value
  return params_dict


def count_tfrecord(path):
  """Count number of examples contains in tfrecord file."""
  total_example_num = len(list(tf.data.TFRecordDataset(path)))
  return total_example_num


def run_example_generation(generate_examples_args,
                           path_dir_args,
                           pretty_output=True):
  """Run generate_examples_main.py command.

  If pretty_output is True, run the command with display of the progress bar.
  Otherwise, run the command with standard generated stdout output.
  """
  if not pretty_output:
    launch_pexpect_process(
        'generate_examples_main.py',
        generate_examples_args,
        path_dir_args,
        use_pexpect=False)
    return

  progress_bar = ProgressBar({'value': 0, 'max': 1}, 'example_progress')

  child = launch_pexpect_process(
      'generate_examples_main.py',
      generate_examples_args,
      path_dir_args,
      use_pexpect=True)

  JOB_CREATION_PATTERN = 'Create job: <Job(.*clientRequestId:.*)>'

  num_buildings = 1
  while child.isalive():
    i = child.expect(
        [JOB_CREATION_PATTERN, pexpect.EOF],
        timeout=3000)
    if i == 0:
      data = pd.read_parquet(
          f'{generate_examples_args["output_dir"]}/processed_buildings.parquet')
      num_buildings = data.shape[0]
      print(f'Found {num_buildings} buildings in area of interest.')
      progress_bar.update({'value': 0, 'max': num_buildings})
      job_params = parse_dataflow_job_creation_params(
          child.match.group(1).decode())
      job_name = job_params['name']
      job_id = job_params['id']
      job_location = job_params['location']
      job_project = job_params['projectId']
      job_status_pattern = f'Job {job_id} is in state JOB_STATE_([A-Z]+)'
      print(f'Your Dataflow job is :\n{job_name}')
      url = ('https://console.cloud.google.com/dataflow/jobs/'
             f'{job_location}/{job_id}?project={job_project}')
      print(f'Detailed monitoring page - Dataflow job id {job_id}: {url}')
      break
    else:
      print(child.before.decode())
      child.close()
      raise Exception('Job terminated unexpectedly.')

  generated_examples_metric = DataflowMetricFetcher(job_project, job_name,
                                                    'generated_examples_count')
  rejected_examples_metric = DataflowMetricFetcher(job_project, job_name,
                                                   'rejected_examples_count')

  job_state = None
  while child.isalive():
    i = child.expect([job_status_pattern, pexpect.TIMEOUT, pexpect.EOF],
                     timeout=15)
    if i == 0:
      job_state = child.match.group(1).decode()
      print(f'Dataflow job state: {job_state}')
      if job_state == 'DONE':
        progress_bar.update({'value': num_buildings, 'max': num_buildings})
        total_example_counter = 0
        for k in range(20):
          if 'labels_file' in generate_examples_args:
            file_directory = f'labeled/labeled-000{k:02d}-of-00020.tfrecord'
          else:
            file_directory = f'unlabeled/unlabeled-000{k:02d}-of-00020.tfrecord'

          tfrecord_path = os.path.join(
              os.path.join(generate_examples_args['output_dir'], 'examples'),
              file_directory)
          total_example_counter += count_tfrecord(tfrecord_path)
        print(
            f'{total_example_counter} building examples were extracted within '
            'the Area Of Interest and TIFF images')
    elif i == 1 or i == 2:
      if job_state == 'RUNNING':
        examples_processed = 0
        t, v = generated_examples_metric.get_latest_value()
        if t:
          examples_processed += int(v)
        t, v = rejected_examples_metric.get_latest_value()
        if t:
          examples_processed += int(v)
        progress_bar.update({'value': examples_processed, 'max': num_buildings})
      if i == 2:
        child.close()
        break


class LabelingJob:
  """Class for querying information on labeling task and dataset."""

  def __init__(self, endpoint, project, location, labeling_job):
    self._endpoint = endpoint
    self._project = project
    self._location = location
    self._labeling_job = labeling_job
    self._access_token = self.get_access_token()

    job_info = self.get_info()
    self._dataset = job_info['datasets'][0]

    assert len(job_info['specialistPools']) == 1
    # projects/{project_id}/locations/{location}/specialistPools/{pool_id}
    parts = job_info['specialistPools'][0].split('/')
    assert len(parts) == 6
    assert parts[4] == 'specialistPools'
    self._pool_id = parts[5]

  def get_access_token(self):
    return subprocess.check_output(
        'gcloud auth print-access-token'.split()).decode().rstrip('.\r\n')

  def get_header(self):
    return {
        'Authorization': f'Bearer {self._access_token}',
        'Content-Type': 'application/json',
    }

  def get_info(self):
    """Return the percentage of data items labeled.

    Warning: There is a long lag between when items are labeled and when this
    value is updated.
    """
    parent = (f'projects/{self._project}/locations/{self._location}/'
              f'dataLabelingJobs/{self._labeling_job}')
    url = f'https://{self._endpoint}/v1/{parent}'
    header = self.get_header()
    r = requests.get(url, headers=header)
    if not r.ok:
      r.raise_for_status()
    return r.json()

  def get_completion_percentage(self):
    """Return the percentage of data items labeled.

    Warning: There is a long lag between when items are labeled and when this
    value is updated.
    """
    info = self.get_info()
    return info.get('labelingProgress', 0)

  def get_data_items(self):
    parent = (f'projects/{self._project}/locations/{self._location}/datasets/'
              f'{self._dataset}/dataItems')
    url = f'https://{self._endpoint}/v1/{parent}'
    items = []
    page_token = None
    header = self.get_header()
    while True:
      if page_token:
        r = requests.get(url, headers=header, params={'pageToken': page_token})
      else:
        r = requests.get(url, headers=header)
      if not r.ok:
        r.raise_for_status()

      result_json = r.json()
      items.extend(result_json['dataItems'])
      if 'nextPageToken' in result_json:
        page_token = result_json['nextPageToken']
      else:
        break
    return items

  def get_labels(self, data_item_name):
    url = f'https://{self._endpoint}/v1/{data_item_name}/annotations'
    header = self.get_header()
    response = requests.get(url, headers=header)
    if not response.ok:
      response.raise_for_status()
    response_json = response.json()
    labels = []
    if 'annotations' in response_json:
      for a in response_json['annotations']:
        labels.append(a['payload']['displayName'])
    return labels

  def get_worker_url(self):
    """Returns the URL workers can use to access the labeling interface.

    The syntax of the URL was determined by reverse engineering, so there's no
    guarantee that it won't change in the future.
    """
    location = self._location.replace('-', '_')
    return ('https://datacompute.google.com/w/'
            f'cloudml_data_specialists_{location}_{self._pool_id}')

  def get_manager_url(self):
    """Returns the URL managers can use to access the task management interface.

    The syntax of the URL was determined by reverse engineering, so there's no
    guarantee that it won't change in the future.
    """
    location = self._location.replace('-', '_')
    return ('https://datacompute.google.com/cm/'
            f'cloudml_data_specialists_{location}_{self._pool_id}/tasks')


def run_labeling_task_creation(create_label_task_args,
                               path_dir_args,
                               pretty_output=True):
  """Run create_cloud_labeling_task.py command.

  If pretty_output is True, run the command a reduced stdout output and return
  of processed information about task and dataset created
  (dataset_id, dataset_name, labelingjob_id, labelingjob_instruction).
  Otherwise, run the command with standard generated stdout output and return
  None.
  """
  if not pretty_output:
    launch_pexpect_process('create_cloud_labeling_task.py',
                           create_label_task_args, path_dir_args, False)
    return None

  child = launch_pexpect_process('create_cloud_labeling_task.py',
                                 create_label_task_args, path_dir_args, True)

  dataset_created_pattern = ('] ImageDataset created. Resource name: '
                             'projects/[^/]+/locations/[^/]+/datasets/([0-9]+)')
  labeling_job_created_pattern = '] Data labeling job created:'

  location = create_label_task_args['cloud_location']
  project = create_label_task_args['cloud_project']

  output = b''
  while child.isalive():
    i = child.expect(
        [
            dataset_created_pattern,
            labeling_job_created_pattern,
            pexpect.EOF,
            pexpect.TIMEOUT,
        ],
        timeout=1800,
    )

    if isinstance(child.before, bytes):
      output += child.before
    if isinstance(child.after, bytes):
      output += child.after

    if i == 0:
      dataset_id = child.match.group(1).decode()
      print('Creating data labeling job...')
    elif i == 1:
      print('Data labeling job created.')

      url = (f'https://{location}-aiplatform.googleapis.com/v1/projects/'
             f'{project}/locations/{location}/dataLabelingJobs')
      data = make_gcp_http_request(url)
      data = list(
          filter(
              lambda d: create_label_task_args['dataset_name'] in d[
                  'displayName'],
              data['dataLabelingJobs'],
          ))[0]

      dataset_id = int(data['datasets'][0].split('/')[-1])
      dataset_name = data['displayName']
      labelingjob_id = int(data['name'].split('/')[-1])
      labelingjob_instruction = data['instructionUri']

      print(f'\nLabeling dataset {dataset_name} created, with ID {dataset_id}')
      print(f'\nData Labeling job {labelingjob_id} created')

      labeling_job = LabelingJob(
          f'{location}-aiplatform.googleapis.com',
          project,
          location,
          labelingjob_id,
      )

      print('Instruction URL: {}'.format(
          labelingjob_instruction.replace('gs://',
                                          'https://storage.cloud.google.com/')))
      print(f'Worker URL: {labeling_job.get_worker_url()}')
      print(f'Manager URL: {labeling_job.get_manager_url()}')
      print('Detailed monitoring page: '
            'https://console.cloud.google.com/vertex-ai/locations/'
            f'{location}/labeling-tasks/{labelingjob_id}?project={project}')

      break
    elif i == 2:
      print('An unexpected error occurred. Output of command was:')
      print(child.before.decode())
      child.close()
      raise Exception('Job terminated unexpectedly.')
    else:
      child.close()
      raise Exception('Job timed out. Full output:\n' + output.decode())

  child.close()
  return dataset_id, dataset_name, labelingjob_id, labelingjob_instruction


def create_labeled_dataset(create_labeled_dataset_args, path_dir_args):
  """Run create_labeled_dataset.py command."""
  child = launch_pexpect_process('create_labeled_dataset.py',
                                 create_labeled_dataset_args, path_dir_args,
                                 True)
  print('Labeling dataset used for labeled datasets creation :'
        f'{create_labeled_dataset_args["cloud_dataset_ids"]}')
  print('Creating labeled datasets...')
  child.expect(pexpect.EOF, timeout=None)
  child.close()
  if child.exitstatus != 0:
    print('An unexpected error occurred. Output of command was:')
    print(child.before.decode())
  else:
    print('Labeled dataset created.')


def concat_caption_pilimage(image_before, image_after):
  """Concatenate the before and after images for visualisation."""
  img_before = caption_pilformat(image_before, 'before')
  img_after = caption_pilformat(image_after, 'after')

  w, h = img_before.size

  img_concat = Image.new('RGB', (2 * w, h), 'white')
  img_concat.paste(img_before, (0, 0))
  img_concat.paste(img_after, (w, 0))

  return img_concat


def caption_pilformat(img_data, caption):
  """Create the caption text on the images."""
  base64_encoded = base64.b64encode(img_data)
  im_bytes = base64.b64decode(base64_encoded)
  byte_encoded = io.BytesIO(im_bytes)

  img = Image.open(byte_encoded)
  wd, hg = img.size

  img_ = Image.new('RGB', (wd + int(wd / 10), hg + int(hg / 5)), 'white')
  img_.paste(img, (int(wd / 20), int(hg / 5)))

  wd, hg = img_.size
  img_cap = ImageDraw.Draw(img_)
  _, _, w, _ = img_cap.textbbox((0, 0), caption)
  img_cap.text(((wd - w) / 2, 0), caption, fill=(0, 0, 0))

  return img_


def visualize_labeled_examples(path, max_examples=None):
  """Visualise the example labeled by the operators using ipyplot package."""
  pre_images = []
  post_images = []
  labels = []
  labels_split = []
  total_example_num = len(list(tf.data.TFRecordDataset(path)))
  print('Number of examples: {}.'.format(total_example_num))

  if max_examples is None:
    max_examples = total_example_num

  for record in tf.data.TFRecordDataset(path):
    e = tf.train.Example()
    e.ParseFromString(record.numpy())
    labels_split.append(e.features.feature['label'].float_list.value[0])
    if len(pre_images) < max_examples:
      pre_images.append(e.features.feature['pre_image_png'].bytes_list.value[0])
      post_images.append(
          e.features.feature['post_image_png'].bytes_list.value[0])
      labels.append(e.features.feature['label'].float_list.value[0])

  labels_counter = dict(collections.Counter(labels_split))
  map_value = {
      0:
          'Undamaged/bad examples {}/{}'.format(
              int(len(labels) - sum(labels)), labels_counter[0]),
      1:
          'Damaged {}/{}'.format(int(sum(labels)), labels_counter[1])
  }
  labels = list((pd.Series(labels)).map(map_value))

  images = [
      concat_caption_pilimage(pre_images[idx], post_images[idx])
      for idx in range(len(pre_images))
  ]

  ipyplot.plot_class_tabs(
      images,
      labels,
      max_imgs_per_tab=max_examples,
      tabs_order=[map_value[1], map_value[0]],
      img_width=200)

  return total_example_num


def timestamp_to_datetime(timestamp):
  return pd.to_datetime(timestamp)


def run_train_and_eval_job(run_train_eval_args,
                           path_dir_args,
                           email_manager,
                           sleep=None,
                           pretty_output=True,
                           load_tensorboard=False,
                           path_log_tensorboard=None):
  """Run the shell launch_vertex_job.py command for training and evaluation job.

  If load_tensorboard=True and path_log_tensorboard is provided,
  tensorboard is display to track the progress of the training job and
  performance metrics.
  """

  if not pretty_output:
    launch_pexpect_process(['launch_vertex_job.py', 'launch_vertex_job.py'],
                           run_train_eval_args,
                           path_dir_args,
                           use_pexpect=False,
                           sleep=sleep)
    return

  # Create the progress bar and metrics displays.
  progress_bar = ProgressBar({'value': 0, 'max': 100})
  metrics_display = False

  # Store Job IDs of training and evaluation jobs.
  # Keep track of timestamp of most recent logs to process only fresher logs.
  train_job_id, eval_job_id = None, None
  curr_epoch = None
  total_num_epochs = None
  total_num_img = None
  progress_args = None
  train_most_recent_timestamp = pd.Timestamp.utcnow()
  eval_most_recent_timestamp = pd.Timestamp.utcnow()

  train_state, eval_state = None, None

  # Run the child program.
  child = launch_pexpect_process(
      ['launch_vertex_job.py', 'launch_vertex_job.py'],
      run_train_eval_args,
      path_dir_args,
      use_pexpect=True,
      sleep=sleep)
  while not child.closed:
    # Expects 5 different patterns, or EOF (meaning the program terminated).
    # Each pattern is a regex and you can use regex match groups "()" to extract
    # a part of the matched text for later use.
    pattern_idx = child.expect(
        [
            'I.*] Creating CustomJob',
            'I.*] CustomJob created\\. Resource name: .*/([0-9]*)',
            'I.*] View Custom Job:\r\n(.*)\r\nCustomJob',
            ('I.*] CustomJob .*/([0-9]*) current'
             ' state:\r\nJobState.JOB_STATE_PENDING'),
            ('I.*] CustomJob .*/([0-9]*) current'
             ' state:\r\nJobState.JOB_STATE_RUNNING'),
            'I.*] CustomJob run completed.',
            pexpect.EOF,
        ],
        timeout=None,
    )
    if pattern_idx == 0:
      progress_bar.update({'value': 0, 'max': 100})
    elif pattern_idx == 1:  # A job was created, so store its ID.
      job_id = child.match.group(1).decode()
      if train_job_id is None:
        train_job_id = job_id
        progress_bar.update({'value': 1, 'max': 100})
        print(
            f'Checkpoints will be saved here :\n{run_train_eval_args[0]["train_dir"]}'
        )
        print(
            f'Your Training CustomJob is :\n{run_train_eval_args[0]["display_name"]}'
        )
      elif eval_job_id is None:
        if job_id != train_job_id:
          eval_job_id = job_id
          print(
              f'Your Evaluation CustomJob is :\n{run_train_eval_args[1]["display_name"]}'
          )

    elif pattern_idx == 2:  # Jobs are created.
      if job_id == train_job_id:
        print('Training CustomJob created\nDetailed monitoring page - '
              f'Train job id {job_id} : {child.match.group(1).decode()}')
      else:
        print('Evaluation CustomJob created\nDetailed monitoring page - '
              f'Eval job id {job_id} : {child.match.group(1).decode()}')

    elif pattern_idx == 3:  # Jobs are pending, so update progress bar.
      job_id = child.match.group(1).decode()
      if job_id == train_job_id:
        progress_args = {'value': 5, 'max': 100}
        progress_bar.update(progress_args)
        if train_state is None:
          train_state = 'PENDING'
          print(f'Training CustomJob state: {train_state}')
      else:
        if eval_state is None:
          eval_state = 'PENDING'
          print(f'Evaluation CustomJob state: {eval_state}')

    elif pattern_idx == 4:  # Jobs are running, update progress bar or metrics.
      job_id = child.match.group(1).decode()
      if job_id == train_job_id:
        if train_state == 'PENDING':
          train_state = 'RUNNING'
          print(f'Training CustomJob state: {train_state}')
      else:
        if eval_state == 'PENDING':
          eval_state = 'RUNNING'
          print(f'Evaluation CustomJob state: {eval_state}\n')

      log_data = _download_eval_job_log(job_id)
      if log_data:
        if job_id == train_job_id:
          # If training job, then update the progress bar.
          curr_epoch = None
          for log in log_data:
            log_timestamp = timestamp_to_datetime(log['timestamp'])
            if log_timestamp < train_most_recent_timestamp:
              # If logs have not been refreshed, ignore them.
              break
            else:
              train_most_recent_timestamp = log_timestamp
            if log_timestamp == train_most_recent_timestamp:
              matches = re.search(
                  'Epoch ([0-9]*/[0-9]*):   [0-9]*%.* [0-9]*/([0-9]*)',
                  log['jsonPayload']['message'])
              if matches:
                matches = matches.groups()
                log_epoch = int(matches[0].split('/')[0])
                if total_num_epochs is None:
                  total_num_epochs = int(matches[0].split('/')[1])
                if total_num_img is None:
                  total_num_img = int(matches[-1])
                if curr_epoch is None or log_epoch > curr_epoch:
                  # Logs can be received at different times, so check for the
                  # highest epoch number logged.
                  curr_epoch = log_epoch
                  progress_args.update({
                      'value':
                          5 + int(95. * curr_epoch / (total_num_epochs + 1))
                  })
                  progress_bar.update(progress_args)
                  break
        else:
          # If evaluation job, then update the metrics display.
          train_label_acc = None
          train_label_auc = None
          test_acc = None
          test_auc = None
          for log in log_data:
            log_timestamp = timestamp_to_datetime(log['timestamp'])
            if log_timestamp < eval_most_recent_timestamp:
              # If logs have not been refreshed, ignore them.
              break
            if train_label_acc is None:
              train_label_matches = re.search(
                  r'Epoch: ([0-9]*), .* Train_Label AUC: ([0-9]*\.[0-9]*), ' +
                  r'Train_Label Accuracy: ([0-9]*\.[0-9]*)',
                  log['jsonPayload']['message'])
              if train_label_matches:
                train_label_epoch = min(
                    (int(train_label_matches.groups()[0]) // total_num_img) + 1,
                    total_num_epochs)
                train_label_epoch = f'{train_label_epoch}/{total_num_epochs}'
                train_label_auc = train_label_matches.groups()[1]
                train_label_acc = train_label_matches.groups()[2]
            elif test_acc is None:
              test_matches = re.search(
                  r'Epoch: ([0-9]*), .* Test AUC: ([0-9]*\.[0-9]*), '
                  r'Test Accuracy: ([0-9]*\.[0-9]*)',
                  log['jsonPayload']['message'])
              if test_matches:
                test_epoch = min(
                    (int(test_matches.groups()[0]) // total_num_img) + 1,
                    total_num_epochs)
                test_epoch = f'{test_epoch}/{total_num_epochs}'
                test_auc = test_matches.groups()[1]
                test_acc = test_matches.groups()[2]
            else:
              eval_most_recent_timestamp = log_timestamp
              break
          if train_label_acc is not None and test_acc is not None:
            if not metrics_display:
              progress_bar._job_type = 'training_progress'
              progress_args.update({
                  'timestamp':
                      eval_most_recent_timestamp.strftime('%Y-%m-%d, %H:%M:%S'),
                  'train_epoch':
                      0,
                  'train_label_acc':
                      0,
                  'train_label_auc':
                      0,
                  'test_epoch':
                      0,
                  'test_acc':
                      0,
                  'test_auc':
                      0
              })
              progress_bar.update(progress_args)
              metrics_display = True
              if load_tensorboard:
                load_start_tensorboard(path_log_tensorboard)
            else:
              progress_args.update({
                  'timestamp':
                      eval_most_recent_timestamp.strftime('%Y-%m-%d, %H:%M:%S'),
                  'train_epoch':
                      train_label_epoch,
                  'train_label_acc':
                      train_label_acc,
                  'train_label_auc':
                      train_label_auc,
                  'test_epoch':
                      test_epoch,
                  'test_acc':
                      test_acc,
                  'test_auc':
                      test_auc
              })
              progress_bar.update(progress_args)

    elif pattern_idx == 5:  # Job completed. Email user a notification.
      print('Training CustomJob state: DONE\n')
      email_subject = 'Skai Training Complete'
      email_body = ('Training has completed! Please return to the Colab to '
                    'visualize results.')
      os.system(f'printf "Subject: {email_subject}\n\n{email_body}" '
                f'| msmtp {email_manager}')
      progress_args.update({'value': 100})
      progress_bar.update(progress_args)
    else:
      child.close()


def _download_eval_job_log(eval_job_id):
  with tempfile.NamedTemporaryFile() as output:
    log_filter = (
        f'"resource.labels.job_id={eval_job_id} severity=ERROR \\"Epoch\\""')
    command = f'gcloud logging read {log_filter} --format json > {output.name}'
    if os.system(command) == 0:
      return json.load(output)
    return None


def get_train_eval_job_id(project, location, job_name):
  """Return the job id of the train or evaluation job name."""
  url = 'https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/customJobs'.format(
      location, project, location)
  data = make_gcp_http_request(url)
  data = list(
      filter(lambda d: job_name in d['displayName'], data['customJobs']))[0]
  return int(data['name'].split('/')[-1])


def get_epoch_number(path_experiment, id_eval_job, checkpoint_selection,
                     checkpoint_index):
  """Return the epoch number of choosen method or a specific checkpoint."""
  if checkpoint_selection == 'most_recent':
    most_recent_epoch_file = os.path.join(f'gs://{path_experiment}',
                                          'checkpoints', 'last_processed_epoch')
    os.system(f'gsutil cp {most_recent_epoch_file} /tmp/last_processed_epoch')
    with open('/tmp/last_processed_epoch', 'r') as epoch_f:
      epoch_num = epoch_f.read()

  elif checkpoint_selection in ['top_auc_test', 'top_acc_test']:
    epoch_num_acc = None
    metrics_acc = None
    epoch_num_auc = None
    metrics_auc = None
    log_data = _download_eval_job_log(id_eval_job)
    if log_data:
      for log in log_data:
        test_matches = re.search(
            r'Epoch: ([0-9]*),.*Test AUC: ([0-9]*\.[0-9]*), '
            r'Test Accuracy: ([0-9]*\.[0-9]*)', log['jsonPayload']['message'])
        if test_matches:

          if epoch_num_acc is None:
            epoch_num_acc = test_matches.groups()[0]
            metrics_acc = float(test_matches.groups()[2])
          elif (int(test_matches.groups()[0]) > int(epoch_num_acc) and
                float(test_matches.groups()[2]) == metrics_acc) or (float(
                    test_matches.groups()[2]) > metrics_acc):
            epoch_num_acc = test_matches.groups()[0]
            metrics_acc = float(test_matches.groups()[2])

          if epoch_num_auc is None:
            epoch_num_auc = test_matches.groups()[0]
            metrics_auc = float(test_matches.groups()[1])
          elif (int(test_matches.groups()[0]) > int(epoch_num_auc) and
                float(test_matches.groups()[1]) == metrics_auc) or (float(
                    test_matches.groups()[1]) > metrics_auc):
            epoch_num_auc = test_matches.groups()[0]
            metrics_auc = float(test_matches.groups()[1])

      if checkpoint_selection == 'top_auc_test':
        epoch_num = epoch_num_auc
      elif checkpoint_selection == 'top_acc_test':
        epoch_num = epoch_num_acc

  elif checkpoint_selection == 'index_number':
    epoch_num = str(int(checkpoint_index))

  metrics_acc = None
  metrics_auc = None
  log_data = _download_eval_job_log(id_eval_job)
  if log_data:
    for log in log_data:
      test_matches = re.search(
          r'Epoch: ([0-9]*),.*Test AUC: ([0-9]*\.[0-9]*), '
          r'Test Accuracy: ([0-9]*\.[0-9]*)', log['jsonPayload']['message'])
      if test_matches:
        if int(test_matches.groups()[0]) == int(epoch_num):
          metrics_acc = test_matches.groups()[2]
          metrics_auc = test_matches.groups()[1]
          break

  if checkpoint_selection == "top_auc_test":
    print(f'Checkpoint used: {epoch_num}, '
          f'Test AUC (best): {metrics_auc}, '
          f'Test Accuracy: {metrics_acc}')
  elif checkpoint_selection == "top_acc_test":
    print(f'Checkpoint used: {epoch_num}, '
          f'Test AUC: {metrics_auc}, '
          f'Test Accuracy (best): {metrics_acc}')
  else:
    print(f'Checkpoint used: {epoch_num}, '
          f'Test AUC: {metrics_auc}, '
          f'Test Accuracy: {metrics_acc}')

  return epoch_num.zfill(8)


def run_inference_and_prediction_job(run_infer_args,
                                     path_dir_args,
                                     epoch,
                                     pretty_output=True):
  """Run the shell launch_vertex_job.py command for inference job."""

  if not pretty_output:
    launch_pexpect_process(
        'launch_vertex_job.py',
        run_infer_args,
        path_dir_args,
        use_pexpect=False)
    return

  # Initialize progress bar.
  progress_bar = ProgressBar({'value': 0, 'max': 100})
  curr_idx = 0
  job_state = None

  # Run the child program.
  child = launch_pexpect_process(
      'launch_vertex_job.py', run_infer_args, path_dir_args, use_pexpect=True)

  while not child.closed:
    # Expects 5 different patterns, or EOF (meaning the program terminated).
    # Each pattern is a regex and you can use regex match groups "()" to extract
    # a part of the matched text for later use.
    pattern_idx = child.expect([
        'I.*] CustomJob created\\.',
        ('I.*] CustomJob .*/([0-9]*) current'
         ' state:\r\nJobState.JOB_STATE_PENDING'),
        ('I.*] CustomJob .*/([0-9]*) current'
         ' state:\r\nJobState.JOB_STATE_RUNNING'),
        ('I.*] CustomJob .*/([0-9]*) current'
         ' state:\r\nJobState.JOB_STATE_SUCCEEDED'),
        'I.*] CustomJob run completed.', pexpect.EOF
    ],
                               timeout=None)

    if pattern_idx == 0:  # A job was created.
      progress_bar.update({'value': 5, 'max': 100})
      print('Inference CustomJob created')
      print('Checkpoint used for inference :\n'
            f'{run_infer_args["eval_ckpt"]}.data-00000-of-00001')
    elif pattern_idx == 1:  # Job Pending.
      if job_state != 'PENDING':
        job_state = 'PENDING'
        print(f'Inference CustomJob state: {job_state}')
      progress_bar.update({'value': 10, 'max': 100})
    elif pattern_idx == 2:  # Job Running.
      if job_state != 'RUNNING':
        job_state = 'RUNNING'
        print(f'Inference CustomJob state: {job_state}')
      starting_progress = 20
      max_progress = 90
      curr_progress = starting_progress + (curr_idx * 2)
      if curr_idx == 0:
        progress_bar.update({'value': starting_progress, 'max': 100})
      elif curr_progress < max_progress:
        # Update while job is running only until progress hits 90.
        progress_bar.update({'value': curr_progress, 'max': 100})
      curr_idx += 1
    elif pattern_idx == 3:  # Job Succeeded.
      progress_bar.update({'value': 99, 'max': 100})
    elif pattern_idx == 4:  # Job Completed.
      if job_state != 'DONE':
        job_state = 'DONE'
        print(f'Inference CustomJob state: {job_state}')
      progress_bar.update({'value': 100, 'max': 100})
      preds_file = os.path.join(f'{run_infer_args["train_dir"]}', 'predictions',
                                f'test_ckpt_{int(epoch)}.geojson')
      os.system(f'gsutil cp {preds_file} /tmp/predictions.geojson')
      print(f'Predictions saved in :\n{preds_file}')
    else:
      child.close()


def create_folium_map(geojson_path, pathgcp_before, pathgcp_after):
  """Creates an interactive Folium map.

  Displays images before and after if they are all TIFF files, an predictions on
  two layers: one per building with circle marker, and one as a heat map of
  damaged building.
  """
  basemaps = {
      'Google Maps':
          folium.TileLayer(
              tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
              attr='Google',
              name='Google Maps',
              overlay=True,
              control=True),
      'Google Satellite':
          folium.TileLayer(
              tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
              attr='Google',
              name='Google Satellite',
              overlay=True,
              control=True),
      'Google Satellite Hybrid':
          folium.TileLayer(
              tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
              attr='Google',
              name='Google Satellite',
              overlay=True,
              control=True)
  }

  with open(geojson_path, 'r') as f:
    predictions = json.load(f)

  damaged_preds = {'type': predictions['type'], 'features': []}
  undamaged_preds = {'type': predictions['type'], 'features': []}

  # Count number of buildings per class.
  num_damaged_buildings = 0
  num_undamaged_buildings = 0
  for feat in predictions['features']:
    if feat['properties']['class_1'] >= 0.5:
      num_damaged_buildings += 1
      damaged_preds['features'].append(feat)
    else:
      num_undamaged_buildings += 1
      undamaged_preds['features'].append(feat)

  lat = predictions['features'][0]['properties']['latitude']
  lon = predictions['features'][0]['properties']['longitude']

  # Create a folium map object. Location is latitude, longitude.
  my_map = folium.Map(location=[lat, lon], zoom_start=16, max_zoom=20)

  # Add custom basemaps.
  basemaps['Google Maps'].add_to(my_map)
  basemaps['Google Satellite Hybrid'].add_to(my_map)

  no_cog_file = ''
  error_message = []
  for image_path in pathgcp_before.split(',') + pathgcp_after.split(','):
    try:
      ee.Image.loadGeoTIFF(image_path).getMapId()
    except Exception as e:
      if re.search('The GeoTIFF is invalid or is not cloud optimized:', str(e)):
        no_cog_file = f'{no_cog_file}{image_path}\n'
        error_message.append(e)

  if error_message:
    print(
        'The following TIFF image(s) need to be Cloud Optimzed GeoTIFF file(s) '
        'in order to be visualized in the map using EarthEngine:\n'
        f'{no_cog_file}')

  else:
    for image_path in pathgcp_before.split(','):
      map_id_dict = ee.Image.loadGeoTIFF(image_path).getMapId()
      folium.raster_layers.TileLayer(
          tiles=map_id_dict['tile_fetcher'].url_format,
          attr='COG',
          name=f'Pre-Disaster Imagery {os.path.basename(image_path)}',
          overlay=True,
          control=True,
          max_zoom=20,
      ).add_to(my_map)

    for image_path in pathgcp_after.split(','):
      map_id_dict = ee.Image.loadGeoTIFF(image_path).getMapId()
      folium.raster_layers.TileLayer(
          tiles=map_id_dict['tile_fetcher'].url_format,
          attr='COG',
          name=f'Post-Disaster Imagery {os.path.basename(image_path)}',
          overlay=True,
          control=True,
          max_zoom=20,
      ).add_to(my_map)

  # Add predictions.
  # Set style parameters.
  style_function = lambda x: {
      'radius':
          3,
      'weight':
          1,
      'fill':
          True,
      'color':
          '#ff0000' if float(x['properties']['class_1']) >= 0.5 else '#00ff00',
      'fillColor':
          '#ff0000' if float(x['properties']['class_1']) >= 0.5 else '#00ff00',
      'fillOpacity':
          0.3,
  }

  marker_function = lambda x: folium.CircleMarker(
      [x['properties']['latitude'], x['properties']['longitude']])

  folium.features.GeoJson(
      damaged_preds,
      name='Damaged Predictions',
      style_function=style_function,
      marker=folium.CircleMarker(),
  ).add_to(my_map)
  folium.features.GeoJson(
      undamaged_preds,
      name='Undamaged Predictions',
      style_function=style_function,
      show=False,
      marker=folium.CircleMarker(),
  ).add_to(my_map)

  data_heat = [[
      x['properties']['latitude'], x['properties']['longitude'],
      (x['properties']['class_1'] >= 0.5)
  ] for x in predictions['features']]

  HeatMap(
      data_heat,
      name='Heatmap Predictions',
      gradient={
          0.9: 'grey',
          0.95: 'yellow',
          0.99: 'orange',
          0.995: 'red',
          1: 'darkred'
      }).add_to(my_map)

  my_map.add_child(folium.LayerControl())

  print('Number of Damaged Buildings: ', num_damaged_buildings)
  print('Number of Undamaged Buildings: ', num_undamaged_buildings)
  print('Total: ', int(num_undamaged_buildings) + int(num_damaged_buildings))
  display(my_map)
