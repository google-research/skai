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

"""Utility functions for skai colab notebook."""

import base64
import collections
import io
import json
import os
import re
import subprocess
import time

import ee
import folium
from folium.plugins import HeatMap
import ipyplot
from IPython.display import display, HTML, Javascript
import pandas as pd
import pexpect
import pyproj
import requests
import tensorflow as tf

from google.cloud import monitoring_v3
from PIL import Image, ImageDraw, ImageFont

def make_gcp_http_request(url):
  token=subprocess.check_output('gcloud auth print-access-token'.split()).decode().rstrip('.\r\n')
  response = requests.get(url=url, headers = {"Authorization": "Bearer {token}".format(token=token)})
  if not response.ok:
    response.raise_for_status()
  return response.json()

def bucket_exists(project, bucket_name):
  url = f'https://storage.googleapis.com/storage/v1/b?project={project}'
  data = make_gcp_http_request(url)
  buckets = [item['name'] for item in data['items'] if item['kind'] == 'storage#bucket']
  return (bucket_name in buckets)

def create_bucket(project, location, bucket_name):
  os.system(f"""gsutil mb -p {project} -l {location} -b on gs://{bucket_name}""")

def progress(value, max=100):
  css = """
        <style>
          progress {
            border-radius: 7px;
            box-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5) inset;
            width: 80%;
            height: 30px;
            display: block;
          }
          progress::-webkit-progress-bar {
            background-color: rgba(237, 237, 237, 0);
            border-radius: 7px;
          }
          progress::-webkit-progress-value {
            background-color: green;
            border-radius: 7px;
            box-shadow: 1px 1px 1px rgba(0, 0, 0, 0.1) inset;
          }
        </style>
        """
  html = """
          <progress
              value='{value}'
              max='{max}'
          >
            {value}%
          </progress>
        """.format(value=value, max=max)
  return HTML(css + html)


# Add custom basemaps to folium.

basemaps = {
    'Google Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = True,
        control = True
    ),
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Google Terrain': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Terrain',
        overlay = True,
        control = True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
    )
}

def create_folium_map_with_images(pathgcp_before, pathgcp_after):
  # Load before image and get latitude/longitude of map center.

  no_cog_file=''
  error_message=[]
  for image_path in pathgcp_before.split(',')+pathgcp_after.split(','):
    try:
      ee.Image.loadGeoTIFF(image_path).getMapId()
    except Exception as e:
      if re.search('The GeoTIFF is invalid or is not cloud optimized:',str(e)):
        no_cog_file=f'{no_cog_file}{image_path}\n'
        error_message.append(e)

  if error_message:
    print(f'The following TIFF image(s) need to be Cloud Optimzed GeoTIFF file(s) in order to be visualized in the map using EarthEngine:\n{no_cog_file}\n{str(error_message[0])}')
    return

  before_image_path = pathgcp_before.split(',')[0]
  before_map = ee.Image.loadGeoTIFF(before_image_path)
  before_map_id_dict = before_map.getMapId()
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
      name = f'Pre-Disaster Imagery {os.path.basename(image_path)}',
      overlay = True,
      control = True,
      max_zoom = 25,
    ).add_to(my_map)


  for image_path in pathgcp_after.split(','):
    map_id_dict = ee.Image.loadGeoTIFF(image_path).getMapId()
    folium.raster_layers.TileLayer(
      tiles=map_id_dict['tile_fetcher'].url_format,
      attr='COG',
      name = f'Post-Disaster Imagery {os.path.basename(image_path)}',
      overlay = True,
      control = True,
      max_zoom = 25,
    ).add_to(my_map)

  my_map.add_child(folium.LayerControl())
  display(my_map)


## CLASS DEFINITION EXAMPLEJOB

class DataflowMetricFetcher:
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
        'start_time': { 'seconds': start_seconds},
        'end_time': { 'seconds': end_seconds }
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
  def __init__(self, max):
    self._display = display(self.get_html(0, max), display_id=True)

  def get_html(self, value, max):
    return HTML(f'Num generated examples: {value}/{max}<progress value="{value}" max="{max}" style="width: 100%">{value}</progress>')

  def update(self, num_examples, max):
    self._display.update(self.get_html(num_examples, max))

def parse_dataflow_job_creation_params(param_str: str):
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
  pre_images = []
  post_images = []
  labels = []
  labels_split=[]
  total_example_num=len(list(tf.data.TFRecordDataset(path)))
  return total_example_num

def run_example_generation(generate_examples_args,path_dir_args, pretty_output=True):
  if not pretty_output:
    launch_pexpect_process(
        'generate_examples_main.py',
        generate_examples_args,
        path_dir_args, use_pexpect=False)
    return

  progress_bar = ProgressBar(1)

  child = launch_pexpect_process('generate_examples_main.py', 
                                 generate_examples_args,
                                 path_dir_args, use_pexpect=True)

  JOB_CREATION_PATTERN = 'Create job: <Job(.*clientRequestId:.*)>'
  BUILDINGS_MATCHED_PATTERN = 'Found ([0-9]+) buildings in area of interest.'

  num_buildings = 1
  while child.isalive():
    i = child.expect([BUILDINGS_MATCHED_PATTERN, JOB_CREATION_PATTERN, pexpect.EOF], timeout=3000)
    if i == 0:
      num_buildings = int(child.match.group(1))
      print(f'Found {num_buildings} buildings in area of interest.')
      progress_bar.update(0, num_buildings)
    elif i == 1:
      job_params = parse_dataflow_job_creation_params(child.match.group(1).decode())
      job_name = job_params['name']
      job_id = job_params['id']
      job_location = job_params['location']
      job_project = job_params['projectId']
      job_status_pattern = f'Job {job_id} is in state JOB_STATE_([A-Z]+)'
      print(f'Your Dataflow job is :\n{job_name}')
      print(f'Detailed monitoring page - Dataflow job id {job_id}: https://console.cloud.google.com/dataflow/jobs/{job_location}/{job_id}?project={job_project}')
      break
    else:
      print(child.before.decode())
      child.close()
      raise Exception('Job terminated unexpectedly.')

  generated_examples_metric = DataflowMetricFetcher(job_project, job_name, 'generated_examples_count')
  rejected_examples_metric = DataflowMetricFetcher(job_project, job_name, 'rejected_examples_count')

  job_state = None
  while child.isalive():
    i = child.expect([job_status_pattern, pexpect.TIMEOUT, pexpect.EOF], timeout=15)
    if i == 0:
      job_state = child.match.group(1).decode()
      print(f'Dataflow job state: {job_state}')
      if job_state == 'DONE':
        progress_bar.update(num_buildings, num_buildings)
        total_example_counter=0
        for k in range(20):
          file_directory='unlabeled/unlabeled-000{:02d}-of-00020.tfrecord'.format(k)
          tfrecord_path = os.path.join(os.path.join(generate_examples_args['output_dir'],'examples'),file_directory)
          total_example_counter+=count_tfrecord(tfrecord_path)
        print('{} building examples were extracted within the Area Of Interest and TIFF images'.format(total_example_counter))
    elif i == 1 or i == 2:
      if job_state == 'RUNNING':
        examples_processed = 0
        t, v = generated_examples_metric.get_latest_value()
        if t:
          examples_processed += int(v)
        t, v = rejected_examples_metric.get_latest_value()
        if t:
          examples_processed += int(v)
        progress_bar.update(examples_processed, num_buildings)
      if i == 2:
        child.close()
        break

## CLASS DEFINITION LABELINGJOB

class LabelingJob:
  def __init__(self, endpoint, project, location, labeling_job):
    self._endpoint = endpoint
    self._project = project
    self._location = location
    self._labeling_job = labeling_job
    self._access_token = self.get_access_token()

    job_info = self.get_info()
    self._dataset = job_info['datasets'][0]

    assert len(job_info['specialistPools']) == 1
    # Has the format projects/{project_id}/locations/{location}/specialistPools/{pool_id}
    parts = job_info['specialistPools'][0].split('/')
    assert len(parts) == 6
    assert parts[4] == 'specialistPools'
    self._pool_id = parts[5]
    
  def get_access_token(self):
    return subprocess.check_output('gcloud auth print-access-token'.split()).decode().rstrip('.\r\n')
  
  def get_header(self):
    return {
      'Authorization': f'Bearer {self._access_token}',
      'Content-Type': 'application/json',
    }
  
  def get_info(self):
    '''Return the percentage of data items labeled.

    Warning: There is a long lag between when items are labeled and when this
    value is updated.
    '''
    parent = f'projects/{self._project}/locations/{self._location}/dataLabelingJobs/{self._labeling_job}'
    url = f'https://{self._endpoint}/v1/{parent}'
    header = self.get_header()
    r = requests.get(url, headers=header)
    if not r.ok:
      r.raise_for_status()
    return r.json()

  def get_completion_percentage(self):
    '''Return the percentage of data items labeled.

    Warning: There is a long lag between when items are labeled and when this
    value is updated.
    '''
    info = self.get_info()
    return info.get('labelingProgress', 0)
    
  def get_data_items(self):
    parent = f'projects/{self._project}/locations/{self._location}/datasets/{self._dataset}/dataItems'
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
    r = requests.get(url, headers=header)
    if not r.ok:
      r.raise_for_status()
    json = r.json()
    labels = []
    if 'annotations' in json:
      for a in json['annotations']:
        labels.append(a['payload']['displayName'])
    return labels

  def get_worker_url(self):
    '''Returns the URL workers can use to access the labeling interface.

    The syntax of the URL was determined by reverse engineering, so there's no
    guarantee that it won't change in the future.
    '''
    location = self._location.replace('-', '_')
    return f'https://datacompute.google.com/w/cloudml_data_specialists_{location}_{self._pool_id}'

  def get_manager_url(self):
    '''Returns the URL managers can use to access the task management interface.

    The syntax of the URL was determined by reverse engineering, so there's no
    guarantee that it won't change in the future.
    '''
    location = self._location.replace('-', '_')
    return f'https://datacompute.google.com/cm/cloudml_data_specialists_{location}_{self._pool_id}/tasks'

def run_labeling_task_creation(create_label_task_args,path_dir_args, pretty_output=True):
  if not pretty_output:
    launch_pexpect_process(
        'create_cloud_labeling_task.py',
        create_label_task_args,
        path_dir_args, False)
    return None

  child = launch_pexpect_process(
      'create_cloud_labeling_task.py',
      create_label_task_args,
      path_dir_args, True)

  DATASET_CREATED_PATTERN = '] ImageDataset created. Resource name: projects/[^/]+/locations/[^/]+/datasets/([0-9]+)'
  LABELING_JOB_CREATED_PATTERN = '] Data labeling job created:'

  output = b''
  
  while child.isalive():
    i = child.expect(
        [DATASET_CREATED_PATTERN,
          LABELING_JOB_CREATED_PATTERN,
          pexpect.EOF,
          pexpect.TIMEOUT], timeout=1800)

    if isinstance(child.before, bytes):
      output += child.before
    if isinstance(child.after, bytes):
      output += child.after

    if i == 0:
      dataset_id = child.match.group(1).decode()
      print('Creating data labeling job...')
    elif i == 1:
      print('Data labeling job created.')

      url='https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/dataLabelingJobs'.format(create_label_task_args["cloud_location"],
                                                                                                     create_label_task_args["cloud_project"],
                                                                                                     create_label_task_args["cloud_location"])
      data = make_gcp_http_request(url)
      data = list(filter(lambda d: create_label_task_args["dataset_name"] in d['displayName'], data['dataLabelingJobs']))[0]

      dataset_id = int(data['datasets'][0].split('/')[-1])
      dataset_name = data['displayName']
      labelingjob_id= int(data['name'].split('/')[-1])
      labelingjob_instruction= data['instructionUri']

      print(f'\nLabeling dataset {dataset_name} created, with ID {dataset_id}')
      print(f'\nData Labeling job {labelingjob_id} created')

      labeling_job = LabelingJob(f'{create_label_task_args["cloud_location"]}-aiplatform.googleapis.com',
                                 create_label_task_args["cloud_project"],
                                 create_label_task_args["cloud_location"], labelingjob_id)
      
      print('Instruction URL: {}'.format(labelingjob_instruction.replace('gs://','https://storage.cloud.google.com/')))
      print(f'Worker URL: {labeling_job.get_worker_url()}')
      print(f'Manager URL: {labeling_job.get_manager_url()}')
      print(f'Detailed monitoring page: https://console.cloud.google.com/vertex-ai/locations/{create_label_task_args["cloud_location"]}/labeling-tasks/{labelingjob_id}?project={create_label_task_args["cloud_project"]}')

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
  return dataset_id,dataset_name,labelingjob_id,labelingjob_instruction

## CLASS DEFINITION DATASETJOB

def create_labeled_dataset(create_labeled_dataset_args,path_dir_args):
  
  child = launch_pexpect_process(
      'create_labeled_dataset.py',
      create_labeled_dataset_args, 
      path_dir_args,True)

  print('Creating labeled datasets...')
  child.expect(pexpect.EOF, timeout=None)
  child.close()
  if child.exitstatus != 0:
    print('An unexpected error occurred. Output of command was:')
    print(child.before.decode())
  else:
    print('Labeled dataset created.')

## CLASS DEFINTION IPYPLOT

def concat_caption_pilimage(image_before, image_after):
  img_before=caption_pilformat(image_before, "before")
  img_after=caption_pilformat(image_after, "after")

  w, h=img_before.size

  img_concat = Image.new('RGB', (2*w, h), "white")
  img_concat.paste(img_before, (0, 0))
  img_concat.paste(img_after, (w, 0))

  return img_concat

def caption_pilformat(img_data, caption):
  base64_encoded = base64.b64encode(img_data)
  im_bytes = base64.b64decode(base64_encoded)
  byte_encoded=io.BytesIO(im_bytes)

  img=Image.open(byte_encoded)
  wd, hg =img.size

  img_ = Image.new('RGB', (wd+int(wd/10), hg+int(hg/5)), "white")
  img_.paste(img, (int(wd/20),int(hg/5)))

  wd, hg =img_.size
  img_cap = ImageDraw.Draw(img_)
  _,_,w, h = img_cap.textbbox((0, 0),caption)
  img_cap.text(((wd-w)/2,0), caption, fill=(0, 0, 0))

  return img_

def ipyplot_tfrecord(path, max_examples=None):
  pre_images = []
  post_images = []
  labels = []
  labels_split=[]
  total_example_num=len(list(tf.data.TFRecordDataset(path)))
  print('Number of examples: {}.'.format(total_example_num))

  if max_examples==None:
    max_examples=total_example_num

  for record in tf.data.TFRecordDataset(path):
    e = tf.train.Example()
    e.ParseFromString(record.numpy())
    labels_split.append(e.features.feature['label'].float_list.value[0])
    if len(pre_images) < max_examples:
      pre_images.append(e.features.feature['pre_image_png'].bytes_list.value[0])
      post_images.append(e.features.feature['post_image_png'].bytes_list.value[0])
      labels.append(e.features.feature['label'].float_list.value[0])

  labels_counter=dict(collections.Counter(labels_split))
  map_value = {0: 'Undamaged/bad examples {}/{}'.format(int(len(labels)-sum(labels)),labels_counter[0]),
               1: 'Damaged {}/{}'.format(int(sum(labels)),labels_counter[1])}
  labels=list((pd.Series(labels)).map(map_value))
  
  images=[concat_caption_pilimage(pre_images[idx], post_images[idx]) for idx in range(len(pre_images))]

  ipyplot.plot_class_tabs(images, labels,max_imgs_per_tab=max_examples, tabs_order=[map_value[1],map_value[0]], img_width=200)

  return total_example_num


## CLASS DEFINITION TRAINJOB

def write_train_and_eval_launch_script(**args):

  args['hyper_parameters_args']=''

  submission_ending = '''
source {python_env}; export GOOGLE_APPLICATION_CREDENTIALS={path_cred};
python {path_skai}/src/launch_vertex_job.py \\
  --location={cloud_region} \\
  --project={cloud_project} \\
  --job_type=train \\
  --display_name={display_name_train} \\
  --dataset_name={dataset_name} \\
  --train_worker_machine_type=n1-highmem-16 \\
  --train_docker_image_uri_path={train_docker_image_uri_path} \\
  --service_account={service_account} \\
  --train_dir={train_dir} \\
  --train_unlabel_examples={train_unlabel_examples} \\
  --train_label_examples={train_label_examples} \\
  --test_examples={test_examples} & \\
sleep 60 ; python {path_skai}/src/launch_vertex_job.py \\
  --location={cloud_region} \\
  --project={cloud_project} \\
  --job_type=eval \\
  --display_name={display_name_eval} \\
  --dataset_name={dataset_name} \\
  --eval_docker_image_uri_path={eval_docker_image_uri_path} \\
  --service_account={service_account} \\
  --train_dir={train_dir} \\
  --train_unlabel_examples={train_unlabel_examples} \\
  --train_label_examples={train_label_examples} \\
  --test_examples={test_examples}'''.format(**args)
  print(submission_ending)
  with open(args['path_run'], 'w+') as file:
    file.write(submission_ending)
    
def metrics(train_label_acc, train_label_auc, test_acc, test_auc, train_epoch,test_epoch, time):
  html = """
         <h2>Metrics (updated as training progresses {timestamp}):</h2>
         <h3>Labeled Training Set, Epoch {train_epoch}</h3> 
         <p>Accuracy: {train_label_acc}% | AUC: {train_label_auc}</p>
         <h3>Test Set, Epoch {test_epoch}</h3>
         <p>Accuracy: {test_acc}% | AUC: {test_auc}</p>
        """.format(train_label_acc=train_label_acc, train_label_auc=train_label_auc,
                   test_acc=test_acc, test_auc=test_auc,
                   train_epoch=train_epoch,test_epoch=test_epoch,
                   timestamp=time )
  return HTML(html)

def timestamp_to_datetime(timestamp):
  return pd.to_datetime(timestamp)

def update_job_id(job_id, train_job_id,eval_job_id):
  if train_job_id is None:
    train_job_id = job_id
  elif eval_job_id is None:
    if job_id != train_job_id:
      eval_job_id = job_id
  return train_job_id,eval_job_id

def run_train_and_eval_job(path_file,email_manager,load_tensorboard=False,path_log_tensorboard=None):

  # Create the progress bar and metrics displays.
  progress_display = display(progress(0, 100), display_id=True)
  metrics_display = None

  # Store Job IDs of training and evaluation jobs.
  # Keep track of the timestamp of most recent logs to process only fresher logs. 
  train_job_id, eval_job_id = None, None 
  curr_epoch = None
  total_num_epochs = None
  total_num_img= None
  train_most_recent_timestamp = pd.Timestamp.utcnow() 
  eval_most_recent_timestamp = pd.Timestamp.utcnow() 

  train_state, eval_state= None, None 

  # Run the child program.
  child = pexpect.spawn(f'sh {path_file}')
  while not child.closed:
    # Expects 5 different patterns, or EOF (meaning the program terminated).
    # Each pattern is a regex and you can use regex match groups "()" to extract a
    # part of the matched text for later use.
    pattern_idx = child.expect([
      'I.*] Creating CustomJob',
      'I.*] CustomJob created\. Resource name: .*/([0-9]*)',
      'I.*] View Custom Job:\r\n(.*)\r\nCustomJob',
      'I.*] CustomJob .*/([0-9]*) current state:\r\nJobState.JOB_STATE_PENDING',
      'I.*] CustomJob .*/([0-9]*) current state:\r\nJobState.JOB_STATE_RUNNING',
      'I.*] CustomJob run completed.',
      pexpect.EOF
      ], timeout=None)
    if pattern_idx ==0:
      progress_display.update(progress(0, 100))

    elif pattern_idx == 1:  # A job was created, so store its ID.
      job_id = child.match.group(1).decode()
      train_job_id,eval_job_id=update_job_id(job_id,train_job_id,eval_job_id)
      if train_job_id is not None:
        progress_display.update(progress(1, 100))

    elif pattern_idx == 2:  # Jobs are created. 
      if job_id == train_job_id:
        print(f'Training CustomJob created\nDetailed monitoring page - Train job id {job_id} : {child.match.group(1).decode()}')
      else:
        print(f'Evaluation CustomJob created\nDetailed monitoring page - Eval job id {job_id} : {child.match.group(1).decode()}')

    elif pattern_idx == 3:  # Jobs are pending, so update progress bar. 
      job_id = child.match.group(1).decode()
      if job_id == train_job_id:
        progress_display.update(progress(5, 100))
        if train_state is None:          
          train_state='PENDING'
          print(f'Training CustomJob state: {train_state}')
      else:
        if eval_state is None:
          eval_state='PENDING'
          print(f'Evaluation CustomJob state: {eval_state}')


    elif pattern_idx == 4:  # Jobs are running, so update progress bar or metrics.
      job_id = child.match.group(1).decode()
      get_logs_status = os.system(f"""gcloud logging read 'resource.labels.job_id={job_id} severity=ERROR "Epoch"' --format json > /tmp/{job_id}_log""")
      if get_logs_status == 0:
        with open(f'/tmp/{job_id}_log', 'r') as log_file:
          log_data = json.load(log_file)    
          if job_id == train_job_id:
            if train_state=='PENDING':
              train_state='RUNNING'
              print(f'Training CustomJob state: {train_state}')
            # If training job, then update the progress bar.
            curr_epoch = None
            for log in log_data:
              log_timestamp = timestamp_to_datetime(log["timestamp"])
              if log_timestamp < train_most_recent_timestamp:
                # If logs have not been refreshed, ignore them.
                break
              else:
                train_most_recent_timestamp = log_timestamp
              if log_timestamp == train_most_recent_timestamp:
                matches = re.search('Epoch ([0-9]*/[0-9]*):   [0-9]*%.* [0-9]*/([0-9]*)', log['jsonPayload']['message'])
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
                    progress_display.update(progress(5 + int(95. * curr_epoch / (total_num_epochs + 1)), 100))
                    break
          else:
            # If evaluation job, then update the metrics display.
            if eval_state=='PENDING':
              eval_state='RUNNING'
              print(f'Evaluation CustomJob state: {eval_state}\n')

            train_label_acc, train_label_auc, test_acc, test_auc = None, None, None, None
            for log in log_data:
              log_timestamp = timestamp_to_datetime(log["timestamp"])
              if log_timestamp < eval_most_recent_timestamp:
                # If logs have not been refreshed, ignore them.
                break
              if train_label_acc is None:
                train_label_matches = re.search('Epoch: ([0-9]*), .* Train_Label AUC: ([0-9]*\.[0-9]*), Train_Label Accuracy: ([0-9]*\.[0-9]*)', log['jsonPayload']['message'])
                if train_label_matches:
                  train_label_epoch = min((int(train_label_matches.groups()[0])//total_num_img)+1,total_num_epochs)
                  train_label_epoch = f'{train_label_epoch}/{total_num_epochs}'
                  train_label_auc = train_label_matches.groups()[1]
                  train_label_acc = train_label_matches.groups()[2]
              elif test_acc is None:
                test_matches = re.search('Epoch: ([0-9]*), .* Test AUC: ([0-9]*\.[0-9]*), Test Accuracy: ([0-9]*\.[0-9]*)', log['jsonPayload']['message'])
                if test_matches:
                  test_epoch = min((int(test_matches.groups()[0])//total_num_img)+1,total_num_epochs)
                  test_epoch = f'{test_epoch}/{total_num_epochs}'
                  test_auc = test_matches.groups()[1]
                  test_acc = test_matches.groups()[2]
              else:
                eval_most_recent_timestamp = log_timestamp
                break
            if train_label_acc is not None and test_acc is not None:
              if metrics_display is None:
                metrics_display = display(metrics(0, 0, 0, 0, 0, 0, 
                                                  eval_most_recent_timestamp.strftime("%Y-%m-%d, %H:%M:%S")), display_id=True)
                print('\n')
                if load_tensorboard:
                  load_start_tensorboard(path_log_tensorboard)
              else :
                metrics_display.update(metrics(train_label_acc, train_label_auc, test_acc, test_auc,train_label_epoch,test_epoch,
                                               eval_most_recent_timestamp.strftime("%Y-%m-%d, %H:%M:%S")))          
    
    elif pattern_idx == 5:  # Job completed. Email user a notification.
      print(f'Training CustomJob state: DONE\n')
      os.system(f"""printf 'Subject: Skai Training Complete\n\nTraining has completed! Please return to the Colab to visualize results.' | msmtp {email_manager}""")
      progress_display.update(progress(100, 100))    
    else:
      child.close()
      
## CLASS DEFINITION INFERENCEJOB

def get_epoch_number(path_experiment,id_eval_job,checkpoint_selection, checkpoint_index):
  if checkpoint_selection=='most_recent':
    most_recent_epoch_file = os.path.join(f'gs://{path_experiment}', 'checkpoints', 'last_processed_epoch')
    os.system(f'gsutil cp {most_recent_epoch_file} /tmp/last_processed_epoch')
    with open('/tmp/last_processed_epoch', 'r') as epoch_f:
      epoch_num = epoch_f.read()

  elif checkpoint_selection in ["top_auc_test","top_acc_test"]:
    epoch_num_acc,metrics_acc,epoch_num_auc,metrics_auc=None, None, None, None
    get_logs_status = os.system(f"""gcloud logging read 'resource.labels.job_id={id_eval_job} severity=ERROR "Epoch"' --format json > /tmp/{id_eval_job}_log""")
    if get_logs_status == 0:
      with open(f'/tmp/{id_eval_job}_log', 'r') as log_file:
        log_data = json.load(log_file)    
        for log in log_data:
          test_matches = re.search('Epoch: ([0-9]*),.*Test AUC: ([0-9]*\.[0-9]*), Test Accuracy: ([0-9]*\.[0-9]*)', log['jsonPayload']['message'])
          if test_matches:

            if epoch_num_acc==None:
              epoch_num_acc=test_matches.groups()[0]
              metrics_acc=float(test_matches.groups()[2])
            elif (int(test_matches.groups()[0])>int(epoch_num_acc) and float(test_matches.groups()[2])==metrics_acc) or (float(test_matches.groups()[2])>metrics_acc):
              epoch_num_acc=test_matches.groups()[0]
              metrics_acc=float(test_matches.groups()[2])

            if epoch_num_auc==None:
              epoch_num_auc=test_matches.groups()[0]
              metrics_auc=float(test_matches.groups()[1])
            elif (int(test_matches.groups()[0])>int(epoch_num_auc) and float(test_matches.groups()[1])==metrics_auc) or (float(test_matches.groups()[1])>metrics_auc):
              epoch_num_auc=test_matches.groups()[0]
              metrics_auc=float(test_matches.groups()[1])
              

        if checkpoint_selection =="top_auc_test":
          epoch_num = epoch_num_auc
        elif checkpoint_selection =="top_acc_test":
          epoch_num = epoch_num_acc

  elif checkpoint_selection == "index_number":
    epoch_num = str(int(checkpoint_index))

  metrics_acc,metrics_auc=None, None
  get_logs_status = os.system(f"""gcloud logging read 'resource.labels.job_id={id_eval_job} severity=ERROR "Epoch"' --format json > /tmp/{id_eval_job}_log""")
  if get_logs_status == 0:
    with open(f'/tmp/{id_eval_job}_log', 'r') as log_file:
      log_data = json.load(log_file)    
      for log in log_data:
        test_matches = re.search('Epoch: ([0-9]*),.*Test AUC: ([0-9]*\.[0-9]*), Test Accuracy: ([0-9]*\.[0-9]*)', log['jsonPayload']['message'])
        if test_matches:
          if int(test_matches.groups()[0])==int(epoch_num):
            metrics_acc=test_matches.groups()[2]
            metrics_auc=test_matches.groups()[1]
            break

  if checkpoint_selection =="top_auc_test":
    print(f'Checkpoint used: {epoch_num}, Test AUC (best): {metrics_auc}, Test Accuracy: {metrics_acc}')
  elif checkpoint_selection=="top_acc_test":
    print(f'Checkpoint used: {epoch_num}, Test AUC: {metrics_auc}, Test Accuracy (best): {metrics_acc}')
  else:
    print(f'Checkpoint used: {epoch_num}, Test AUC: {metrics_auc}, Test Accuracy: {metrics_acc}')

  return epoch_num.zfill(8)

def write_generate_inference_script(**args):

  submission_ending = '''
source {python_env}; export GOOGLE_APPLICATION_CREDENTIALS={path_cred};
python {path_skai}/src/launch_vertex_job.py \\
  --location={cloud_region} \\
  --project={cloud_project} \\
  --job_type=eval \\
  --display_name={display_name_infer} \\
  --eval_worker_machine_type=n1-highmem-16 \\
  --dataset_name={dataset_name} \\
  --eval_docker_image_uri_path={eval_docker_image_uri_path} \\
  --service_account={service_account} \\
  --train_dir={train_dir} \\
  --test_examples={test_examples} \\
  --eval_ckpt={eval_model_ckpt} \\
  --inference_mode=True \\
  --save_predictions=True'''.format(**args)
  print(submission_ending)
  with open(args['path_run'], 'w+') as file:
    file.write(submission_ending)

def run_inference_and_prediction_job(path_file):

  # Initialize progress bar.
  progress_display = display(progress(0, 100), display_id=True)
  curr_idx = 0
  map = None

  # Run the child program.
  child = pexpect.spawn(f'sh {path_file}')
  while not child.closed:
    # Expects 5 different patterns, or EOF (meaning the program terminated).
    # Each pattern is a regex and you can use regex match groups "()" to extract a
    # part of the matched text for later use.
    pattern_idx = child.expect([
      'CustomJob created\.',
      'JobState\.JOB_STATE_PENDING\r\n',
      'JobState\.JOB_STATE_RUNNING\r\n',
      'JobState\.JOB_STATE_SUCCEEDED\r\n',
      'CustomJob run completed\.',
      pexpect.EOF], timeout=None)
    if pattern_idx == 0:  # A job was created.
      progress_display.update(progress(5, 100))
    elif pattern_idx == 1:  # Job Pending.
      progress_display.update(progress(10, 100))
    elif pattern_idx == 2:  # Job Running.
      starting_progress = 20
      max_progress = 90
      curr_progress = starting_progress + (curr_idx * 2)
      if curr_idx == 0:
        progress_display.update(progress(starting_progress, 100))
      elif curr_progress < max_progress:
        # Update while job is running only until progress hits 90.
        progress_display.update(progress(curr_progress, 100))
      curr_idx += 1
    elif pattern_idx == 3 or pattern_idx == 4:  # Job Completed.
      progress_display.update(progress(100, 100))
    else:
      child.close()

def create_folium_map(geojson_path, pathgcp_before, pathgcp_after):
  with open(geojson_path, 'r') as f:
    predictions = json.load(f)

  damaged_preds = {
      'type': predictions['type'],
      'features': []
  }
  undamaged_preds = {
      'type': predictions['type'],
      'features': []
  }

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

  no_cog_file=''
  error_message=[]
  for image_path in pathgcp_before.split(',')+pathgcp_after.split(','):
    try:
      ee.Image.loadGeoTIFF(image_path).getMapId()
    except Exception as e:
      if re.search('The GeoTIFF is invalid or is not cloud optimized:',str(e)):
        no_cog_file=f'{no_cog_file}{image_path}\n'
        error_message.append(e)

  if error_message:
    print(f'The following TIFF image(s) need to be Cloud Optimzed GeoTIFF file(s) in order to be visualized in the map using EarthEngine:\n{no_cog_file}\n')
  
  else:
    for image_path in pathgcp_before.split(','):
      map_id_dict = ee.Image.loadGeoTIFF(image_path).getMapId()
      folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='COG',
        name = f'Pre-Disaster Imagery {os.path.basename(image_path)}',
        overlay = True,
        control = True,
        max_zoom = 20,
      ).add_to(my_map)

    for image_path in pathgcp_after.split(','):
      map_id_dict = ee.Image.loadGeoTIFF(image_path).getMapId()
      folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='COG',
        name = f'Post-Disaster Imagery {os.path.basename(image_path)}',
        overlay = True,
        control = True,
        max_zoom = 20,
      ).add_to(my_map)

  # Add predictions.
  # Set style parameters.
  style_function = lambda x: {
    'radius': 3,
    'weight': 1,
    'fill': True,
    'color': '#ff0000' if float(x['properties']['class_1']) >= 0.5 else '#00ff00',
    'fillColor': '#ff0000' if float(x['properties']['class_1']) >= 0.5 else '#00ff00',
    'fillOpacity': 0.3,
  }

  marker_function = lambda x : folium.CircleMarker([x['properties']['latitude'],x['properties']['longitude']])

  folium.features.GeoJson(damaged_preds, name='Damaged Predictions', 
                          style_function=style_function,
                          marker=folium.CircleMarker(),
                          ).add_to(my_map)
  folium.features.GeoJson(undamaged_preds, name='Undamaged Predictions', 
                          style_function=style_function,show =False,
                          marker=folium.CircleMarker(),
                          ).add_to(my_map)

  data_heat=[[x['properties']['latitude'],x['properties']['longitude'],(x['properties']['class_1']>= 0.5)] for x in predictions['features']]
  
  HeatMap(data_heat, name='Heatmap Predictions',gradient={0.9: 'grey',0.95: 'yellow',0.99: 'orange', 0.995: 'red', 1:'darkred'}).add_to(my_map)

  my_map.add_child(folium.LayerControl())

  print('Number of Damaged Buildings: ', num_damaged_buildings)
  print('Number of Undamaged Buildings: ', num_undamaged_buildings)
  print('Total: ', int(num_undamaged_buildings) + int(num_damaged_buildings))
  display(my_map)




