# Copyright 2024 Google LLC
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

"""Eagle Eye - Image Labeling for SKAI disaster response.

To run locally, run `pip install -r requirements.txt` in a virtualenv,
and run with `python main.py`.
"""

import concurrent.futures
import datetime
import io
import json
import math
import os
import random
import time
import firebase_admin
from firebase_admin import auth
from flask import abort
from flask import Flask
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import send_file
import google.api_core.exceptions
from google.cloud import firestore
from google.cloud import storage
import pandas as pd
from project import get_completed_task_count
from project import get_project_from_db
from project import get_task_count
from project import get_task_count_with_label
from project import get_tasks
from project import LABELED
from project import NOT_LABELED
from project import Project
from project import TASKS_PER_PAGE
from task import get_task_from_db
from task import normalize_image_url_suffix
from task import Task
from task import TASK_LABEL_MAP

app = Flask(__name__)
storage_client = storage.Client()
firebase_app = firebase_admin.initialize_app()
firestore_db = firestore.Client(database=os.environ.get('FIRESTORE_DB'))
MAX_WORKER_THREAD_COUNT = 1000


LABEL_FILTER_OPTIONS = (
    [('', 'All')]
    + list(TASK_LABEL_MAP.items())
    + [
        (LABELED, 'Labeled'),
        (NOT_LABELED, 'Not Labeled'),
    ]
)


def _open_gcs_file(path: str, mode: str):
  """Opens a GCS path."""
  if not path.startswith('gs://'):
    raise RuntimeError('GCS path does not begin with "gs://".')
  bucket_name, _, file_name = path[5:].partition('/')
  if not bucket_name:
    raise RuntimeError('Path does not specify a bucket.')
  if not file_name:
    raise RuntimeError('Path does not have a file name component.')
  try:
    bucket = storage_client.get_bucket(bucket_name)
  except (
      google.api_core.exceptions.Forbidden,
      google.api_core.exceptions.NotFound,
      google.api_core.exceptions.BadRequest,
  ) as e:
    raise RuntimeError(f'Error reading bucket "{bucket_name}": {e}') from e

  blob = bucket.get_blob(file_name)
  if not blob:
    raise RuntimeError('Path not found.')
  return blob.open(mode)


def _parse_image_metadata_file(path):
  """Reads and returns a set of Tasks.

  Parses the given image metadata file. Returns None if the metadata file does
  not exist or does not conform to the expected format.

  Args:
    path: Path to open.

  Returns:
    List of tasks.

  Raises:
    RuntimeError: If file parsing fails in some way.
  """
  start_time = time.perf_counter()
  with _open_gcs_file(path, 'r') as image_metadata_file:
    df = pd.read_csv(image_metadata_file)

  for column in ('example_id', 'pre_image_path', 'post_image_path'):
    if column not in df.columns:
      raise RuntimeError(
          f'Metadata file does not contain required column "{column}".'
      )

  tasks_local = [
      Task(row['example_id'], row['pre_image_path'], row['post_image_path'])
      for _, row in df.iterrows()
  ]

  end_time = time.perf_counter()
  print(
      f'parsed image metadata file "{path}" in'
      f' {end_time - start_time} sec'
  )
  return tasks_local


@app.route('/')
def index():
  """Default route, unauthenticated access.

  If user does not have a Firebase session token, requests a new
  Firebase auth token, generates a session token, and sets it as a
  page cookie.

  Redirects to the /projects endpoint after successful authorization.
  """
  with open('config.json') as config_file:
    config_json = json.load(config_file)
    return render_template('default.html', configs=config_json)


def require_authentication(decorated_func):
  """A decorator function checking for a valid session token.

  Checks for a valid session token before executing the given
  decorated_func, or redirects to the default route for sign-in if the
  session token is invalid.

  If the session token is valid, passes the `email` field from the
  session token to the decorated method.
  """

  def wrapper(*args, **kwargs):
    session_token = request.cookies.get('sessionToken')
    if session_token:
      try:
        decoded_token = auth.verify_session_cookie(session_token)
        kwargs['user_email'] = decoded_token['email']
        return decorated_func(*args, **kwargs)
      except (
          auth.InvalidIdTokenError,
          auth.ExpiredIdTokenError,
          auth.RevokedIdTokenError,
      ) as e:
        print('invalid auth token, redirecting to /.', e)
    return redirect('/')

  wrapper.__name__ = decorated_func.__name__
  return wrapper


def require_admin(decorated_func):
  """A stronger version of require_authentication.

  Checks for a valid session token with an admin custom claim before
  executing the given decorated_func, otherwise redirects to the
  default route.
  """

  def wrapper(*args, **kwargs):
    session_token = request.cookies.get('sessionToken')
    if session_token:
      try:
        decoded_token = auth.verify_session_cookie(session_token)
        if decoded_token.get('admin'):
          return decorated_func(*args, **kwargs)
      except (
          auth.InvalidIdTokenError,
          auth.ExpiredIdTokenError,
          auth.RevokedIdTokenError,
      ) as e:
        print('invalid admin token:', e)
    return abort(401, 'Unauthorized for admin functionality.')

  wrapper.__name__ = decorated_func.__name__
  return wrapper


def _check_is_admin(session_token):
  """Checks if the user is an admin, based on the given session token."""
  claims = auth.verify_session_cookie(session_token)
  return claims.get('admin')


@app.route('/projects', methods=['GET'])
@require_authentication
def homepage(user_email):
  """Returns the eagleeye homepage."""
  del user_email  # Unused

  is_admin = _check_is_admin(request.cookies.get('sessionToken'))

  all_projects_snapshot = firestore_db.collection('projects').get()
  all_projects = sorted(
      [
          Project(
              project_snapshot.id,
              project_snapshot.get('creation_time'),
              project_snapshot.get('name'),
              project_snapshot.get('image_metadata_path'),
          )
          for project_snapshot in all_projects_snapshot
      ],
      key=lambda x: x.creation_time,
      reverse=True,
  )

  project_task_count_map = {
      project.project_id: get_task_count(project.project_id, firestore_db)
      for project in all_projects
  }
  project_labeled_task_count_map = {
      project.project_id: get_completed_task_count(
          project.project_id, firestore_db
      )
      for project in all_projects
  }

  return render_template(
      'homepage.html',
      is_admin=is_admin,
      all_projects=all_projects,
      project_task_count_map=project_task_count_map,
      project_labeled_task_count_map=project_labeled_task_count_map,
  )


@app.route('/getSessionCookie', methods=['POST'])
def getSessionCookie():
  """Creates and sets a sessionToken cookie from the given authToken.

  Firebase auth tokens have a short (1h) expiration. This endpoint
  creates a Firebase session token (with a 14d expiration) and sets a
  cookie on the request browser with a corresponding 14d expiration.
  """
  auth_token = request.json['authToken']
  expires_in = datetime.timedelta(days=14)
  try:
    session_cookie = auth.create_session_cookie(
        auth_token, expires_in=expires_in
    )
    response = jsonify({'status': 'success'})
    expires = datetime.datetime.now() + expires_in
    response.set_cookie(
        'sessionToken',
        session_cookie,
        expires=expires,
        httponly=True,
        secure=True,
    )
    return response
  except firebase_admin.exceptions.FirebaseError as e:
    print(f'Exception in getSessionCookie: {e}')
    return abort(401, 'Failed to create session cookie')


@app.route('/projects/new', methods=['GET'])
@require_admin
def new_project_form():
  """Returns a form to create a new project."""
  return render_template('newproject.html')


@app.route('/projects/new', methods=['POST'])
@require_admin
def create_new_project():
  """Creates a new project with the given parameters."""
  project_name = request.json['projectName']
  image_metadata_path = request.json['imageMetadataPath']
  if not project_name or not image_metadata_path:
    return 'Invalid project creation parameters', 400
  print(
      f'creating project with name: {project_name}, image_metadata file:'
      f' {image_metadata_path}'
  )

  print('reading image metadata file, writing tasks to firestore')
  try:
    tasks_local = _parse_image_metadata_file(image_metadata_path)
  except RuntimeError as e:
    print(f'Error reading image metadata file: {e}')
    return abort(400, 'Error occurred in parsing image metadata file.')
  if tasks_local is None:
    return 'Invalid image metadata file path.', 400

  new_project = Project(None, None, project_name, image_metadata_path)
  new_project.write_to_firestore(firestore_db)

  print(f'Populating tasks for project: {new_project.project_id}')
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=MAX_WORKER_THREAD_COUNT
  ) as executor:
    for task in tasks_local:
      executor.submit(
          task.write_to_firestore, new_project.project_id, firestore_db
      )

  return jsonify(success=True, projectId=new_project.project_id)


@app.route('/project/<path:project_id>/reopen', methods=['GET'])
@require_admin
def reopen_project_form(project_id):
  """Returns a form to reopen a project."""
  project = get_project_from_db(firestore_db, project_id)
  if not project:
    return abort(404, 'Invalid project id')
  return render_template(
      'reopen_project.html',
      project_id=project.project_id,
      project_name=project.name,
  )


@app.route('/project/<path:project_id>/reopen', methods=['POST'])
@require_admin
def reopen_project(project_id):
  """Reopens the given project."""
  image_metadata_path = request.json['imageMetadataPath']
  if not image_metadata_path:
    return 'Invalid project reopen parameters', 400
  print(
      f'reopening project: {project_id}, image metadata file:'
      f' {image_metadata_path}'
  )

  print('reading image metadata file, writing tasks to firestore')
  try:
    tasks_local = _parse_image_metadata_file(image_metadata_path)
  except RuntimeError as e:
    print(
        f'Error reading image metadata file: {image_metadata_path},'
        f' exception: {e}'
    )
    return abort(400, 'Error occurred in parsing image metadata file.')
  if tasks_local is None:
    return 'Invalid image metadata file path.', 400

  project = get_project_from_db(firestore_db, project_id)
  if not project:
    return abort(404, 'Invalid project id')

  print(f'Populating tasks for reopened project: {project.project_id}')
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=MAX_WORKER_THREAD_COUNT
  ) as executor:
    for task in tasks_local:
      executor.submit(task.write_to_firestore, project.project_id, firestore_db)

  return redirect(f'/project/{project.project_id}/summary')


@app.route(
    '/project/<path:project_id>/task/<path:example_id>', methods=['POST']
)
@require_authentication
def submit_task(project_id, example_id, user_email):
  """Handles the task submission POST request.

  Writes the labeled task to Firestore. Note that this overwrites the
  existing label, if one is currently set.

  Args:
    project_id: project id
    example_id: example id
    user_email: email of the labeler
  """
  print(f'handling submit task for example id: {example_id}')
  project = get_project_from_db(firestore_db, project_id)
  if not project:
    return abort(404, 'Invalid project id')

  task = get_task_from_db(firestore_db, project.project_id, example_id)
  task.labeler = user_email
  task.label = request.json['assessment']
  task.start_time = request.json['taskStartTime']
  task.submit_time = request.json['taskSubmitTime']
  task.write_to_firestore(project.project_id, firestore_db)

  return jsonify(success=True)


@app.route('/project/<path:project_id>/task/<path:example_id>', methods=['GET'])
@require_authentication
def get_task(project_id, example_id, user_email):
  """Returns the task labeling page for the given project_id and example_id."""
  del user_email  # Unused

  print(f'Returning task: {example_id}, in project: {project_id}')
  project = get_project_from_db(firestore_db, project_id)
  if not project:
    return abort(404, 'Invalid project id')

  all_tasks_count = get_task_count(project.project_id, firestore_db)
  labeled_task_count = get_completed_task_count(
      project.project_id, firestore_db
  )

  task_firestore = (
      firestore_db.collection(project.project_id).document(example_id).get()
  )
  task = Task(
      example_id,
      task_firestore.get('preImage'),
      task_firestore.get('postImage'),
      task_firestore.get('label'),
  )

  normalized_pre_image_suffix = normalize_image_url_suffix(task.pre_image)
  normalized_post_image_suffix = normalize_image_url_suffix(task.post_image)
  return render_template(
      'task.html',
      labels=TASK_LABEL_MAP,
      current_label=task.label,
      project_id=project.project_id,
      project_labeled_task_count=labeled_task_count,
      project_task_count=all_tasks_count,
      example_id=example_id,
      pre_image=normalized_pre_image_suffix,
      post_image=normalized_post_image_suffix,
  )


@app.route('/project/<path:project_id>/next', methods=['GET'])
@require_authentication
def get_next_task(project_id, user_email):
  """Returns a currently unlabeled task, in a pseudo-random order.

  If there are no remaining unlabeled tasks, redirects to the project
  summary page.

  LIMIT_TASK_COUNT_READS implements a tradeoff between limiting the
  amount of data read from firestore on each call to get_next_task(),
  and introducing enough randomness such that multiple users
  concurrently labeling tasks are unlikely to have duplicate tasks /
  wasted effort.

  Args:
    project_id: project id
    user_email: email of the labeler
  """
  del user_email  # Unused

  LIMIT_TASK_COUNT_READS = 20
  project = get_project_from_db(firestore_db, project_id)
  if not project:
    return abort(404, 'Invalid project id')

  unlabeled_tasks = (
      firestore_db.collection(project.project_id)
      .where(filter=firestore.FieldFilter('label', '==', ''))
      .limit_to_last(LIMIT_TASK_COUNT_READS)
      .get()
  )
  if len(unlabeled_tasks) <= 0:
    print('project has no remaining tasks, redirect to summary.')
    return redirect(f'/project/{project.project_id}/summary')

  next_task_index = random.randrange(0, len(unlabeled_tasks))
  next_task = unlabeled_tasks[next_task_index]

  next_example_id = next_task.get('exampleId')
  print(f'returning next example id: {next_example_id}')
  return redirect(f'/project/{project.project_id}/task/{next_example_id}')


@app.route('/project/<path:project_id>/summary', methods=['GET'])
@require_authentication
def summary(project_id, user_email):
  """Returns the project summary page."""
  del user_email  # Unused

  is_admin = _check_is_admin(request.cookies.get('sessionToken'))

  project = get_project_from_db(firestore_db, project_id)
  if project.name is None:
    return 'Project with the given id was not found.', 400

  all_tasks_count = get_task_count(project_id, firestore_db)
  labeled_task_count = get_completed_task_count(project_id, firestore_db)
  destroyed_tasks_count = get_task_count_with_label(
      project_id, firestore_db, 'destroyed'
  )
  major_damage_tasks_count = get_task_count_with_label(
      project_id, firestore_db, 'major_damage'
  )
  minor_damage_tasks_count = get_task_count_with_label(
      project_id, firestore_db, 'minor_damage'
  )
  no_damage_tasks_count = get_task_count_with_label(
      project_id, firestore_db, 'no_damage'
  )
  bad_example_tasks_count = get_task_count_with_label(
      project_id, firestore_db, 'bad_example'
  )
  not_sure_tasks_count = get_task_count_with_label(
      project_id, firestore_db, 'not_sure'
  )

  labeled_task_percent = '0.0%'
  destroyed_task_percent = '0.0%'
  major_damage_task_percent = '0.0%'
  minor_damage_task_percent = '0.0%'
  no_damage_task_percent = '0.0%'
  bad_example_task_percent = '0.0%'
  not_sure_task_percent = '0.0%'
  if all_tasks_count > 0:
    labeled_task_percent = '{:.0%}'.format(labeled_task_count / all_tasks_count)
    if labeled_task_count > 0:
      destroyed_task_percent = '{:.0%}'.format(
          destroyed_tasks_count / labeled_task_count
      )
      major_damage_task_percent = '{:.0%}'.format(
          major_damage_tasks_count / labeled_task_count
      )
      minor_damage_task_percent = '{:.0%}'.format(
          minor_damage_tasks_count / labeled_task_count
      )
      no_damage_task_percent = '{:.0%}'.format(
          no_damage_tasks_count / labeled_task_count
      )
      bad_example_task_percent = '{:.0%}'.format(
          bad_example_tasks_count / labeled_task_count
      )
      not_sure_task_percent = '{:.0%}'.format(
          not_sure_tasks_count / labeled_task_count
      )

  return render_template(
      'summary.html',
      is_admin=is_admin,
      creation_time=project.creation_time.strftime('%Y-%m-%d %I:%M %p %Z'),
      project_name=project.name,
      project_id=project.project_id,
      task_count=str(all_tasks_count),
      labeled_task_count=str(labeled_task_count),
      labeled_task_percent=labeled_task_percent,
      destroyed_task_count=str(destroyed_tasks_count),
      destroyed_task_percent=destroyed_task_percent,
      major_damage_task_count=str(major_damage_tasks_count),
      major_damage_task_percent=major_damage_task_percent,
      minor_damage_task_count=str(minor_damage_tasks_count),
      minor_damage_task_percent=minor_damage_task_percent,
      no_damage_task_count=str(no_damage_tasks_count),
      no_damage_task_percent=no_damage_task_percent,
      bad_example_task_count=str(bad_example_tasks_count),
      bad_example_task_percent=bad_example_task_percent,
      not_sure_task_count=str(not_sure_tasks_count),
      not_sure_task_percent=not_sure_task_percent,
  )


@app.route('/project/<path:project_id>/task_table', methods=['GET'])
@require_authentication
def task_table(project_id, user_email):
  """Returns a page with a paginated list of tasks for the project."""
  del user_email  # Unused

  page = request.args.get('page', 1, type=int)
  label = request.args.get('label', '', type=str)
  project = get_project_from_db(firestore_db, project_id)
  if not project:
    return abort(404, 'Invalid project id')
  if not any(option[0] == label for option in LABEL_FILTER_OPTIONS):
    print(f'Received invalid label: {label}')
    return abort(404, 'Invalid label filter')

  tasks_count = get_task_count_with_label(
      project.project_id, firestore_db, label
  )
  max_page = math.ceil(tasks_count / TASKS_PER_PAGE) if tasks_count > 0 else 1
  page = max(1, min(int(page), max_page))
  project_tasks = get_tasks(project.project_id, firestore_db, page, label)

  return render_template(
      'task_table.html',
      project_id=project.project_id,
      page=page,
      max_page=max_page,
      tasks_count=tasks_count,
      selected_label=label,
      label_filter_options=LABEL_FILTER_OPTIONS,
      tasks=project_tasks,
      tasks_per_page=TASKS_PER_PAGE,
      has_next_page=page < max_page,
  )


@app.route('/project/<path:project_id>/download_csv', methods=['GET'])
@require_authentication
def download_csv(project_id, user_email):
  """Returns a csv containing the labeled tasks for the project."""
  del user_email  # Unused

  print(f'Producing results csv for project: {project_id}')
  project = get_project_from_db(firestore_db, project_id)
  if not project:
    return abort(404, 'Invalid project id')

  filename = f'labeled_tasks_{project.project_id}'
  df = pd.DataFrame(
      [
          (
              task_dict.get('exampleId'),
              task_dict.get('label'),
              task_dict.get('labeler', ''),
              task_dict.get('startTime', ''),
              task_dict.get('submitTime', ''),
          )
          for task in firestore_db.collection(project.project_id).stream()
          if ((task_dict := task.to_dict()).get('label'))
      ],
      columns=[
          'example_id',
          'string_label',
          'labeler',
          'start_time',
          'end_time',
      ],
  )
  output_bytes = io.BytesIO()
  df.to_csv(output_bytes, index=False)
  output_bytes.seek(0)
  return send_file(
      output_bytes,
      mimetype='text/csv',
      download_name=f'{filename}.csv',
  )


@app.route('/project/<path:project_id>/delete', methods=['POST'])
@require_admin
def delete_project(project_id):
  """Deletes a project and cleans up all resources."""
  print(f'Deleting project: {project_id}')
  firestore_db.collection('projects').document(project_id).delete()

  project_tasks = firestore_db.collection(project_id).list_documents()
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=MAX_WORKER_THREAD_COUNT
  ) as executor:
    for task in project_tasks:
      print(f'Deleting task: {task.id}, in collection: {project_id}')
      executor.submit(task.delete)

  return jsonify(success=True)


@app.route('/project/<path:project_id>/resetlabels', methods=['POST'])
@require_admin
def reset_labels(project_id):
  """Reverts all labels for the given project."""
  print(f'Reverting all labels for project: {project_id}')
  project_tasks = (
      firestore_db.collection(project_id)
      .where(filter=firestore.FieldFilter('label', '!=', ''))
      .stream()
  )
  for task_snapshot in project_tasks:
    task = Task(
        task_snapshot.get('exampleId'),
        task_snapshot.get('preImage'),
        task_snapshot.get('postImage'),
    )
    task.write_to_firestore(project_id, firestore_db)
  return jsonify(success=True)


if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0', port=8080)
