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

"""Project data access object and Project utility methods."""

import time
from google.cloud.firestore_v1 import aggregation
from google.cloud.firestore_v1.base_query import FieldFilter

TASKS_PER_PAGE = 25

LABELED = 'labeled'
NOT_LABELED = 'not_labeled'


def get_project_from_db(firestore_db, project_id):
  """Returns a Project, read from firestore, or None if it does not exist."""
  project_firestore = (
      firestore_db.collection('projects').document(project_id).get()
  )
  if not project_firestore.exists:
    print(f'Project with id: {project_id} does not exist')
    return None
  return Project(
      project_id,
      project_firestore.get('creation_time'),
      project_firestore.get('name'),
      project_firestore.get('image_metadata_path'),
  )


def _get_task_query(project_id, firestore_db, label=None):
  query = firestore_db.collection(project_id)
  if label:
    if label == NOT_LABELED:
      query = query.where(filter=FieldFilter('label', '==', ''))
    elif label == LABELED:
      query = query.where(filter=FieldFilter('label', '!=', ''))
    else:
      query = query.where(filter=FieldFilter('label', '==', label))
  return query


def get_tasks(project_id, firestore_db, page=1, label=None):
  """Returns a paginated list of tasks in the project."""
  query = _get_task_query(project_id, firestore_db, label)
  return query.limit(TASKS_PER_PAGE).offset((page - 1) * TASKS_PER_PAGE).get()


def get_task_count(project_id, firestore_db):
  """Returns the total task count for the project."""
  query = firestore_db.collection(project_id)
  results = query.count().get()
  return int(results[0][0].value)


def get_task_count_with_label(project_id, firestore_db, label):
  """Returns the number of tasks in the project matching the given label.

  If label is None, returns the total task count.
  """
  if not label:
    return get_task_count(project_id, firestore_db)
  query = _get_task_query(project_id, firestore_db, label)
  aggregate_query = aggregation.AggregationQuery(query)
  results = aggregate_query.count().get()
  return int(results[0][0].value)


def get_completed_task_count(project_id, firestore_db):
  """Returns the number of tasks with any label."""
  query = _get_task_query(project_id, firestore_db, LABELED)
  aggregate_query = aggregation.AggregationQuery(query)
  results = aggregate_query.count().get()
  return int(results[0][0].value)


class Project:
  """A Project data type.

  On initial creation, project_id and creation_time should be set to
  None. These fields will be populated after the first write to
  firestore, after which the id will be populated with the
  auto-generated id and the timestamp with the time of the first
  write.
  """

  def __init__(self, project_id, creation_time, name, image_metadata_path):
    self.project_id = project_id
    self.creation_time = creation_time
    self.name = name
    self.image_metadata_path = image_metadata_path

  def write_to_firestore(self, firestore_db):
    start_time = time.perf_counter()
    if self.project_id is None:
      # When writing a project to firestore for the first time, add an
      # empty object and store the firestore auto-generated id and
      # creation time.
      creation_time, new_project_ref = firestore_db.collection('projects').add(
          {}
      )
      self.project_id = new_project_ref.id
      self.creation_time = creation_time

    firestore_format = {
        'name': self.name,
        'creation_time': self.creation_time,
        'image_metadata_path': self.image_metadata_path,
    }
    firestore_db.collection('projects').document(self.project_id).set(
        firestore_format
    )
    end_time = time.perf_counter()
    print(
        f'wrote project with id: {self.project_id} to firestore, in '
        f'{end_time - start_time} sec'
    )
