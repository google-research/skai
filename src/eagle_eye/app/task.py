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

"""Task data access object, with a method for writing to firestore."""

import time


TASK_LABEL_MAP = {
    'destroyed': 'Destroyed',
    'major_damage': 'Major Damage',
    'minor_damage': 'Minor Damage',
    'no_damage': 'No Damage',
    'bad_example': 'Bad Example',
    'not_sure': "I'm Not Sure",
}


def get_task_from_db(firestore_db, project_id, example_id):
  """Returns a Task, read from firestore."""
  task_firestore = (
      firestore_db.collection(project_id).document(example_id).get()
  )
  task_dict = task_firestore.to_dict()
  return Task(
      example_id,
      task_dict.get('preImage'),
      task_dict.get('postImage'),
      task_dict.get('label'),
      task_dict.get('labeler', ''),
  )


def filter_tasks_by_label(tasks, label, should_match=True):
  """Returns the subset of tasks that match (or do not match, if should_match is false) the given label."""
  if should_match:
    return [task for task in tasks if task.get('label') == label]
  else:
    return [task for task in tasks if task.get('label') != label]


def normalize_image_url_suffix(image_url):
  """Normalize an image url prefixed by either gs:// or /bigstore/.

  For either image url format, return just the image url suffix
  i.e. the portion after either valid prefix.
  """
  if image_url.startswith('gs://'):
    return image_url[5:]
  elif image_url.startswith('/bigstore/'):
    return image_url[10:]
  else:
    raise ValueError(
        'Failed to normalize an image url, does not conform to gs:// or'
        f' /bigstore/ prefix: {image_url}'
    )


class Task:
  """A Task data type."""

  def __init__(
      self,
      example_id,
      pre_image,
      post_image,
      label='',
      labeler='',
      start_time=None,
      submit_time=None,
  ):
    """Initializes a Task object."""
    self.example_id = example_id
    self.pre_image = pre_image
    self.post_image = post_image
    self.label = label
    self.labeler = labeler
    self.start_time = start_time
    self.submit_time = submit_time

  def write_to_firestore(self, project_id, firestore_db):
    """Writes the Task to the given project_id firestore collection."""
    start_time = time.perf_counter()
    firestore_format = {
        'projectId': project_id,
        'exampleId': self.example_id,
        'preImage': self.pre_image,
        'postImage': self.post_image,
        'label': self.label,
        'labeler': self.labeler,
        'startTime': self.start_time,
        'submitTime': self.submit_time,
    }
    firestore_db.collection(project_id).document(self.example_id).set(
        firestore_format
    )
    end_time = time.perf_counter()
    print(
        f'wrote task with example_id: {self.example_id} to firestore, in'
        f' {end_time - start_time} sec'
    )
