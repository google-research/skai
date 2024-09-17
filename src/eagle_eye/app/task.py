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
  return Task(
      example_id,
      task_firestore.get('preImage'),
      task_firestore.get('postImage'),
      task_firestore.get('label'),
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

  def __init__(self, example_id, pre_image, post_image, label=''):
    """Initializes a Task object."""
    self.example_id = example_id
    self.pre_image = pre_image
    self.post_image = post_image
    self.label = label

  def write_to_firestore(self, project_id, firestore_db):
    """Writes the Task to the given project_id firestore collection."""
    start_time = time.perf_counter()
    firestore_format = {
        'projectId': project_id,
        'exampleId': self.example_id,
        'preImage': self.pre_image,
        'postImage': self.post_image,
        'label': self.label,
    }
    firestore_db.collection(project_id).document(self.example_id).set(
        firestore_format
    )
    end_time = time.perf_counter()
    print(
        f'wrote task with example_id: {self.example_id} to firestore, in'
        f' {end_time - start_time} sec'
    )
