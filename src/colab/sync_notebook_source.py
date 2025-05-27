r"""Tool to sync notebooks with their source py files using Jupytext.

Example invocations:

Convert Python source file to IPython notebook.

$ python sync_notebook_source.py update-ipynb \
    skai_assessment_notebook.py skai_assessment_notebook.ipynb


Convert IPython notebook to Python source file.

$ python sync_notebook_source.py update-python \
    skai_assessment_notebook.py skai_assessment_notebook.ipynb
"""

import ast
from collections.abc import Sequence
import os
import sys

from absl import app
import jupytext
import jupytext.config


# Currently, these are parameters from the skai assessment notebook. Can add
# more parameter sets later, or just use the same common set of parameters
# across different notebooks.
_PARAMS = {
    'GCP_PROJECT': '',
    'GCP_LOCATION': '',
    'GCP_BUCKET': '',
    'GCP_SERVICE_ACCOUNT': '',
    'CLOUD_RUN_PROJECT': '',
    'CLOUD_RUN_LOCATION': '',
    'SERVICE_ACCOUNT_KEY': '',
    'BUILDING_SEGMENTATION_MODEL_PATH': '',
    'BUILDINGS_METHOD': 'open_buildings',
    'USER_BUILDINGS_FILE': '',
    'ASSESSMENT_NAME': '',
    'EVENT_DATE': '',
    'OUTPUT_DIR': '',
    'BEFORE_IMAGE_0': '',
    'BEFORE_IMAGE_1': '',
    'BEFORE_IMAGE_2': '',
    'BEFORE_IMAGE_3': '',
    'BEFORE_IMAGE_4': '',
    'BEFORE_IMAGE_5': '',
    'BEFORE_IMAGE_6': '',
    'BEFORE_IMAGE_7': '',
    'BEFORE_IMAGE_8': '',
    'BEFORE_IMAGE_9': '',
    'AFTER_IMAGE_0': '',
    'AFTER_IMAGE_1': '',
    'AFTER_IMAGE_2': '',
    'AFTER_IMAGE_3': '',
    'AFTER_IMAGE_4': '',
    'AFTER_IMAGE_5': '',
    'AFTER_IMAGE_6': '',
    'AFTER_IMAGE_7': '',
    'AFTER_IMAGE_8': '',
    'AFTER_IMAGE_9': '',
    'DAMAGE_SCORE_THRESHOLD': 0.9,
    'MAX_LABELING_IMAGES': 500,
    'IMAGE_RESOLUTION': 0.5,
    'MINOR_IS_DAMAGED': False,
    'SAMPLING_METHOD': 'uniform',
}


def replace_param_assignment(line: str, param_values: dict[str, str]) -> str:
  """Replaces the RHS of an assignment statement with a specified value.

  Args:
    line: A line of code.
    param_values: Mapping from variable names to the values to assign
      to them.

  Returns:
    The altered line of code if it's a valid parameter assignment, otherwise
    return the unaltered code.
  """
  code, comment_char, comment = line.partition('#')
  if '@param' not in comment:
    return line

  tree = ast.parse(code)
  if len(tree.body) > 1:
    # The code consists of multiple statements, it's not a parameter assignment.
    return line
  node = tree.body[0]
  if isinstance(node, ast.Assign) and len(node.targets) == 1:
    param_name = node.targets[0].id
    if (new_value := param_values.get(param_name)) is not None:
      node.value.s = new_value
      new_code = ast.unparse(tree)
      new_line = f'{new_code}  {comment_char}{comment}'
      print(f'Modified line: {new_line}')
      return new_line
  return line


def extract_param_value(
    line: str, param_names: set[str]) -> tuple[str | None, str | None]:
  """Extracts the value of a parameter, if the code is a valid param assignment.

  Args:
    line: A line of code.
    param_names: Set of parameter names to extract.

  Returns:
    A tuple (parameter name, value), if the code is a param assignment and the
    parameter is one of the specified ones.
  """
  code, _, comment = line.partition('#')
  if '@param' not in comment:
    return None, None
  tree = ast.parse(code)
  if len(tree.body) > 1:
    # The code consists of multiple statements, it's not a parameter assignment.
    return None, None
  node = tree.body[0]
  if (isinstance(node, ast.Assign) and
      (len(node.targets) == 1) and
      (node.targets[0].id in param_names)):
    return (node.targets[0].id, node.value.s)
  return None, None


def replace_params(notebook, param_values: dict[str, str]):
  """Replaces the values of parameter assignments in a notebook.

  Parameter assignments are those lines of code that contains a comment with
  the string "@param" (i.e. form fields in Colab).

  Args:
    notebook: The notebook object.
    param_values: Mapping from variable names to the values to assign
      to them.
  """
  for cell in notebook.cells:
    edited_lines = []
    for line in cell.source.split('\n'):
      edited_lines.append(replace_param_assignment(line, param_values))
    cell.source = '\n'.join(edited_lines)


def extract_param_values(notebook, param_names: set[str]) -> dict[str, str]:
  """Extracts parameter settings from a notebook."""
  param_values = {}
  for cell in notebook.cells:
    for line in cell.source.split('\n'):
      name, value = extract_param_value(line, param_names)
      if name is not None:
        param_values[name] = value
  return param_values


def write_python_source(notebook, output_path: str):
  config = jupytext.config.JupytextConfiguration()
  config.cell_metadata_filter = '-all,cellView'
  jupytext.write(notebook, output_path, fmt='py:percent', config=config)


def write_ipynb(notebook, output_path: str):
  jupytext.write(notebook, output_path, fmt='notebook')


def update_ipynb(
    python_source_path: str,
    ipynb_path: str,
    params_to_preserve: set[str]):
  """Updates IPython notebook with .py source code.

  Preserves parameter settings in the IPython notebook so that user-specific
  settings are not erased.

  Args:
    python_source_path: Path to .py file.
    ipynb_path: Path to IPython notebook.
    params_to_preserve: Parameter names to preserve.
  """
  if os.path.exists(ipynb_path):
    target_notebook = jupytext.read(ipynb_path)
    param_values = extract_param_values(target_notebook, params_to_preserve)
  else:
    param_values = {}

  source_notebook = jupytext.read(python_source_path)
  replace_params(source_notebook, param_values)
  write_ipynb(source_notebook, ipynb_path)


def update_python_source(
    python_source_path: str,
    ipynb_path: str,
    default_values: dict[str, str]):
  """Updates .py source code with IPython notebook.

  Parameter values are reset to defaults to avoid leaking user-specific settings
  to the source of truth.

  Args:
    python_source_path: Path to .py file.
    ipynb_path: Path to IPython notebook.
    default_values: Default parameter values.
  """
  notebook = jupytext.read(ipynb_path)
  replace_params(notebook, default_values)
  write_python_source(notebook, python_source_path)


def main(argv: Sequence[str]) -> None:
  if len(argv) != 4:
    raise app.UsageError(
        f'{argv[0]} <update-python|update-ipynb> <py file> <ipynb file>'
    )
  if argv[1] == 'update-python':
    update_python_source(argv[2], argv[3], _PARAMS)
  elif argv[1] == 'update-ipynb':
    update_ipynb(argv[2], argv[3], set(_PARAMS.keys()))
  else:
    print(f'Unknown command: "{argv[1]}"')
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
