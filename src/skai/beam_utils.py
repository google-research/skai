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
"""Utility functions for Apache Beam pipelines."""

import pathlib
import platform
import apache_beam as beam
import apache_beam.io.fileio as fileio
import apache_beam.options.value_provider as value_provider


PipelineOptions = beam.options.pipeline_options.PipelineOptions


class _BinarySink(fileio.FileSink):
  """Sink class for writing records of (filename, contents) in WriteToFiles."""

  def open(self, file_handle):
    self._file_handle = file_handle

  def write(self, record: tuple[str, bytes]):
    """Writes the binary content in the second element of the record to file."""
    self._file_handle.write(record[1])

  def flush(self):
    self._file_handle.flush()


def _file_naming_function(unused_window: str, unused_pane: str,
                          unused_shard_index: int, unused_total_shards: int,
                          unused_compression: str, destination: str) -> str:
  """File naming function for use with WriteToFiles.

  This function results in files that are simply named by the destination
  parameter and does not record any shard or window information in the file
  name.
  """
  return destination


def write_records_as_files(
    records: beam.PCollection,
    output_dir: str,
    temp_dir: str,
    stage_name: str):
  """Writes each record in a PCollection as individual files.

  Each record in the PCollection must be a tuple of (str, bytes) where the first
  element is the file name and the second element is the file contents.

  Args:
    records: PCollection to write.
    output_dir: Output directory.
    temp_dir: Temporary directory.
    stage_name: Beam stage name.

  Returns:
    Pipeline result.
  """
  temp_dir_value = value_provider.StaticValueProvider(str, temp_dir)
  return (records
          | stage_name >> fileio.WriteToFiles(
              path=output_dir,
              file_naming=_file_naming_function,
              destination=lambda record: record[0],
              temp_directory=temp_dir_value,
              sink=_BinarySink()))


def _get_setup_file_path():
  return str(pathlib.Path(__file__).parent.parent / 'setup.py')



def _get_dataflow_container_image() -> str | None:

  """Gets default dataflow image based on Python version.

  Returns:
    Dataflow container image path.
  """
  py_version = '.'.join(platform.python_version().split('.')[:2])
  if py_version in ['3.10', '3.11']:
    return f'gcr.io/skai-project-388314/skai-inference/dataflow_py_{py_version}_image'

  raise ValueError(
      f'Dataflow SDK supports Python versions 3.10+, not {py_version}'
  )


def get_pipeline_options(
    use_dataflow: bool,
    job_name: str,
    project: str,
    region: str,
    temp_dir: str,
    max_workers: int,
    worker_service_account: str | None,
    worker_type: str | None,
    worker_machine_type: str | None,
    dataflow_service_options: str
) -> PipelineOptions:
  """Returns dataflow pipeline options.

  Args:
    use_dataflow: Whether to use Dataflow or local runner.
    job_name: Name of Dataflow job.
    project: GCP project.
    region: GCP region.
    temp_dir: Temporary data location.
    max_workers: Maximum number of Dataflow workers.
    worker_service_account: Email of the service account will launch workers.
        If None, uses the project's default Compute Engine service account
        (<project-number>-compute@developer.gserviceaccount.com).
    worker_type: Dataflow worker type.

  Returns:
    Dataflow options.
  """
  if not use_dataflow:
    return PipelineOptions.from_dictionary({
        'runner': 'DirectRunner',
        'direct_num_workers': 10,
        'direct_running_mode': 'multi_processing',
    })

  if not project or not region:
    raise ValueError(
        'cloud_project and cloud_region must be specified when using '
        'Dataflow.')

  options = {
      'job_name': job_name,
      'project': project,
      'region': region,
      'temp_location': temp_dir,
      'runner': 'DataflowRunner',
      'experiment': 'use_runner_v2',
      'sdk_container_image': _get_dataflow_container_image(),
      'sdk_location': 'container',
      'setup_file': _get_setup_file_path(),
      'max_num_workers': max_workers,
      'use_public_ips': False,  # Avoids hitting public ip quota bottleneck.
      'worker_machine_type': worker_machine_type,
      'disk_size_gb': 200,
      'dataflow_service_options':dataflow_service_options
  }
  if worker_service_account:
    options['service_account_email'] = worker_service_account
  if worker_type:
    options['worker_type'] = worker_type
  return PipelineOptions.from_dictionary(options)
