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

from typing import Tuple

import apache_beam as beam
import apache_beam.io.fileio as fileio
import apache_beam.options.value_provider as value_provider


class _BinarySink(fileio.FileSink):
  """Sink class for writing records of (filename, contents) in WriteToFiles."""

  def open(self, file_handle):
    self._file_handle = file_handle

  def write(self, record: Tuple[str, bytes]):
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
