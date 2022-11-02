# Copyright 2022 Google LLC
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
"""Library for reading raster images."""

import dataclasses
import logging
import time
from typing import Dict, Iterable, List, Tuple

import apache_beam as beam
import cv2
import numpy as np
import pyproj
import rasterio
import rasterio.plot
import rtree

Metrics = beam.metrics.Metrics

# Maximum size of a single patch read.
_MAX_PATCH_SIZE = 2048


@dataclasses.dataclass(order=True, frozen=True)
class _Window:
  """Class representing a window in pixel coordinates.

  Attributes:
    window_id: Arbitrary string identifier for this window.
    column: Starting column of window.
    row: Starting row of window.
    width: Width of window in pixels.
    height: Height of window in pixels.
  """
  window_id: str
  column: int
  row: int
  width: int
  height: int

  def expand(self, other):
    """Returns a new window that covers this window and another one.

    Args:
      other: The other window.

    Returns:
      A new window.
    """
    x1 = min(self.column, other.column)
    y1 = min(self.row, other.row)
    x2 = max(self.column + self.width, other.column + other.width)
    y2 = max(self.row + self.height, other.row + other.height)
    return _Window('', x1, y1, x2 - x1, y2 - y1)

  def extents(self) -> Tuple[int, int, int, int]:
    """Return the extents of the window.

    Returns:
      A tuple (min col, min row, max col, max row).
    """
    return (self.column,
            self.row,
            self.column + self.width,
            self.row + self.height)

  def area(self) -> int:
    return self.width * self.height


class _WindowGroup:
  """A group of windows, covered by an overall window.
  """

  def __init__(self, window: _Window):
    self.window = window
    self.members = [window]

  def add_window(self, other: _Window):
    self.window = self.window.expand(other)
    self.members.append(other)

  def extract_members(self, group_data: np.ndarray):
    for i, member in enumerate(self.members):
      column_start = member.column - self.window.column
      column_end = column_start + member.width
      row_start = member.row - self.window.row
      row_end = row_start + member.height
      if column_end > group_data.shape[1] or row_end > group_data.shape[0]:
        raise ValueError('Member window exceeds group window bounds.')
      yield i, group_data[row_start:row_end, column_start:column_end, :]


def _in_bounds(x: float, y: float, bounds) -> bool:
  return (bounds.left <= x <= bounds.right and
          bounds.bottom <= y <= bounds.top)


def _get_windows(
    raster,
    window_size: int,
    coordinates: List[Tuple[str, float, float]]) -> List[_Window]:
  """Computes windows in pixel coordinates for a raster.

  Args:
    raster: Input raster.
    window_size: Size of windows to generate. Windows are always square.
    coordinates: Longitude, latitude centroids of windows.

  Returns:
    List of windows.
  """
  transformer = pyproj.Transformer.from_crs(
      'epsg:4326', raster.crs, always_xy=True)
  windows = []
  for example_id, longitude, latitude in coordinates:
    x, y = transformer.transform(longitude, latitude, errcheck=True)
    if not _in_bounds(x, y, raster.bounds):
      continue
    row, col = raster.index(x, y)
    half_size = window_size // 2
    col_off = col - half_size
    row_off = row - half_size
    windows.append(
        _Window(example_id, col_off, row_off, window_size, window_size))
  return windows


def _group_windows(windows: List[_Window]) -> List[_WindowGroup]:
  """Groups overlapping windows to minimize data read from raster.

  The current implementation uses a greedy approach. It repeatedly chooses an
  arbitrary seed window, finds all other windows that intersect it, and groups
  them if the net savings (grouped window area - sum of individual window areas)
  is positive. The process ends when all windows have been grouped.

  Args:
    windows: A list of windows to group.

  Returns:
    Grouped windows.
  """

  groups = []
  ungrouped = set(range(len(windows)))
  index = rtree.index.Index()
  for i, w in enumerate(windows):
    index.insert(i, w.extents())

  while ungrouped:
    seed = ungrouped.pop()
    index.delete(seed, windows[seed].extents())
    group = _WindowGroup(windows[seed])
    changed = True
    while changed:
      changed = False
      overlaps = index.intersection(group.window.extents())
      for i in overlaps:
        other = windows[i]
        new_window = group.window.expand(other)
        if (new_window.width > _MAX_PATCH_SIZE or
            new_window.height > _MAX_PATCH_SIZE):
          continue
        savings = group.window.area() + other.area() - new_window.area()
        if savings > 0:
          group.add_window(other)
          ungrouped.remove(i)
          index.delete(i, windows[i].extents())
          changed = True
    groups.append(group)
  return groups


def _convert_to_uint8(image: np.ndarray) -> np.ndarray:
  """Converts an image to uint8.

  This function currently only handles converting from various integer types to
  uint8, with range checks to make sure the casting is safe. If needed, this
  function can be adapted to handle float types.

  Args:
    image: Input image array.

  Returns:
    uint8 array.

  """
  if not np.issubdtype(image.dtype, np.integer):
    raise TypeError(f'Image type {image.dtype} not supported.')
  if np.min(image) < 0 or np.max(image) > 255:
    raise ValueError(
        f'Pixel values have a range of {np.min(image)}-{np.max(image)}. '
        'Only 0-255 is supported.')
  return image.astype(np.uint8)


def _get_raster_resolution_in_meters(raster) -> float:
  """Covert different resolution unit into meters.

  Args:
    raster: Input raster.
  Returns:
    Resolution in meters.
  Raises:
    ValueError: CRS error
  """
  if not np.isclose(raster.res[0], raster.res[1], rtol=0.0001):
    raise ValueError(
        f'Expecting identical x and y resolutions, got {raster.res[0]}, {raster.res[1]}'
    )
  crs = raster.crs
  try:
    meter_conversion_factor = crs.linear_units_factor[1]
  except rasterio.errors.CRSError as e:
    if crs.to_epsg() == 4326:
      # Raster resolution is expressed in degrees lon/lat. Convert to
      # meters with approximation that 1 degree ~ 111km.
      meter_conversion_factor = 111000
    else:
      raise ValueError(
          f'No linear units factor or unsupported EPSG code, got {e}') from e
  return raster.res[0] * meter_conversion_factor


def _resample_image(image: np.ndarray, patch_size: int) -> np.ndarray:
  return cv2.resize(
      image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)


class ReadRasterWindowGroupFn(beam.DoFn):
  """A beam function that reads window groups from a raster image.

  Attributes:
    _raster_path: Path to raster.
    _raster: Reference to the raster.
    _target_patch_size: Desired size of output patches.
    _tag: String identifier for output patches.
    _gdal_env: GDAL environment configuration.
  """

  def __init__(self,
               raster_path: str,
               target_patch_size: int,
               tag: str,
               gdal_env: Dict[str, str]):
    self._raster = None
    self._raster_path = raster_path
    self._target_patch_size = target_patch_size
    self._tag = tag
    self._gdal_env = gdal_env

    self._num_groups_read = Metrics.counter('skai', 'num_groups_read')
    self._num_windows_read = Metrics.counter('skai', 'num_windows_read')
    self._num_errors = Metrics.counter('skai', 'rasterio_error')
    self._read_time = Metrics.distribution('skai', 'raster_read_time_msec')

  def setup(self) -> None:
    """Opens image raster.

    This simply creates a raster placeholder in memory. It doesn't actually read
    the raster data from disk.
    """
    with rasterio.Env(**self._gdal_env):
      self._raster = rasterio.open(self._raster_path)

  def process(
      self,
      group: _WindowGroup) -> Iterable[Tuple[str, Tuple[str, np.ndarray]]]:
    raster_window = rasterio.windows.Window(group.window.column,
                                            group.window.row,
                                            group.window.width,
                                            group.window.height)
    start_time = time.time()
    try:
      # Currently assumes that bands [1, 2, 3] of the input image are the RGB
      # channels.
      window_data = self._raster.read(
          indexes=[1, 2, 3], window=raster_window, boundless=True,
          fill_value=-1)
    except (rasterio.errors.RasterioError, rasterio.errors.RasterioIOError):
      logging.exception('Raster read error')
      self._num_errors.inc()
      return
    finally:
      elapsed_millis = (time.time() - start_time) * 1000
      self._read_time.update(elapsed_millis)

    self._num_groups_read.inc()

    window_data = np.clip(window_data, 0, None)
    window_data = rasterio.plot.reshape_as_image(window_data)
    window_data = _convert_to_uint8(window_data)
    for i, member_data in group.extract_members(window_data):
      resampled = _resample_image(member_data, self._target_patch_size)
      yield group.members[i].window_id, (self._tag, resampled)


def extract_patches_from_raster(
    pipeline,
    raster_path: str,
    tag: str,
    patch_size: int,
    resolution: float,
    coordinates: List[Tuple[str, float, float]],
    gdal_env: Dict[str, str],
    stage_prefix: str) -> beam.PCollection:
  """Extracts patches from a raster.

  Args:
    pipeline: Beam pipeline.
    raster_path: Path to raster.
    tag: String identifier for the generated patches. e.g. "before" or "after".
    patch_size: Desired size of output patches.
    resolution: Desired resolution of output patches.
    coordinates: List of patch centroids as longitude, latitude coordinates.
    gdal_env: GDAL environment variables.
    stage_prefix: Unique prefix for Beam stage names.

  Returns:
    A collection whose elements are (id, tag, window data).
  """

  with rasterio.Env(**gdal_env):
    raster = rasterio.open(raster_path)
    raster_res = _get_raster_resolution_in_meters(raster)
    scale_factor = resolution / raster_res
    window_size = int(patch_size * scale_factor)
    windows = _get_windows(raster, window_size, coordinates)

  logging.info('Grouping "%s" windows to optimize throughput. '
               'This may take a couple of minutes.', tag)
  window_groups = _group_windows(windows)
  logging.info('Grouped %d windows into %d groups.', len(windows),
               len(window_groups))
  return (pipeline
          | stage_prefix + '_make_window_groups' >> beam.Create(window_groups)
          | stage_prefix + '_read_window_groups' >> beam.ParDo(
              ReadRasterWindowGroupFn(raster_path, patch_size, tag, gdal_env)))
