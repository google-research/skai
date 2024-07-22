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
import functools
import logging
import time
from typing import Iterable, Sequence

import affine
import apache_beam as beam
import numpy as np
import pyproj
import rasterio
import rasterio.plot
import rasterio.warp
import rtree
import shapely.geometry

from skai import buildings
from skai import utils

Metrics = beam.metrics.Metrics
Polygon = shapely.geometry.Polygon

# Maximum size of a single patch read.
_MAX_PATCH_SIZE = 2048

# When splitting coordinates into bins for grouping, this is the number of
# degrees in longitude and latitude each bin spans. Since 1 degree ~ 111km,
# and we expect image resolutions around 0.5m, this means approximately 22K
# pixels per bin.
_BIN_SIZE_DEGREES = 0.1


@dataclasses.dataclass(order=True, frozen=True)
class _Window:
  """Information about a window to extract from a source raster.

  Attributes:
    window_id: Arbitrary string identifier for this window.
    column: Starting column of source window.
    row: Starting row of source window.
    width: Width of source window in pixels.
    height: Height of source window in pixels.
    source_crs: CRS of the source image.
    source_transform: Affine transform of the source window.
    target_crs: CRS of the target image.
    target_transform: Affine transform of the target window.
    target_image_size: Size of target image in pixels.
  """
  window_id: str
  column: int
  row: int
  width: int
  height: int

  source_crs: rasterio.crs.CRS | None = None
  source_transform: affine.Affine | None = None
  target_crs: rasterio.crs.CRS | None = None
  target_transform: affine.Affine | None = None
  target_image_size: int | None = None

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

  def extents(self) -> tuple[int, int, int, int]:
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

  def reproject(self, source_image: np.ndarray) -> np.ndarray:
    """Reprojects image into target CRS."""
    target_image = np.zeros(
        (3, self.target_image_size, self.target_image_size), dtype=np.uint8)
    rasterio.warp.reproject(
        source_image,
        target_image,
        src_transform=self.source_transform,
        src_crs=self.source_crs,
        dst_transform=self.target_transform,
        dst_crs=self.target_crs,
        resampling=rasterio.warp.Resampling.bilinear,
    )
    return target_image


@dataclasses.dataclass(frozen=True)
class _RasterBin:
  """A bin in a raster image.

  Each raster image is split into a number of bins to increase parallelism.
  Essentially, we overlay a grid over the raster where each cell spans
  _BIN_SIZE_DEGREES degrees in both the longitude and latitude axes. All windows
  falling into the same bin are grouped and processed sequentially, while the
  bins are processed in parallel.
  """

  raster_path: str
  x_bin_index: int
  y_bin_index: int


@dataclasses.dataclass(frozen=True)
class _RasterPoint:
  """A point in a raster image.
  """
  raster_path: str
  longitude: float
  latitude: float

  def get_bin(self) -> _RasterBin:
    return _RasterBin(
        self.raster_path,
        int(self.longitude // _BIN_SIZE_DEGREES),
        int(self.latitude // _BIN_SIZE_DEGREES),
    )


class _WindowGroup:
  """A group of windows, covered by an overall window.
  """

  def __init__(self, window: _Window):
    self.window = window
    self.members = [window]

  def add_window(self, other: _Window):
    self.window = self.window.expand(other)
    self.members.append(other)

  def extract_images(self, group_data: np.ndarray):
    for member in self.members:
      column_start = member.column - self.window.column
      column_end = column_start + member.width
      row_start = member.row - self.window.row
      row_end = row_start + member.height
      # Note that rasterio uses (channel, row, col) order instead of the usual
      # (row, col, channel) order.
      if column_end > group_data.shape[2] or row_end > group_data.shape[1]:
        raise ValueError('Member window exceeds group window bounds.')
      source_image = group_data[:, row_start:row_end, column_start:column_end]
      yield member.window_id, member.reproject(source_image)


@functools.cache
def _get_transformer(source_crs, target_crs) -> pyproj.Transformer:
  """Returns a cached Transformer object to optimize runtime."""
  return pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _compute_window(
    raster,
    window_id: str,
    longitude: float,
    latitude: float,
    target_image_size: int,
    target_resolution: float,
) -> _Window:
  """Computes window information.

  Algorithm for computing source window coordinates:

  1. Select the *target CRS*. This should be the UTM zone for the
     building centroid. Both the before and after images will be
     reprojected into this CRS.
  2. Convert longitude, latitude of the building centroid into the
     target CRS.
  3. Find the coordinates in the target CRS of the corners of the NxN
     pixel window we want to extract. This is easy in UTM coordinates
     since we know exactly how many meters corresponds to N pixels in
     the target resolution, so we only need to offset that many meters
     from the centroid to get the corners.
  4. Project the corner coordinates into the raster's native CRS (i.e.
     the source CRS). Then convert these coordinates into row, column
     coordinates in pixel space. This gives you the window to pass into
     the raster.read call. Call this the *source window*. Note that the
     size of the source window may be very different from the size of
     the target window we requested.
     - The corner coordinates will also give us the *source transform*,
       which tells us where the source window is located in the
       raster's native CRS.

  Later on in the pipeline, the computed source window will be read from the
  source raster. This image must then be reprojected into the target CRS in
  order to maintain consistency between the pre-disaster and post-disaster
  images.

  This reprojection requires the source transform and the *target transform*.
  The target transform is calculated from the source transform using the
  function rasterio.warp.calculate_default_transform, and is also included in
  the returned _Window object.

  Args:
    raster: Rasterio raster handle.
    window_id: An id for this window.
    longitude: Longitude of center of window.
    latitude: Latitude of center of window.
    target_image_size: Size of the target image in pixels.
    target_resolution: Resolution of the target image.

  Returns:
    All information needed to extract and reproject this window encapsulated
    in a _Window object.
  """
  source_crs = raster.crs

  # First find the corner coordinates of a [window_size] x [window_size] pixel
  # window in the target CRS. Always use UTM for the target CRS so that
  # rectangles are easy to derive.
  target_crs = rasterio.CRS.from_string(
      utils.convert_wgs_to_utm(longitude, latitude)
  )
  transformer = _get_transformer('EPSG:4326', target_crs)
  x, y = transformer.transform(longitude, latitude, errcheck=True)

  half_box_len_meters = (target_image_size / 2) * target_resolution
  target_left = x - half_box_len_meters
  target_right = x + half_box_len_meters
  target_bottom = y - half_box_len_meters
  target_top = y + half_box_len_meters

  # Map these coordinates back into the source CRS to get the window
  # coordinates in that CRS.
  source_transformer = _get_transformer(target_crs, source_crs)
  src_left, src_bottom = source_transformer.transform(
      target_left, target_bottom, errcheck=True)
  src_right, src_top = source_transformer.transform(
      target_right, target_top, errcheck=True)

  # Map the source coordinates into pixel space.
  #
  # The top-left corner of the image maps to row=0, col=0. So in coordinate
  # space, the largest y coordinate maps to the smallest row, and the smallest
  # y coordinate maps to the largest row. This is quite unintuitive, but is the
  # accepted convention.
  min_row, min_col = raster.index(src_left, src_top)
  max_row, max_col = raster.index(src_right, src_bottom)
  window_width = max_col - min_col
  window_height = max_row - min_row

  # The source window affine transform should correspond to the top-left
  # coordinates of the source window.
  source_transform = affine.Affine(
      a=raster.transform.a,
      b=raster.transform.b,
      c=src_left,
      d=raster.transform.d,
      e=raster.transform.e,
      f=src_top)

  # Compute the target transform.
  target_transform, _, _ = rasterio.warp.calculate_default_transform(
      source_crs,
      target_crs,
      width=window_width,
      height=window_height,
      left=src_left,
      bottom=src_bottom,
      top=src_top,
      right=src_right,
      resolution=target_resolution)

  return _Window(
      window_id=window_id,
      column=min_col,
      row=min_row,
      width=window_width,
      height=window_height,
      source_crs=source_crs,
      source_transform=source_transform,
      target_crs=target_crs,
      target_transform=target_transform,
      target_image_size=target_image_size,
  )


def _group_windows(
    raster_and_windows: tuple[_RasterBin, Sequence[_Window]],
) -> Iterable[tuple[str, _WindowGroup]]:
  """Groups overlapping windows to minimize data read from raster.

  The current implementation uses a greedy approach. It repeatedly chooses an
  arbitrary seed window, finds all other windows that intersect it, and groups
  them if the net savings (grouped window area - sum of individual window areas)
  is positive. The process ends when all windows have been grouped.

  Args:
    raster_and_windows: Raster path + list of all windows in that raster.

  Yields:
    Grouped windows.
  """
  windows = list(raster_and_windows[1])
  ungrouped = set(range(len(windows)))
  index = rtree.index.Index()
  for i, w in enumerate(windows):
    index.insert(i, w.extents())

  while ungrouped:
    seed = ungrouped.pop()
    group = _WindowGroup(windows[seed])
    changed = True
    while changed:
      changed = False
      overlaps = set(index.intersection(group.window.extents()))
      for i in overlaps.intersection(ungrouped):
        other = windows[i]
        new_window = group.window.expand(other)
        if (new_window.width > _MAX_PATCH_SIZE or
            new_window.height > _MAX_PATCH_SIZE):
          continue
        savings = group.window.area() + other.area() - new_window.area()
        if savings > 0:
          group.add_window(other)
          ungrouped.remove(i)
          changed = True
    Metrics.counter('skai', 'num_window_groups_created').inc()
    yield (raster_and_windows[0].raster_path, group)


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


def _generate_raster_points(
    raster_path: str, buildings_path: str
) -> Iterable[_RasterPoint]:
  """Generates raster x building centroids.

  This function will filter out buildings whose centroids are not in the bounds
  of the image.

  Args:
    raster_path: Path to raster image.
    buildings_path: Path to buildings file (usually a parquet file).

  Yields:
    A _RasterPoint for each building centroid in the bounds of the image.
  """
  coords_df = buildings.read_building_coordinates(buildings_path)
  raster = rasterio.open(raster_path)
  transformer = _get_transformer(raster.crs, 'EPSG:4326')
  left, top = transformer.transform(
      raster.bounds.left, raster.bounds.top, errcheck=True
  )
  right, bottom = transformer.transform(
      raster.bounds.right, raster.bounds.bottom, errcheck=True
  )
  for _, row in coords_df.iterrows():
    longitude = row['longitude']
    latitude = row['latitude']
    if (left <= longitude <= right) and (bottom <= latitude <= top):
      Metrics.counter('skai', 'num_raster_coords_pairs').inc()
      yield _RasterPoint(raster_path, longitude, latitude)


def get_rgb_indices(raster: rasterio.io.DatasetReader) -> list[int]:
  """Returns the indices of the RGB channels in the raster."""
  index_by_color = {}
  for i, colorinterp in enumerate(raster.colorinterp):
    index_by_color[colorinterp] = i + 1
  if rasterio.enums.ColorInterp.red not in index_by_color:
    raise ValueError('Raster does not have a red channel.')
  if rasterio.enums.ColorInterp.green not in index_by_color:
    raise ValueError('Raster does not have a green channel.')
  if rasterio.enums.ColorInterp.blue not in index_by_color:
    raise ValueError('Raster does not have a blue channel.')
  return [
      index_by_color[rasterio.enums.ColorInterp.red],
      index_by_color[rasterio.enums.ColorInterp.green],
      index_by_color[rasterio.enums.ColorInterp.blue],
  ]


class MakeWindow(beam.DoFn):
  """Beam function for creating a window from a raster and coordinates.

  Attributes:
    _rasters: Mapping from raster paths to raster handles.
    _target_patch_size: Size of target window in pixels.
    _target_resolution: Desired resolution of target window.
    _gdal_env: GDAL environment configuration.
  """

  def __init__(
      self,
      target_patch_size: int,
      target_resolution: float,
      gdal_env: dict[str, str],
  ):
    self._rasters = {}
    self._target_patch_size = target_patch_size
    self._target_resolution = target_resolution
    self._gdal_env = gdal_env

  def setup(self):
    self._rasterio_env = rasterio.Env(**self._gdal_env)

  def process(
      self, raster_point: _RasterPoint
  ) -> Iterable[tuple[_RasterBin, _Window]]:
    """Creates a window from a raster point.

    Args:
      raster_point: The input point.

    Yields:
      Tuples of (_RasterBin, _Window). The _RasterBin key value is needed for
      grouping by key in the next step of the beam pipeline.
    """
    with self._rasterio_env:
      if (raster := self._rasters.get(raster_point.raster_path)) is None:
        raster = rasterio.open(raster_point.raster_path)
        self._rasters[raster_point.raster_path] = raster

      window = _compute_window(
          raster,
          utils.encode_coordinates(
              raster_point.longitude, raster_point.latitude
          ),
          raster_point.longitude,
          raster_point.latitude,
          self._target_patch_size,
          self._target_resolution,
      )
      Metrics.counter('skai', 'num_windows_created').inc()
      yield (raster_point.get_bin(), window)


class ReadRasterWindowGroupFn(beam.DoFn):
  """A beam function that reads window groups from a raster image.

  Attributes:
    _rasters: Mapping from raster paths to raster handles.
    _gdal_env: GDAL environment configuration.
  """

  def __init__(self, gdal_env: dict[str, str]):
    self._rasters = {}
    self._gdal_env = gdal_env

    self._num_groups_read = Metrics.counter('skai', 'num_groups_read')
    self._num_windows_read = Metrics.counter('skai', 'num_windows_read')
    self._num_errors = Metrics.counter('skai', 'rasterio_error')
    self._read_time = Metrics.distribution('skai', 'raster_read_time_msec')

  def process(
      self, raster_and_group: tuple[str, _WindowGroup]
  ) -> Iterable[tuple[str, tuple[str, np.ndarray]]]:
    raster_path = raster_and_group[0]
    group = raster_and_group[1]

    start_time = time.time()
    if raster_path in self._rasters:
      raster = self._rasters[raster_path]
    else:
      with rasterio.Env(**self._gdal_env):
        raster = rasterio.open(raster_path)
      self._rasters[raster_path] = raster

    rgb_indices = get_rgb_indices(raster)
    raster_window = rasterio.windows.Window(
        group.window.column,
        group.window.row,
        group.window.width,
        group.window.height,
    )
    try:
      # Currently assumes that bands [1, 2, 3] of the input image are the RGB
      # channels.
      window_data = raster.read(
          indexes=rgb_indices,
          window=raster_window,
          boundless=True,
          fill_value=-1,
      )
    except (rasterio.errors.RasterioError, rasterio.errors.RasterioIOError):
      logging.exception('Raster read error')
      self._num_errors.inc()
      return
    finally:
      elapsed_millis = (time.time() - start_time) * 1000
      self._read_time.update(elapsed_millis)

    self._num_groups_read.inc()

    window_data = np.clip(window_data, 0, None)
    window_data = _convert_to_uint8(window_data)
    for window_id, channel_first_image in group.extract_images(window_data):
      image = rasterio.plot.reshape_as_image(channel_first_image)
      yield (window_id, (raster_path, image))


def extract_patches_from_rasters(
    pipeline: beam.Pipeline,
    buildings_path: str,
    raster_paths: list[str],
    patch_size: int,
    resolution: float,
    gdal_env: dict[str, str],
    stage_prefix: str) -> beam.PCollection:
  """Extracts patches from rasters.

  Args:
    pipeline: Beam pipeline.
    buildings_path: Path to building footprints file.
    raster_paths: Raster paths.
    patch_size: Desired size of output patches.
    resolution: Desired resolution of output patches.
    gdal_env: GDAL environment variables.
    stage_prefix: Unique prefix for Beam stage names.

  Returns:
    A collection whose elements are (id, (image path, window data)).
  """

  return (
      pipeline
      | stage_prefix + '_encode_raster_paths' >> beam.Create(raster_paths)
      | stage_prefix + '_generate_raster_points'
      >> beam.FlatMap(_generate_raster_points, buildings_path)
      | stage_prefix + '_make_windows'
      >> beam.ParDo(MakeWindow(patch_size, resolution, gdal_env))
      | stage_prefix + '_group_by_raster_bin' >> beam.GroupByKey()
      | stage_prefix + '_group_windows'
      >> beam.FlatMap(_group_windows)
      | stage_prefix + '_reshuffle' >> beam.Reshuffle()
      | stage_prefix + '_read_window_groups'
      >> beam.ParDo(ReadRasterWindowGroupFn(gdal_env))
  )


def get_raster_bounds(
    raster_path: str, gdal_env: dict[str, str]) -> Polygon:
  """Returns raster bounds as a shapely Polygon.

  Args:
    raster_path: Raster path.
    gdal_env: GDAL environment variables.

  Returns
    Bounds of raster in longitude, latitude polygon.
  """
  with rasterio.Env(**gdal_env):
    raster = rasterio.open(raster_path)
    transformer = pyproj.Transformer.from_crs(
        raster.crs, 'epsg:4326', always_xy=True
    )
    x1, y1 = transformer.transform(
        raster.bounds.left, raster.bounds.bottom, errcheck=True
    )
    x2, y2 = transformer.transform(
        raster.bounds.right, raster.bounds.top, errcheck=True
    )
    return shapely.geometry.box(x1, y1, x2, y2)
