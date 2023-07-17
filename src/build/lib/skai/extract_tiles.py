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
"""Extracts equal-sized tiles from a GeoTIFF."""

import dataclasses
from typing import Any, Dict, Iterable, Tuple

import affine
import apache_beam as beam
import numpy as np
import pyproj
import rasterio
import rasterio.plot
from skai import utils
import tensorflow as tf
from skai import extract_tiles_constants

Example = tf.train.Example
PipelineOptions = beam.options.pipeline_options.PipelineOptions
# Extents of a window in a raster, represented as (x, y, width, height).
ExtentsType = Tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True, order=True)
class Tile:
  """Class for holding information about an image tile.

  Attributes:
    x: x pixel coordinate of lower-left corner of the tile.
    y: y pixel coordinate of lower-left corner of the tile.
    width: Tile width in pixels.
    height: Tile height in pixels.
    margin_size: Tile margin in pixels.
    column: The column of this tile in the grid covering the source image.
    row: The row of this tile in the grid covering the source image.
  """
  x: int
  y: int
  width: int
  height: int
  margin_size: int
  column: int
  row: int


def _create_tile_example(image: np.array, tile: Tile, crs: str,
                         affine_transform: affine.Affine) -> Example:
  """Creates an Example for the building detection model from an image tile.

  The x and y pixel offsets of the tile in the original image is encoded in the
  example so that tile pixel coordinates can be translated into image pixel
  coordinates.

  The tile column and row information are added to the example for deduplication
  later in the building detection pipeline.

  Args:
    image: Image of the tile as an numpy array.
    tile: Tile information.
    crs: Coordinate reference system of the image this tile came from.
    affine_transform: Affine transform of the image this tile came from.

  Returns:
    Tensorflow example.
  """
  example = Example()
  utils.add_int64_feature(extract_tiles_constants.IMAGE_HEIGHT, image.shape[0],
                          example)
  utils.add_int64_feature(extract_tiles_constants.IMAGE_WIDTH, image.shape[1],
                          example)
  utils.add_int64_feature(extract_tiles_constants.MARGIN_SIZE, tile.margin_size,
                          example)
  utils.add_int64_feature(extract_tiles_constants.X_OFFSET, tile.x, example)
  utils.add_int64_feature(extract_tiles_constants.Y_OFFSET, tile.y, example)
  utils.add_int64_feature(extract_tiles_constants.TILE_COL, tile.column,
                          example)
  utils.add_int64_feature(extract_tiles_constants.TILE_ROW, tile.row, example)
  utils.add_bytes_feature(extract_tiles_constants.IMAGE_FORMAT, b'png',
                          example)
  utils.add_bytes_feature(extract_tiles_constants.IMAGE_ENCODED,
                          tf.io.encode_png(image).numpy(), example)
  utils.add_bytes_feature(extract_tiles_constants.CRS, crs.encode(), example)

  transform_tuple = tuple(affine_transform)
  if transform_tuple[6:] != (0.0, 0.0, 1.0):
    raise ValueError('Last 3 values in affine transform should be (0, 0, 1), '
                     f'but got {transform_tuple[6:]}')
  utils.add_float_list_feature(extract_tiles_constants.AFFINE_TRANSFORM,
                               transform_tuple[:6], example)
  return example


def _get_pixel_bounds_for_aoi(
    image: Any, aoi: Any) -> Tuple[int, int, int, int]:
  """Gets the pixel coordinates of the rectangle spanned by the AOI.

  Args:
    image: Rasterio image.
    aoi: Area of interest as a shapely geometry in long/lat coordinates.

  Returns:
    Rectangle coordinates: (min_col, min_row, max_col, max_row).
  """
  x1, y1, x2, y2 = aoi.bounds
  transformer = pyproj.Transformer.from_crs(
      'epsg:4326', image.crs.to_string().lower(), always_xy=True)
  tx1, ty1 = transformer.transform(x1, y1, errcheck=True)
  tx2, ty2 = transformer.transform(x2, y2, errcheck=True)

  # Order of rows are reversed because row 0 is at the top of the image and
  # increases going down.
  max_row, min_col = image.index(tx1, ty1)
  min_row, max_col = image.index(tx2, ty2)

  # Clamp values to extents of image.
  min_col = max(0, min_col)
  min_row = max(0, min_row)
  max_col = min(image.width, max_col)
  max_row = min(image.height, max_row)

  assert 0 <= min_col <= max_col <= image.width, (
      f'Expecting 0 <= {min_col} <= {max_col} <= image.width')
  assert 0 <= min_row <= max_row <= image.height, (
      f'Expecting 0 <= {min_row} <= {max_row} <= image.height')

  return (min_col, min_row, max_col, max_row)


def get_tiles(x_min: int, y_min: int, x_max: int, y_max: int,
              tile_size: int, margin: int) -> Iterable[Tile]:
  """Generates a set of tiles that would completely cover a rectangle.

  Args:
    x_min: Minimum x coordinate.
    y_min: Minimum y coordinate.
    x_max: Maximum x coordinate.
    y_max: Maximum y coordinate.
    tile_size: Size of each tile in pixels.
    margin: Size of the margin for each tile in pixels.

  Yields:
    A grid of tiles that covers the rectangle.
  """
  if x_min < 0:
    raise ValueError(f'Minimum x value must be non-negative, got {x_min}')
  if y_min < 0:
    raise ValueError(f'Minimum y value must be non-negative, got {y_min}')
  for col, x in enumerate(range(x_min + margin, x_max - margin, tile_size)):
    for row, y in enumerate(range(y_min + margin, y_max - margin, tile_size)):
      x_start = x - margin
      x_end = x + tile_size + margin
      width = x_end - x_start
      y_start = y - margin
      y_end = y + tile_size + margin
      height = y_end - y_start
      yield Tile(x_start, y_start, width, height, margin, col, row)


def get_tiles_for_aoi(image_path: str,
                      aoi: Any,
                      tile_size: int,
                      margin: int,
                      gdal_env: Dict[str, str]) -> Iterable[Tile]:
  """Generates a set of tiles that would completely cover an AOI.

  Args:
    image_path: Path of the GeoTIFF image.
    aoi: Area of interest as a shapely geometry.
    tile_size: Size of each tile in pixels.
    margin: Size of the margin for each tile in pixels.
    gdal_env: GDAL environment configuration.

  Yields:
    A grid of tiles that covers the AOI.
  """
  with rasterio.Env(**gdal_env):
    image = rasterio.open(image_path)
    x_min, y_min, x_max, y_max = _get_pixel_bounds_for_aoi(image, aoi)
    yield from get_tiles(x_min, y_min, x_max, y_max, tile_size, margin)


class ExtractTilesAsExamplesFn(beam.DoFn):
  """Extracts tiles from a GeoTIFF and converts them into TF Examples."""

  def __init__(self, input_path: str, gdal_env: Dict[str, str]) -> None:
    self._input_path = input_path
    self._gdal_env = gdal_env

  def setup(self) -> None:
    with rasterio.Env(**self._gdal_env):
      self._input_file = rasterio.open(self._input_path)
      self._crs = self._input_file.crs.to_string()
      self._affine_transform = self._input_file.transform

  def process(self, tile: Tile) -> Iterable[bytes]:
    """Extract a tile from the source image and encode it as an Example.

    Args:
      tile: The tile to extract.

    Yields:
      Serialized Tensorflow Example containing tile data.
    """
    if tile.x < 0 or tile.y < 0:
      raise ValueError(f'Tile extents out of bounds: x={tile.x}, y={tile.y}')

    window = rasterio.windows.Window(tile.x, tile.y, tile.width, tile.height)
    window_data = self._input_file.read(
        window=window, boundless=True, fill_value=0)
    window_data = rasterio.plot.reshape_as_image(window_data)
    # Dimensions should be (row, col, channel).
    height, width, _ = window_data.shape

    # Pad to size requested by the tile, if needed.
    width_pad = tile.width - width
    height_pad = tile.height - height
    if width_pad > 0 or height_pad > 0:
      window_data = np.pad(
          window_data, ((0, height_pad), (0, width_pad), (0, 0)))

    example = _create_tile_example(window_data, tile, self._crs,
                                   self._affine_transform)
    yield example.SerializeToString()
