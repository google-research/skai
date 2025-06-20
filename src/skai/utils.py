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

"""Utility functions for skai package."""

import base64
import collections
import glob
import io
import math
import os
import struct
import tempfile
from typing import Iterable, Sequence
import zipfile

from absl import flags
import geopandas as gpd
import PIL.Image
import tensorflow as tf


Example = tf.train.Example
Image = PIL.Image.Image


def serialize_image(image: Image, image_format: str) -> bytes:
  """Serialize image using the specified format.

  Args:
    image: Input image.
    image_format: Image format to use, e.g. "jpeg"

  Returns:
    Serialized bytes.
  """
  buffer = io.BytesIO()
  image.save(buffer, format=image_format)
  return buffer.getvalue()


def deserialize_image(serialized_bytes: bytes, image_format: str) -> Image:
  return PIL.Image.open(io.BytesIO(serialized_bytes), formats=[image_format])


def add_int64_feature(feature_name: str, value: int, example: Example) -> None:
  """Add int64 feature to tensorflow Example."""
  example.features.feature[feature_name].int64_list.value.append(value)


def add_int64_list_feature(
    feature_name: str, value: Iterable[int], example: Example
) -> None:
  """Add int64 list feature to tensorflow Example."""
  example.features.feature[feature_name].int64_list.value.extend(value)


def add_float_feature(
    feature_name: str, value: float, example: Example
) -> None:
  """Add float feature to tensorflow Example."""
  example.features.feature[feature_name].float_list.value.append(value)


def add_float_list_feature(
    feature_name: str, value: Iterable[float], example: Example
) -> None:
  """Add float list feature to tensorflow Example."""
  example.features.feature[feature_name].float_list.value.extend(value)


def add_bytes_list_feature(
    feature_name: str, value: Iterable[bytes], example: Example
) -> None:
  """Add bytes list feature to tensorflow Example."""
  example.features.feature[feature_name].bytes_list.value.extend(value)


def add_bytes_feature(
    feature_name: str, value: bytes, example: Example
) -> None:
  """Add bytes feature to tensorflow Example."""
  example.features.feature[feature_name].bytes_list.value.append(value)


def get_int64_feature(example: Example, feature_name: str) -> Sequence[int]:
  return list(example.features.feature[feature_name].int64_list.value)


def get_float_feature(example: Example, feature_name: str) -> Sequence[float]:
  return list(example.features.feature[feature_name].float_list.value)


def get_bytes_feature(example: Example, feature_name: str) -> Sequence[bytes]:
  return list(example.features.feature[feature_name].bytes_list.value)


def reformat_flags(flags_list: list[flags.Flag]) -> list[str]:
  """Converts Flag objects to strings formatted as command line arguments.

  Args:
    flags_list: List of Flag objects.
  Returns:
    List of strings, each representing a command line argument.
  """
  formatted_flags = []
  for flag in flags_list:
    if flag.value is not None:
      formatted_flag = f'--{flag.name}='
      if isinstance(flag.value, list):
        formatted_flag += ','.join(flag.value)
      else:
        formatted_flag += f'{flag.value}'
      formatted_flags.append(formatted_flag)
  return formatted_flags


def encode_coordinates(longitude: float, latitude: float) -> str:
  packed = struct.pack('<ff', longitude, latitude)
  return base64.b16encode(packed).decode('ascii')


def decode_coordinates(encoded_coords: str) -> tuple[float, float]:
  buffer = base64.b16decode(encoded_coords.encode('ascii'))
  return struct.unpack('<ff', buffer)


def get_utm_crs(lon: float, lat: float):
  """Based on lat and lng, return best utm epsg-code."""
  utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
  if len(utm_band) == 1:
    utm_band = '0' + utm_band
  if lat >= 0:
    epsg_code = '326' + utm_band
  else:
    epsg_code = '327' + utm_band
  return f'EPSG:{epsg_code}'


def convert_to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
  """Converts a GeoDataFrame to UTM coordinates."""
  sample_gdf = gdf.sample(min(len(gdf), 50))
  centroid = sample_gdf.unary_union.centroid
  return gdf.to_crs(get_utm_crs(centroid.x, centroid.y))


def expand_file_patterns(patterns: Iterable[str]) -> list[str]:
  """Returns the list of paths matched by a list of URI patterns.

  Args:
    patterns: List of file patterns.

  Returns:
    List of matched paths.

  Raises:
    ValueError if any patterns do not match any files, or if files are
      duplicated.
  """
  if not patterns:
    raise ValueError('No patterns to expand')

  paths = []
  for pattern in patterns:
    if (pattern.startswith('/') or
        pattern.startswith('file://') or
        pattern.startswith('gs://') or
        pattern.startswith('s3://')):
      matched = tf.io.gfile.glob(pattern)
      if not matched:
        raise ValueError(
            f'The file pattern "{pattern}" does not match any files'
        )
      paths.extend(matched)
    else:
      paths.append(pattern)

  duplicates = [
      (p, c) for p, c in collections.Counter(paths).items() if c > 1
  ]
  if duplicates:
    raise ValueError(
        'The following input files matched more than one pattern: '
        + ', '.join(f'{p}: {c} times' for p, c in duplicates)
    )
  return paths


def create_zipped_shapefile(
    gdf: gpd.GeoDataFrame, shapefile_name: str, output_path: str
):
  """Writes GeoDataFrame to zipped shapefile.

  Args:
    gdf: The GeoDataFrame.
    shapefile_name: Name of the shapefile, without the .shp suffix.
    output_path: Path to write the zipped shapefile to.
  """
  with tempfile.TemporaryDirectory() as temp_dir:
    shapefile_dir = os.path.join(temp_dir, 'shapefile')
    os.mkdir(shapefile_dir)
    gdf.to_file(os.path.join(shapefile_dir, f'{shapefile_name}.shp'))
    zip_path = os.path.join(temp_dir, '{shapefile_name}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as f:
      for file in glob.glob(os.path.join(shapefile_dir, '*')):
        f.write(file, os.path.basename(file))
    tf.io.gfile.copy(zip_path, output_path, overwrite=True)
