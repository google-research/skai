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
import io
import math
import struct
from typing import Iterable, List, Sequence, Tuple

from absl import flags
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


def reformat_flags(flags_list: List[flags.Flag]) -> List[str]:
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


def decode_coordinates(encoded_coords: str) -> Tuple[float, float]:
  buffer = base64.b16decode(encoded_coords.encode('ascii'))
  return struct.unpack('<ff', buffer)


def convert_wgs_to_utm(lon: float, lat: float):
  """Based on lat and lng, return best utm epsg-code."""
  utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
  if len(utm_band) == 1:
    utm_band = '0' + utm_band
  if lat >= 0:
    epsg_code = '326' + utm_band
  else:
    epsg_code = '327' + utm_band
  return f'EPSG:{epsg_code}'
