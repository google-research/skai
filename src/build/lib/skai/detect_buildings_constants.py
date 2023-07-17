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
"""Constants for detect_buildings.py"""
import tensorflow as tf

# TF Example feature names
TILE_ROW = 'tile_row'
TILE_COL = 'tile_col'
TILE_PIXEL_ROW = 'tile_pixel_row'
TILE_PIXEL_COL = 'tile_pixel_col'
MARGIN_SIZE = 'margin_size'
CONFIDENCE = 'confidence'
MASK = 'mask'
CRS = 'crs'
AFFINE_TRANSFORM = 'affine_transform'
LONGITUDE = 'longitude'
LATITUDE = 'latitude'
AREA = 'area'

# Output TF Example feature specification.
FEATURES = {
    TILE_ROW: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    TILE_COL: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    TILE_PIXEL_ROW: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    TILE_PIXEL_COL: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    CONFIDENCE: tf.io.FixedLenFeature([], tf.float32, default_value=0),
    MASK: tf.io.FixedLenFeature([3], tf.string, default_value=['', '', '']),
    CRS: tf.io.FixedLenFeature([], tf.string, default_value=''),
    AFFINE_TRANSFORM: tf.io.FixedLenFeature(
        [6], tf.float32, default_value=[1., 0., 0., 0., 1., 0.]),
    LONGITUDE: tf.io.FixedLenFeature([], tf.float32, default_value=0),
    LATITUDE: tf.io.FixedLenFeature([], tf.float32, default_value=0),
    AREA: tf.io.FixedLenFeature([], tf.float32, default_value=0),
}
