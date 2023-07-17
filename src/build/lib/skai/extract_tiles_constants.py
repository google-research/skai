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
import tensorflow as tf

# Output TF Example feature names.
IMAGE_WIDTH = 'image/width'
IMAGE_HEIGHT = 'image/height'
X_OFFSET = 'x'
Y_OFFSET = 'y'
IMAGE_FORMAT = 'image/format'
IMAGE_ENCODED = 'image/encoded'
TILE_ROW = 'tile_row'
TILE_COL = 'tile_column'
CRS = 'crs'
AFFINE_TRANSFORM = 'affine_transform'
MARGIN_SIZE = 'margin_size'


# Output TF Example feature specification.
FEATURES = {
    IMAGE_WIDTH: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    IMAGE_HEIGHT: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    X_OFFSET: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    Y_OFFSET: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    TILE_ROW: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    TILE_COL: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    IMAGE_FORMAT: tf.io.FixedLenFeature([], tf.string, default_value='png'),
    IMAGE_ENCODED: tf.io.FixedLenFeature([], tf.string, default_value=''),
    CRS: tf.io.FixedLenFeature([], tf.string, default_value=''),
    AFFINE_TRANSFORM: tf.io.FixedLenFeature(
        [6], tf.float32, default_value=[1., 0., 0., 0., 1., 0.]),
    MARGIN_SIZE: tf.io.FixedLenFeature([], tf.int64, default_value=0),
}
