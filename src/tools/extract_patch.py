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

r"""A tool for extracting patches from images.

This is useful for checking if the example generation pipeline is reading
images correctly.

Example invocation:

$ python tools/extract_patch.py \
  --image_path=gs://bucket/path/to/image.tif \
  --longitude=-89.80628454 \
  --latitude=29.56588473 \
  --output=/tmp/patch.png
"""

from absl import app
from absl import flags
import PIL.Image
import rasterio
from skai import generate_examples

FLAGS = flags.FLAGS

flags.DEFINE_string('image_path', '', 'Path to image.')
flags.DEFINE_float('longitude', 0.0, 'Longitude of patch center.')
flags.DEFINE_float('latitude', 0.0, 'Latitude of patch center.')
flags.DEFINE_integer('size', 256, 'Patch size in pixels.')
flags.DEFINE_float('resolution', 0.5, 'Resolution.')
flags.DEFINE_string('output', '', 'Output path.')


def main(_) -> None:
  raster = rasterio.open(FLAGS.image_path)
  patch = generate_examples.get_patch_at_coordinate(raster, FLAGS.longitude,
                                                    FLAGS.latitude, FLAGS.size,
                                                    FLAGS.resolution, 0)
  image = PIL.Image.fromarray(patch)
  image.save(FLAGS.output)

if __name__ == '__main__':
  app.run(main)
