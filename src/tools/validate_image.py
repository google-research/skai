"""Determines whether an image can be used for SKAI.

Usage examples:

For GeoTIFFs stored on your workstation:
python validate_image.py --image_path=/path/to/image.tif

For GeoTIFFs stored in Google Cloud Storage:
python validate_image.py --image_path=gs://bucket/path/to/image.tif

For Earth Engine image assets:
python validate_image.py --image_path=EEDAI:projects/my-project/assets/image

Note that you may need to set environment variables such as
GOOGLE_APPLICATION_CREDENTIALS in order for rasterio/gdal to access your files
in Google Cloud Store or Earth Engine.
"""

from collections.abc import Sequence
import sys
from absl import app
from absl import flags
import rasterio

from skai import read_raster

FLAGS = flags.FLAGS
_IMAGE_PATH = flags.DEFINE_string('image_path', '', 'Path to image.')


def validate_image(image_path: str) -> bool:
  """Determines whether an image can be used for SKAI.

  Args:
    image_path: Path to image.

  Returns:
    True if image is valid.
  """
  valid = True
  r = rasterio.open(image_path)
  if len(r.block_shapes) != 3:
    print(f'Image should have 3 bands, but has {len(r.block_shapes)}.')
    valid = False
  else:
    print('Image has 3 bands')
  if not r.is_tiled:
    print('Image is not tiled.')
    valid = False
  else:
    print('Image is tiled')
  for i, s in enumerate(r.block_shapes):
    if s[0] > 512 or s[1] > 512:
      print(f'Band {i} has shape {s}. Tiles should be at most 512x512')
      valid = False
    else:
      print(f'Band {i} is tiled and the tile size is {s[0]}x{s[1]}')

  try:
    rgb_bands = read_raster.get_rgb_indices(r)
  except ValueError as e:
    print(e)
    valid = False
    rgb_bands = None

  if rgb_bands:
    for i, band in enumerate('RGB'):
      print(f'{band} channel is band {i}')

  for i in range(3):
    if r.dtypes[i] != 'uint8':
      print(f'Band {i + 1} should have data type "uint8", but is {r.dtypes[i]}')
      valid = False
    else:
      print(f'Datatype for band {i + 1} is uint8')
  print('Image is valid' if valid else 'Image is not valid')
  return valid


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not validate_image(_IMAGE_PATH.value):
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
