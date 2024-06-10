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

FLAGS = flags.FLAGS
_IMAGE_PATH = flags.DEFINE_string('image_path', '', 'Path to image.')


def validate_image(image_path: str) -> bool:
  """Determines whether an image can be used for SKAI.

  Args:
    image_path: Path to image.

  Returns:
    True if image is valid.
  """
  r = rasterio.open(image_path)
  if len(r.block_shapes) != 3:
    print(f'Image should have 3 bands, but has {len(r.block_shapes)}.')
    return False
  print('Image has 3 bands')
  if not r.is_tiled:
    print('Image is not tiled.')
    return False
  print('Image is tiled')
  for i, s in enumerate(r.block_shapes):
    if s[0] > 512 or s[1] > 512:
      print(f'Band {i} has shape {s}. Tiles should be at most 512x512')
      return False
  print(f'Image tile sizes are {r.block_shapes}')
  if r.colorinterp[0] != rasterio.enums.ColorInterp.red:
    print(
        f'Band 1 interpretation should be red, but is {r.colorinterp[0].name}'
    )
    return False
  if r.colorinterp[1] != rasterio.enums.ColorInterp.green:
    print(
        f'Band 1 interpretation should be green, but is {r.colorinterp[1].name}'
    )
    return False
  if r.colorinterp[2] != rasterio.enums.ColorInterp.blue:
    print(
        f'Band 1 interpretation should be blue, but is {r.colorinterp[2].name}'
    )
    return False
  print('Image bands are in RGB order')
  for i in range(3):
    if r.dtypes[i] != 'uint8':
      print(f'Band {i + 1} should have data type "uint8", but is {r.dtypes[i]}')
      return False
  print('Pixel values are uint8')
  return True


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if validate_image(_IMAGE_PATH.value):
    print('Image is valid')
  else:
    print('Image is not valid')
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
