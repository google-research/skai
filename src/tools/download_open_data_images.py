r"""Downloads images from Vantor Open Data Program.

This script will download the individual tiles for each image in a collection,
combine those tiles into a single COG-compatible GeoTIFF for each image, and
upload all the GeoTIFFs to a GCS bucket.

This script uses the leafmap library to get information about Open Data images.
Install it with pip like this:

$ pip install leafmap

It also requires you have the standard GDAL tools (gdal_translate, etc.)
installed on your workstation. On ubuntu/debian linux, install it like this:

$ sudo apt install gdal-bin

TODO(jzxu): Currently gdalbuildvrt will fail when tiles for a single image have
different projections. This can be fixed by reprojecting the problem tiles into
the common projection using gdalwarp.

Example usage:

$ python download_open_data_images.py \
    --collection_id=Hurricane-Melissa-Oct-2025 \
    --output_dir=/tmp/open_data/hurricane_melissa \
    --cloud_dir=gs://your-bucket/hurricane_melissa_images
"""

from collections.abc import Sequence
import concurrent.futures
import os
import subprocess as sub
import time

from absl import app
from absl import flags
import geopandas as gpd
import leafmap
import pandas as pd
import requests
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('collection_id', None, 'Collection ID')
flags.DEFINE_string(
    'output_dir', None, 'Output directory on local workstation.'
)
flags.DEFINE_string('cloud_dir', None, 'Output directory in GCS.')


def _download_tile(url: str, output_path: str, max_retries: int = 5) -> bool:
  """Downloads a single tile.

  Args:
    url: Tile URL.
    output_path: Output path.
    max_retries: Number of times to retry.

  Returns:
    True if and only if the download completed successfully.
  """
  for _ in range(max_retries):
    try:
      with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
      print(e)
      time.sleep(1)
  return False


def download_tiles_parallel(
    urls: list[str], output_paths: str, max_workers: int = 10
) -> None:
  """Manages the parallel download of all tile URLs."""
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    for result in tqdm.tqdm(
        executor.map(_download_tile, urls, output_paths), total=len(urls)
    ):
      if not result:
        raise ValueError('Failed to download a tile.')


def gdalbuildvrt(tile_paths: list[str], vrt_path: str) -> bool:
  """Calls gdalbuildvrt to combine tiles into a VRT."""
  try:
    sub.run(['gdalbuildvrt', '-strict', vrt_path] + tile_paths, check=True)
  except sub.CalledProcessError:
    return False
  return True


def gdal_translate(input_path: str, output_path: str):
  """Calls gdal_translate to convert image formats."""
  gdal_translate_options = [
      'COMPRESS=JPEG',
      'PHOTOMETRIC=YCBCR',
      'JPEG_QUALITY=90',
      'TILED=YES',
      'BIGTIFF=YES',
      'NUM_THREADS=ALL_CPUS',
  ]
  command = ['gdal_translate', input_path, output_path]
  for option in gdal_translate_options:
    command.extend(['-co', option])
  sub.run(command, check=True)


def gdaladdo(path: str):
  sub.run(
      ['gdaladdo', path, '--config', 'GDAL_NUM_THREADS', 'ALL_CPUS'], check=True
  )


def gsutil_exists(path: str):
  try:
    sub.run(
        ['gsutil', 'ls', path],
        check=True,
        stdout=sub.DEVNULL,
        stderr=sub.DEVNULL,
    )
  except sub.CalledProcessError:
    return False
  return True


def get_child_info(collection_id: str, child_id: str) -> gpd.GeoDataFrame:
  """Returns a GeoDataFrame with information about a child collection."""
  gdf = leafmap.maxar_items(
      collection_id=collection_id,
      child_id=child_id,
      return_gdf=True,
      assets=['visual'],
  )
  first = gdf.iloc[0]
  properties = {
      c: [first[c]] for c in ('datetime', 'platform', 'catalog_id', 'utm_zone')
  }
  for c in ('view:off_nadir', 'view:azimuth', 'view:incidence_angle',
            'view:sun_azimuth', 'view:sun_elevation', 'tile:clouds_area',
            'tile:clouds_percent'):
    properties[c.split(':')[1]] = [gdf[c].mean()]
  return gpd.GeoDataFrame(properties, geometry=[gdf.unary_union])


def make_geotiff(
    collection_id: str,
    child_id: str,
    output_dir: str,
    output_path: str,
):
  """Makes a GeoTIFF for a child collection."""
  print('Fetching data')
  gdf = leafmap.maxar_items(
      collection_id=collection_id,
      child_id=child_id,
      return_gdf=True,
      assets=['visual'],
  )
  if len(gdf['platform'].unique()) > 1:
    raise ValueError(
        f'Child collection {child_id} has multiple platforms: '
        f'{gdf["platform"].unique()}'
    )
  platform = gdf['platform'].iloc[0]
  if platform not in ['GE01', 'WV02', 'WV03']:
    print(f'{platform} not supported yet.')
    return False

  print(f'Child collection {child_id} is {platform} and has {len(gdf)} tiles.')

  os.makedirs(output_dir, exist_ok=True)

  csv_path = os.path.join(output_dir, f'{child_id}.csv')
  gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)

  tiles_dir = os.path.join(output_dir, 'tiles')
  tile_paths = [os.path.join(tiles_dir, f'{q}.tif') for q in gdf['quadkey']]
  os.makedirs(tiles_dir, exist_ok=True)
  print(f'Downloading {len(tile_paths)} tiles')
  download_tiles_parallel(gdf['visual'].tolist(), tile_paths)

  vrt_path = os.path.join(output_dir, f'{child_id}.vrt')
  print('Building VRT')
  if not gdalbuildvrt(tile_paths, vrt_path):
    print(f'Failed to build VRT for {child_id}, skipping.')
    return False

  print('Creating TIF')
  gdal_translate(vrt_path, output_path)

  print('Adding Overviews')
  gdaladdo(output_path)

  print('Done')
  return True


def download_collection(
    collection_id: str,
    output_dir: str,
    cloud_dir: str,
):
  """Downloads all images in a collection."""
  children = leafmap.maxar_child_collections(collection_id)
  child_infos = []
  for child_id in tqdm.tqdm(children, desc='Downloading collection'):
    child_output_dir = os.path.join(output_dir, child_id)
    geotiff_path = os.path.join(child_output_dir, f'{child_id}.tif')
    cloud_path = os.path.join(cloud_dir, f'{child_id}.tif')

    if gsutil_exists(cloud_path):
      print(f'{cloud_path} already exists, skipping.')
      child_infos.append(get_child_info(collection_id, child_id))
      continue

    if os.path.exists(geotiff_path):
      print(f'Using existing {geotiff_path}.')
    else:
      if not make_geotiff(
          collection_id, child_id, child_output_dir, geotiff_path
      ):
        print(f'{geotiff_path} not created, skipping.')
        continue

    print(f'Copying to {cloud_path}')
    sub.run(['gsutil', 'cp', geotiff_path, cloud_path], check=True)
    child_infos.append(get_child_info(collection_id, child_id))

  info_path = os.path.join(output_dir, f'{collection_id}.shp.zip')
  cloud_info_path = os.path.join(cloud_dir, f'{collection_id}.shp.zip')
  gdf = gpd.GeoDataFrame(pd.concat(child_infos, ignore_index=True))
  gdf.to_file(info_path)
  sub.run(['gsutil', 'cp', info_path, cloud_info_path], check=True)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.collection_id:
    print('Please specify a collection_id.')
    print('Available collection ids:')
    print('\n'.join(leafmap.maxar_collections()))
    return

  if not FLAGS.output_dir:
    print('Please specify an output directory.')
    return

  if not FLAGS.cloud_dir:
    print('Please specify a cloud directory.')
    return

  download_collection(
      FLAGS.collection_id, FLAGS.output_dir, FLAGS.cloud_dir
  )

if __name__ == '__main__':
  app.run(main)
