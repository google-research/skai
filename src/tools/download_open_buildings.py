"""Downloads buliding footprints from Open Buildings."""

from collections.abc import Sequence

from absl import app
from absl import flags
import ee
import geopandas as gpd
from skai import earth_engine


FLAGS = flags.FLAGS
flags.DEFINE_string('aoi_path', None, 'AOI path', required=True)
flags.DEFINE_string('output_path', None, 'Output path.', required=True)
flags.DEFINE_float('confidence', 0.6, 'Confidence threshold.')
flags.DEFINE_string(
    'open_buildings_feature_collection',
    'GOOGLE/Research/open-buildings/v3/polygons',
    'Earth Engine feature collection containing Open Buildings footprints.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  ee.Initialize()
  aoi = gpd.read_file(FLAGS.aoi_path)
  earth_engine.get_open_buildings(
      list(aoi.geometry),
      FLAGS.open_buildings_feature_collection,
      FLAGS.confidence,
      False,
      FLAGS.output_path,
  )


if __name__ == '__main__':
  app.run(main)
