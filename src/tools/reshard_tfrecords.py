"""Reshard examples into TFRecords in a deterministic manner."""

from typing import Sequence

from absl import app
from absl import flags

from skai import generate_examples

FLAGS = flags.FLAGS
flags.DEFINE_string('examples_pattern', None, 'Examples pattern', required=True)
flags.DEFINE_string('output_dir', None, 'Output directory.', required=True)
flags.DEFINE_string('prefix', None, 'Prefix for output files.', required=True)
flags.DEFINE_integer('num_shards', 1000, 'Number of shards to split into.')
flags.DEFINE_string('cloud_project', None, 'GCP project name.')
flags.DEFINE_string('cloud_region', None, 'GCP region, e.g. us-central1.')
flags.DEFINE_string(
    'worker_service_account', None,
    'Service account that will launch Dataflow workers. If unset, workers will '
    'run with the project\'s default Compute Engine service account.')
flags.DEFINE_string('temp_dir', None, 'Temporary directory for Beam.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  generate_examples.reshard_tfrecords(
      FLAGS.examples_pattern,
      FLAGS.output_dir,
      FLAGS.prefix,
      FLAGS.num_shards,
      FLAGS.cloud_project,
      FLAGS.cloud_region,
      FLAGS.worker_service_account,
      FLAGS.temp_dir,
  )

if __name__ == '__main__':
  app.run(main)
