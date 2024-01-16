"""A script for backfilling TFRecords that don't contain the int64_id feature.
"""

from typing import Iterator

from absl import app
from absl import flags
import apache_beam as beam
from skai import beam_utils
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string('cloud_project', None, 'GCP project name.')
flags.DEFINE_string('cloud_region', None, 'GCP region, e.g. us-central1.')
flags.DEFINE_string(
    'worker_service_account', None,
    'Service account that will launch Dataflow workers. If unset, workers will '
    'run with the project\'s default Compute Engine service account.')
flags.DEFINE_integer(
    'max_dataflow_workers', None, 'Maximum number of dataflow workers'
)
flags.DEFINE_string('examples_pattern', None, 'Input examples pattern.')
flags.DEFINE_string('output_path', None, 'Output directory.')


@beam.typehints.with_input_types(tf.train.Example)
@beam.typehints.with_output_types(tf.train.Example)
class AddInt64Id(beam.DoFn):
  """DoFn for adding int64_id feature to examples."""

  def process(self, example: tf.train.Example) -> Iterator[tf.train.Example]:
    # pylint: disable=g-import-not-at-top
    # These import statements must be here instead of at the top of the file to
    # get picked up by DataFlow workers.
    import binascii
    import copy
    import struct
    # pylint: enable=g-import-not-at-top

    if 'int64_id' in example.features.feature:
      yield example
      return

    example_id = (
        example.features.feature['example_id'].bytes_list.value[0].decode()
    )
    int64_id = struct.unpack('<q', binascii.a2b_hex(example_id[:16]))[0]
    new_example = copy.deepcopy(example)
    new_example.features.feature['int64_id'].int64_list.value.append(int64_id)
    yield new_example


def main(_) -> None:
  num_output_shards = len(tf.io.gfile.glob(FLAGS.examples_pattern))
  if num_output_shards == 1:
    output_shard_template = ''
    output_suffix = ''
    use_dataflow = False
  else:
    output_shard_template = None
    output_suffix = '.tfrecord'
    use_dataflow = True

  temp_dir = f'{FLAGS.output_path}.beam_temp'
  pipeline_options = beam_utils.get_pipeline_options(
      use_dataflow,
      'backfill-int64-id',
      FLAGS.cloud_project,
      FLAGS.cloud_region,
      temp_dir,
      FLAGS.max_dataflow_workers,
      FLAGS.worker_service_account,
      machine_type=None,
      accelerator=None,
      accelerator_count=0,
  )
  pipeline = beam.Pipeline(options=pipeline_options)
  _ = (
      pipeline
      | 'read_examples'
      >> beam.io.tfrecordio.ReadFromTFRecord(
          FLAGS.examples_pattern, coder=beam.coders.ProtoCoder(tf.train.Example)
      )
      | 'add_int64_id' >> beam.ParDo(AddInt64Id())
      | 'write_output'
      >> beam.io.tfrecordio.WriteToTFRecord(
          FLAGS.output_path,
          file_name_suffix=output_suffix,
          num_shards=num_output_shards,
          shard_name_template=output_shard_template,
          coder=beam.coders.ProtoCoder(tf.train.Example),
      )
  )
  pipeline.run()


if __name__ == '__main__':
  app.run(main)
