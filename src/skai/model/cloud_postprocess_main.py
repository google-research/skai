r"""Postprocess inference result by idntifying clouds.

Example command:
  python cloud_postprocess_main.py \
    --input_file=/path/to/input/file \
    --output_file=/path/to/output/file \
    --distance_threshold=500
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import pandas as pd
from skai.model import cloud_postprocess_lib
import tensorflow as tf


_INPUT_FILE = flags.DEFINE_string(
    "input_file", None, "Path to csv file to be postprocessed ", required=True
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "Path to output csv file, where the postprocessed result will be written",
    required=True,
)

_DISTANCE_THRESHOLD = flags.DEFINE_integer(
    "distance_threshold",
    500,
    "Distance threshold in meters within which the cloud is identified"
)


def main(argv: Sequence[str]) -> None:
  del argv

  # Read input csv file.
  with tf.io.gfile.GFile(_INPUT_FILE.value, "r") as f:
    df = pd.read_csv(f)

  # Postprocess inference result.
  result_df = cloud_postprocess_lib.identify_clouds(
      df, _DISTANCE_THRESHOLD.value
  )

  with tf.io.gfile.GFile(_OUTPUT_FILE.value, "w") as f:
    result_df.to_csv(f, index=False)


if __name__ == "__main__":
  app.run(main)
