"""Functions for running model inference in beam."""

import time
from typing import Any, Iterator, List, Dict
import apache_beam as beam
import apache_beam.utils.shared as beam_shared
import numpy as np
from skai import utils
from skai.model import data
import tensorflow as tf

def set_gpu_memory_growth() -> None:
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)
  return
set_gpu_memory_growth()
class InferenceModel(object):
  """Abstract base class for an inference model.

  This should be subclassed for each actual model type.
  """

  def prepare_model(self) -> None:
    """Prepares model for inference calls.

    This function will be called in the setup function of the DoFn.
    """
    raise NotImplementedError()

  def predict_scores(self, batch: List[tf.train.Example]) -> np.ndarray:
    """Predicts scores for a batch of input examples."""
    raise NotImplementedError()


def _extract_image_or_blank(
    example: tf.train.Example, feature: str, image_size: int
) -> np.ndarray:
  """Extracts an image from a TF Example.

  If the image feature is missing, returns a blank image instead.

  Args:
    example: Example to extract image from.
    feature: Image feature name.
    image_size: Image size.

  Returns:
    Image as a numpy array, or an array of 0s if image feature is missing.
  """
  if feature in example.features.feature:
    image_bytes = utils.get_bytes_feature(example, feature)[0]
    return data.decode_and_resize_image(image_bytes, image_size).numpy()
  return np.zeros((image_size, image_size, 3), dtype=np.float32)


class TF2InferenceModel(InferenceModel):
  """InferenceModel wrapper for SKAI TF2 models."""

  _model_dir: str
  _image_size: int
  _post_image_only: bool
  _model: Any

  def __init__(
      self, model_dir: str, image_size: int, post_image_only: bool):
    self._model_dir = model_dir
    self._image_size = image_size
    self._post_image_only = post_image_only
    self._shared_handle = beam_shared.Shared()
    self._model = None

  def _make_dummy_input(self):
    num_channels = 3 if self._post_image_only else 6
    image = np.zeros(
        (1, self._image_size, self._image_size, num_channels), dtype=np.float32
    )
    return {'small_image': image, 'large_image': image}

  def _extract_images_or_blanks(
      self,
      example: tf.train.Example,
      pre_image_feature: str,
      post_image_feature: str,
  ) -> np.ndarray:
    """Extracts pre and post disaster images from an example.

    If the image feature is not present, this function will use a blank image
    (all zeros) as a placeholder.

    Args:
      example: Example to extract the image from.
      pre_image_feature: Name of feature storing the pre-disaster image byte.
      post_image_feature: Name of feature storing the post-disaster image byte.

    Returns:
      Numpy array representing the concatenated images.
    """
    post_image = _extract_image_or_blank(
        example, post_image_feature, self._image_size
    )
    if self._post_image_only:
      return post_image
    pre_image = _extract_image_or_blank(
        example, pre_image_feature, self._image_size
    )
    return np.concatenate([pre_image, post_image], axis=2)

  def prepare_model(self) -> None:
    # Use a shared handle so that the model is only loaded once per worker and
    # shared by all processing threads. For more details, see
    #
    # https://medium.com/google-cloud/cache-reuse-across-dofns-in-beam-a34a926db848
    def load():
      model = tf.keras.models.load_model(
          self._model_dir, compile=False
      )
      model.optimizer = None
      # Call predict once to make sure any hidden lazy initialization is
      # triggered. See https://stackoverflow.com/a/43393252
      model.predict(self._make_dummy_input())
      return model

    self._model = self._shared_handle.acquire(load)

  def predict_scores(self, batch: List[tf.train.Example]) -> np.ndarray:
    model_input = self._extract_image_arrays(batch)
    outputs = self._model.predict(model_input)
    return outputs['main'][:, 1]

  def _extract_image_arrays(
      self,
      examples: List[tf.train.Example],
  ) -> Dict[str, np.ndarray]:
    """Reads images from a batch of examples as numpy arrays."""
    small_images = []
    large_images = []
    for example in examples:
      small_images.append(
          self._extract_images_or_blanks(
              example, 'pre_image_png', 'post_image_png'
          )
      )
      large_images.append(
          self._extract_images_or_blanks(
              example, 'pre_image_png_large', 'post_image_png_large'
          )
      )
    return {
        'small_image': np.stack(small_images),
        'large_image': np.stack(large_images),
    }


class ModelInference(beam.DoFn):
  """Model inference DoFn."""

  def __init__(self, score_feature: str, model: InferenceModel):
    self._score_feature = score_feature
    self._model = model
    self._examples_processed = beam.metrics.Metrics.counter(
        'skai', 'examples_processed'
    )
    self._batches_processed = beam.metrics.Metrics.counter(
        'skai', 'batches_processed'
    )
    self._inference_millis = beam.metrics.Metrics.distribution(
        'skai', 'batch_inference_msec'
    )

  def setup(self) -> None:
    self._model.prepare_model()

  def process(
      self, batch: List[tf.train.Example]
  ) -> Iterator[tf.train.Example]:
    start_time = time.process_time()
    scores = self._model.predict_scores(batch)
    elapsed_millis = (time.process_time() - start_time) * 1000
    self._inference_millis.update(elapsed_millis)
    for example, score in zip(batch, scores):
      output_example = tf.train.Example()
      output_example.CopyFrom(example)
      utils.add_float_feature(self._score_feature, score, output_example)
      yield output_example

    self._examples_processed.inc(len(batch))
    self._batches_processed.inc(1)


def run_inference(
    examples: beam.PCollection,
    score_feature: str,
    batch_size: int,
    model: InferenceModel,
) -> beam.PCollection:
  """Runs inference and augments input examples with inference scores.

  Args:
    examples: PCollection of Tensorflow Examples.
    score_feature: Feature name to use for inference scores.
    batch_size: Batch size.
    model: Inference model to use.

  Returns:
    PCollection of Tensorflow Examples augmented with inference scores.
  """
  return (
      examples
      | 'batch'
      >> beam.transforms.util.BatchElements(
          min_batch_size=batch_size, max_batch_size=batch_size
      )
      | 'inference' >> beam.ParDo(ModelInference(score_feature, model))
  )


def _format_example_to_csv_row(example: tf.train.Example) -> str:
  example_id = utils.get_bytes_feature(example, 'example_id')[0].decode()
  longitude, latitude = utils.get_float_feature(example, 'coordinates')
  score = utils.get_float_feature(example, 'score')[0]
  return f'{example_id},{longitude},{latitude},{score}'


def examples_to_csv(examples: beam.PCollection, output_prefix: str) -> None:
  """Converts TF Examples to CSV lines and writes out to file.

  Args:
    examples: PCollection of Tensorflow Examples.
    output_prefix: Prefix of output path.
  """
  _ = (
      examples
      | 'reshuffle_for_output' >> beam.Reshuffle()
      | 'examples_to_csv_lines' >> beam.Map(_format_example_to_csv_row)
      | 'write_csv'
      >> beam.io.textio.WriteToText(output_prefix, file_name_suffix='.csv')
  )


def run_tf2_inference_with_csv_output(
    examples_pattern: str,
    model_dir: str,
    output_prefix: str,
    image_size: int,
    post_image_only: bool,
    batch_size: int,
    pipeline_options):
  """Runs example generation pipeline using TF2 model and outputs to CSV.

  Args:
    examples_pattern: Pattern for input TFRecords.
    model_dir: Model directory.
    output_prefix: Prefix of CSV output path.
    image_size: Image width and height.
    post_image_only: Model expects only post-disaster images.
    batch_size: Batch size.
    pipeline_options: Dataflow pipeline options.
  """
  
  with beam.Pipeline(options=pipeline_options) as pipeline:
    examples = (
        pipeline
        | 'read_tfrecords'
        >> beam.io.tfrecordio.ReadFromTFRecord(
            examples_pattern, coder=beam.coders.ProtoCoder(tf.train.Example))
        | 'reshuffle_input' >> beam.Reshuffle()
    )
    model = TF2InferenceModel(model_dir, image_size, post_image_only)
    scored_examples = run_inference(examples, 'score', batch_size, model)
    examples_to_csv(scored_examples, output_prefix)
