"""Library for using Vision-Language for zero-shot evaluation of damage."""

import abc
import string
from typing import Iterator

import jax
import numpy as np


def _batch_array(
    array: np.ndarray, batch_size: int = 32
) -> Iterator[np.ndarray]:
  """Batch a numpy array.

  Args:
    array: The input numpy array to be batched with at least one dimension.
    batch_size: The size of the batch for which the array will be batched.

  Yields:
    A sequence of batched numpy arrays each with batch size of batch_size.
    If number of arrays is not dividable then the last array will have a
    batch size of number_of_arrays % batch_size.
  """
  for i in range(0, array.shape[0], batch_size):
    yield array[i : i + batch_size]


def _expand_labels_with_contexts(labels: list[str], contexts: list[str]):
  """Expand labels with contexts.

  Args:
    labels: List of label descriptions i.e damaged buildings.
    contexts: List of contexts that can be used to format the labels, i.e "This
      a satelite image of {}"

  Returns:
    A list of labels that are formatted with the contexts.
  """

  def _is_formattable(context: str) -> bool:
    if not context:
      return False
    fieldnames = [val[1] for val in string.Formatter().parse(context)]
    if fieldnames[0] is None:
      return False
    return True

  expanded_labels = []
  for context in contexts:
    if not _is_formattable(context):
      expanded_labels.extend(labels)
      continue
    for label in labels:
      expanded_labels.append(context.format(label))

  expanded_labels = list(set(expanded_labels))
  return expanded_labels


class VLM(abc.ABC):
  """Provide a generic interface for Vision Language models."""

  def __init__(self):
    self.label_embeddings = None

  @abc.abstractmethod
  def tokenize(self, texts: list[str]) -> np.ndarray:
    raise NotImplementedError()

  @abc.abstractmethod
  def encode_tokens(self, tokens: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  @abc.abstractmethod
  def preprocess_images(self, images: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  @abc.abstractmethod
  def encode_images(self, images: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  @abc.abstractmethod
  def get_temperature(self) -> float:
    raise NotImplementedError()

  def set_label_embeddings(
      self,
      positive_labels: list[str],
      negative_labels: list[str],
      contexts: list[str],
  ):
    """Set label embeddings.

    Args:
      positive_labels: List of label descriptions of the spositive class i.e
        damaged buildings.
      negative_labels: List of label descriptions of the negative class i.e
        damaged buildings.
      contexts: List of contexts that can be used to format the labels, i.e
        "This a satelite image of {}"
    """

    def _get_embedding(labels, contexts):
      expanded_labels = _expand_labels_with_contexts(labels, contexts)
      embeddings = []
      for labels_batch in _batch_array(np.array(expanded_labels)):
        tokens = self.tokenize(labels_batch.tolist())
        embeddings.append(self.encode_tokens(tokens))
      embeddings = np.concatenate(embeddings, axis=0)
      embedding = np.mean(embeddings, axis=0)
      return embedding

    negative_embedding = _get_embedding(negative_labels, contexts)
    positive_embedding = _get_embedding(positive_labels, contexts)
    self.label_embeddings = np.stack(
        [positive_embedding, negative_embedding], axis=0
    )

  def predict(self, images: np.ndarray) -> np.ndarray:
    """Generate probability scores for a batch of images.

    Args:
      images: A batch of images.

    Returns:
      A 2-D array of shape (batch, 2), where the second dimension contains the
        positive and negative class probabilities respectively.

    Raises:
     ValueError if the connected device is not TPU or labels are not set.
    """
    if self.label_embeddings is None:
      raise ValueError('Label embeddings are not set.')

    if jax.lib.xla_bridge.get_backend().platform != 'tpu':
      raise ValueError('Not connected to TPU.')

    images = self.preprocess_images(images)

    batch_size, image_size, _, _ = images.shape
    num_images_to_augment = 0
    if batch_size % jax.local_device_count() != 0:
      num_images_to_augment = jax.local_device_count() - (
          batch_size % jax.local_device_count()
      )
      images_to_augment = np.zeros((num_images_to_augment,) + images.shape[1:])
      images = np.concatenate([images, images_to_augment], axis=0)

    images = images.reshape(
        jax.local_device_count(),
        (batch_size + num_images_to_augment) // jax.local_device_count(),
        image_size,
        image_size,
        3,
    )
    images_embd = self.encode_images(images)
    images_embd = images_embd.reshape(batch_size + num_images_to_augment, -1)[
        :batch_size, :
    ]
    sims = images_embd @ self.label_embeddings.T * self.get_temperature()
    probability_scores = np.array(jax.nn.softmax(sims, axis=-1))
    return probability_scores
