"""Tests for vlm_zero_shot_lib."""

import dataclasses
import sys
from absl.testing import parameterized
import jax
import mock
import numpy as np
from skai.experiment.zero_shot_eval import vlm_zero_shot_lib
from google3.testing.pybase import googletest

DIVISIBLE_ARRAY = np.random.randint(size=(64, 224, 224, 3), low=0, high=255)
INDIVISIBLE_ARRAY = np.random.randint(size=(73, 224, 224, 3), low=0, high=255)
DIVISIBLE_ONE_DIM_ARRAY = np.random.randint(size=(64,), low=0, high=255)
INDIVISIBLE_ONE_DIM_ARRAY = np.random.randint(size=(73,), low=0, high=255)
PACKAGE = "skai.experiment.zero_shot_eval.vlm_zero_shot_lib."


MAX_INT = sys.maxsize


class _FakeVLM(vlm_zero_shot_lib.VLM):

  def __init__(self, embd_size: int):
    super().__init__()
    self.embd_size = embd_size

  def tokenize(self, texts: list[str]) -> np.ndarray:
    return np.random.randn(len(texts), self.embd_size)

  def encode_tokens(self, tokens: np.ndarray) -> np.ndarray:
    return np.random.randn(*tokens.shape[:-1], self.embd_size)

  def preprocess_images(self, images: np.ndarray) -> np.ndarray:
    return np.random.randn(*images.shape)

  def encode_images(self, images: np.ndarray) -> np.ndarray:
    return np.random.randn(*images.shape[:-3], self.embd_size)

  def get_temperature(self) -> float:
    return 10.0


class VlmZeroShotLibTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("divisible arrays", DIVISIBLE_ARRAY, 32),
      ("indivisible arrays", INDIVISIBLE_ARRAY, 32),
      ("divisible array with one dimension", DIVISIBLE_ONE_DIM_ARRAY, 32),
      ("indivisible array with one dimension", INDIVISIBLE_ONE_DIM_ARRAY, 32),
      ("number of arrays less than batch size", DIVISIBLE_ARRAY, MAX_INT),
  )
  def test_equality_batch_array(self, array, batch_size):
    batched_arrays = vlm_zero_shot_lib._batch_array(array, batch_size)
    stacked_arrays = np.concatenate(list(batched_arrays), axis=0)
    np.testing.assert_array_equal(stacked_arrays, array)

  @parameterized.named_parameters(
      (
          "formattable contexts",
          ["damaged building", "damaged roof"],
          ["This is a satellite image of {}", "This is top down view of {}"],
          [
              "This is a satellite image of damaged building",
              "This is top down view of damaged building",
              "This is a satellite image of damaged roof",
              "This is top down view of damaged roof",
          ],
      ),
      (
          "unformattable contexts",
          ["damaged building", "damaged roof"],
          ["This is a satellite image of", "This is top down view of"],
          ["damaged building", "damaged roof"],
      ),
      (
          "Partially formattable contexts",
          ["damaged building", "damaged roof"],
          ["This is a satellite image of {}", "This is top down view of"],
          [
              "This is a satellite image of damaged building",
              "damaged building",
              "This is a satellite image of damaged roof",
              "damaged roof",
          ],
      ),
  )
  def test_expand_labels_with_contexts(
      self, labels, contexts, expected_labels_with_contexts
  ):
    labels_with_contexts = vlm_zero_shot_lib._expand_labels_with_contexts(
        labels, contexts
    )
    self.assertCountEqual(labels_with_contexts, expected_labels_with_contexts)

  @parameterized.named_parameters(
      (
          "formattable contexts",
          ["damaged building", "damaged roof"],
          ["undamaged buidling", "intact houses"],
          ["This is a satellite image of {}", "This is top down view of {}"],
      ),
      (
          "unformattable contexts",
          ["damaged building", "damaged roof"],
          ["undamaged buidling", "intact houses"],
          ["This is a satellite image of", "This is top down view of"],
      ),
      (
          "Partially formattable contexts",
          ["damaged building", "damaged roof"],
          ["undamaged buidling", "intact houses"],
          ["This is a satellite image of {}", "This is top down view of"],
      ),
  )
  def test_vlm_set_label_embeddings(
      self, positive_labels, negative_labels, contexts
  ):
    embd_size = 1024
    vlm = _FakeVLM(embd_size)
    vlm.set_label_embeddings(positive_labels, negative_labels, contexts)
    self.assertEqual(list(vlm.label_embeddings.shape), [2, 1024])

  @parameterized.named_parameters(
      (
          "divisible arrays",
          DIVISIBLE_ARRAY,
          ["damaged building"],
          ["intact houses"],
          ["This is top down view of {}"],
      ),
      (
          "indivisible arrays",
          INDIVISIBLE_ARRAY,
          ["damaged building"],
          ["intact houses"],
          ["This is top down view of {}"],
      ),
      (
          "unformattable contexts",
          DIVISIBLE_ARRAY,
          ["damaged building", "damaged roof"],
          ["undamaged buidling", "intact houses"],
          ["This is a satellite image of", "This is top down view of"],
      )
  )
  @mock.patch(PACKAGE + "jax")
  def test_vlm_predict(
      self, images, positive_labels, negative_labels, contexts, mocked_jax
  ):
    @dataclasses.dataclass(frozen=True)
    class _MockedClient:
      platform: str

    mocked_jax.local_device_count.return_value = 8
    mocked_jax.nn.softmax.return_value = jax.nn.softmax(
        np.random.randn(images.shape[0], 2)
    )
    mocked_jax.lib.xla_bridge.get_backend.return_value = _MockedClient(
        platform="tpu"
    )

    vlm = _FakeVLM(1024)
    vlm.set_label_embeddings(positive_labels, negative_labels, contexts)
    scores = vlm.predict(images)

    self.assertEqual(scores.shape, (images.shape[0], 2))


if __name__ == "__main__":
  googletest.main()
