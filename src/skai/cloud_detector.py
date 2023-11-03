# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for getting cloudinesss score of example images."""

import abc

import cv2
import numpy as np
import tensorflow as tf


class CloudDetectorBase(abc.ABC):
  """Interface for cloud detector models."""

  @abc.abstractmethod
  def detect_single(self, image: np.ndarray) -> float:
    """Detect cloud in a single image."""
    pass


class CloudDetectorTFlite(CloudDetectorBase):
  """Cloud detector loaded from a tflite file."""

  def __init__(self, path: str):
    with tf.io.gfile.GFile(path, 'rb') as model_file:
      model_bytes = model_file.read()
      self.predictor = tf.lite.Interpreter(model_content=model_bytes)

      # For indexing input and output upon inference.
      input_details = self.predictor.get_input_details()
      output_details = self.predictor.get_output_details()
      self._input_index = input_details[0]['index']
      self._output_index = output_details[0]['index']

      # For resizing images upon inference.
      input_shape = input_details[0]['shape']
      self._expected_height = input_shape[1]
      self._expected_width = input_shape[2]

      self.predictor.allocate_tensors()

  def detect_single(self, image: np.ndarray) -> float:
    """Score cloudiness in a single image.

    Args:
      image: A numpy array that represent an image.
        The requirements for the image
              (1) Image values should be in the range from 0.0 to 1.0.
              (2) The shape of image should be (height, width)

    Returns:
      The cloudiness score of the image.
    """

    # Image should be resized to expected_height, expected_width
    height, width = image.shape[0], image.shape[1]

    if height != self._expected_height or width != self._expected_width:
      image = cv2.resize(
          image,
          dsize=(self._expected_height, self._expected_width),
          interpolation=cv2.INTER_LINEAR,
      )

    image = np.expand_dims(image, axis=0)

    image = (image * 255).astype(np.uint8)

    self.predictor.set_tensor(self._input_index, image)
    # Run inference
    self.predictor.invoke()

    # Get output
    output = self.predictor.get_tensor(self._output_index).squeeze(0)
    scores = output / output.sum()
    return scores[1]
