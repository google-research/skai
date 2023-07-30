# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Control Theory based self-augmentation.

Based on method introduced in ReMixMatch: Semi-Supervised Learning with
Distribution Matching and Augmentation Anchoring.
Link: https://openreview.net/forum?id=HklkeR4KPB.
"""

import collections
import random
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps

OPS = {}
OP = collections.namedtuple('OP', ('f', 'bins'))
Sample = collections.namedtuple('Sample', ('train', 'probe'))

# The range for each parameter is discretized into 17 bins from 0 to 1.
# Other hyperparamters have been tuned and selected in paper.
# (See https://arxiv.org/pdf/1911.09785.pdf Table 5)
NUM_DISCRETIZED_BINS = 17
SCALE_LEVEL = 6
POSTERIZE_LEVEL = 8


def _register(*bins):

  def wrap(f):
    OPS[f.__name__] = OP(f, bins)
    return f

  return wrap


def _has_building_mask_channels(x: np.ndarray) -> bool:
  """Checks to see if the image contains building segmentation channels."""
  return x.shape[2] >= 8


def _image_to_array(image: Image.Image) -> np.ndarray:
  return np.asarray(image).astype('f') / 127.5 - 1


def apply(x: np.ndarray,
          ops: List[Tuple[OP, Tuple[float]]],
          cutout: bool = True) -> np.ndarray:
  """Apply the augmentation operations to the pre- and post-disaster images."""
  if ops is None:
    return x
  # Values are currently -1 to 1 (float), so change to 0 to 255 (byte)
  # First 6 channels will be pre and post imagery. Later channels may be
  # arbitrary features, which should not be augmented. If only post-disaster
  # imagery is being used, there will only be 3 channels.
  use_pre_disaster_image = (x.shape[-1] > 3)
  if use_pre_disaster_image:
    y_pre = Image.fromarray(
        np.round(127.5 * (1 + x[:, :, :3])).clip(0, 255).astype('uint8'))
    y_post = Image.fromarray(
        np.round(127.5 * (1 + x[:, :, 3:6])).clip(0, 255).astype('uint8'))
  else:
    y_post = Image.fromarray(
        np.round(127.5 * (1 + x[:, :, :3])).clip(0, 255).astype('uint8'))
  for op, args in ops:
    if use_pre_disaster_image:
      y_pre = OPS[op].f(y_pre, *args)
    y_post = OPS[op].f(y_post, *args)

  if use_pre_disaster_image:
    y_channels = [_image_to_array(y_pre), _image_to_array(y_post)]
    # TODO(jlee24): Support augmenting mask channels in single image case
    if _has_building_mask_channels(x):
      y_channels.append(x[:, :, 6:8])
    y = np.concatenate(y_channels, axis=-1)
  else:
    y = _image_to_array(y_post)
  return cutout_numpy(y) if cutout else y


class CTAugment:
  """Control Theory Augment (CTAugment), which learns an augmentation policy.

  The CTAugment method uniformly randomly samples transformations to apply
  and dynamically infers magnitudes for each transformation during training.
  For each augmentation parameter, CTAugment learns the likelihood it will
  produce an image which is classified correctly. Using the likelihoods, it then
  only samples augmentations that fall within the network tolerance. Details:
  First, CTAugment divides each parameter for each transformation into bins of
  distortion magnitude. Let `m` be the vector of bin weights for a given
  distortion parameter for a given transformation. At the beginning, all bins
  are initialized with weight 1. At each training step, `depth` transformations
  are sampled uniformly at random. They are then applied to a labeled example
  `x` with label `p` to obtain an augmented version `x'`. Then, the method
  measures the extent to which the model's prediction matches the label.
  The weight for each sampled bin is updated according to that accuracy factor.
  Reference: https://arxiv.org/abs/1911.09785

  Attributes:
    depth: Number of augmentations that are sampled and applied.
    decay: Exponential decay hyperpameter when adjusting bin weights.
    threshold: Minimum bin weight needed to apply transformation.
    rates: Rates of applying each transformation.
  """

  def __init__(self,
               depth: int = 2,
               decay: float = 0.99,
               threshold: float = 0.85):
    """Initializes an instance of the CTAugment class.

    Args:
      depth: Number of augmentations that are sampled and applied.
      decay: Exponential decay hyperpameter when adjusting bin weights.
      threshold: Minimum bin weight needed to apply transformation.
    """
    self.depth = depth
    self.decay = decay
    self.threshold = threshold
    self.rates = {}
    for k, op in OPS.items():
      self.rates[k] = tuple([np.ones(x, 'f') for x in op.bins])

  def _rate_to_p(self, rate: np.ndarray) -> np.ndarray:
    """Modifies the set of bin weights into categorical normalized variables.

    Sets any bin weight below the threshold to 0, which means it won't be
    sampled and thus not applied.

    Args:
      rate: Vector of bin weights.

    Returns:
      p: Modified vector of bin weights.
    """
    p = rate + (1 - self.decay)
    p = p / p.max()
    p[p < self.threshold] = 0
    return p

  def policy(self, probe: bool) -> List[OP]:
    """Creates policy by uniformly sampling from transformations and magnitudes.

    Args:
      probe: True when input is a labeled image, which means the method must
        sample uniformly from the policy to update CTAugment probabilities.
        False when input is an unlabeled image.

    Returns:
      v: A list of transformation functions to apply and its magnitude.
    """
    kl = list(OPS.keys())
    v = []
    if probe:
      for _ in range(self.depth):
        k = random.choice(kl)
        bins = self.rates[k]
        rnd = np.random.uniform(0, 1, len(bins))
        v.append(OP(k, rnd.tolist()))
      return v
    for _ in range(self.depth):
      vt = []
      k = random.choice(kl)
      bins = self.rates[k]
      rnd = np.random.uniform(0, 1, len(bins))
      for r, bin_selected in zip(rnd, bins):
        p = self._rate_to_p(bin_selected)
        segments = p[1:] + p[:-1]
        segment = np.random.choice(
            segments.shape[0], p=segments / segments.sum())
        vt.append((segment + r) / segments.shape[0])
      v.append(OP(k, vt))
    return v

  def update_rates(self, policy: Iterable[np.ndarray], accuracy: float):
    """Updates bin weights based on how well model predicted augmentations.

    Args:
      policy: A list of magnitude bin weight vectors for each transformation.
      accuracy: Measurement of extent to which the model’s prediction matches
        the label.
    """

    for k, bins in policy:
      for p, rate in zip(bins, self.rates[k]):
        p = int(p * len(rate) * 0.999)
        rate[p] = rate[p] * self.decay + accuracy * (1 - self.decay)

  def stats(self) -> str:
    """Print the current rates for each transformation for each magnitude."""
    return '\n'.join('%-16s    %s' %  # pylint: disable=g-complex-comprehension
                     (k, ' / '.join(' '.join('%.2f' % x
                                             for x in self._rate_to_p(rate))
                                    for rate in self.rates[k]))
                     for k in sorted(OPS.keys()))


def _get_enhance_factor(level: float) -> float:
  """Value controlling ImageEnhance.enhance severity.

  Factor of 1. always returns a copy of the original image, while lower
  factors means less and higher factors means more.
  Level passed is between 0 and 1, but we want the final factor to be > 1.

  Args:
    level: Value from augmentation policy between 0 and 1 for severity.

  Returns:
    Factor for ImageEnhance.enhance.
  """
  return 0.1 + 1.9 * level


@_register(NUM_DISCRETIZED_BINS)
def autocontrast(x: Image.Image, level: float) -> Image.Image:
  return Image.blend(x, ImageOps.autocontrast(x), level)


@_register(NUM_DISCRETIZED_BINS)
def blur(x: Image.Image, level: float) -> Image.Image:
  return Image.blend(x, x.filter(ImageFilter.BLUR), level)


@_register(NUM_DISCRETIZED_BINS)
def brightness(x: Image.Image, brightness_factor: float) -> Image.Image:
  return ImageEnhance.Brightness(x).enhance(
      _get_enhance_factor(brightness_factor))


@_register(NUM_DISCRETIZED_BINS)
def color(x: Image.Image, color_factor: float) -> Image.Image:
  return ImageEnhance.Color(x).enhance(_get_enhance_factor(color_factor))


@_register(NUM_DISCRETIZED_BINS)
def contrast(x: Image.Image, contrast_factor: float) -> Image.Image:
  return ImageEnhance.Contrast(x).enhance(_get_enhance_factor(contrast_factor))


def cutout_numpy(img: np.ndarray, level: float = 0.5) -> np.ndarray:
  """Apply cutout with severity provided by level.

  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
  This operation applies a `level * img_height` x `level * img_width` mask of
  zeros to a random location within `img`.

  Args:
    img: Numpy image that cutout will be applied to.
    level: Severity with which to apply cutout. A level of 0.5 means cutout area
      will have half the height and width of the full image.

  Returns:
    A numpy tensor that is the result of applying the cutout mask to `img`.
  """
  img_height, img_width = img.shape[0], img.shape[1]
  cut_height = int(img_height * level)
  cut_width = int(img_height * level)
  up = np.random.randint(img_height - cut_height)
  left = np.random.randint(img_width - cut_width)
  img[up:up + cut_height, left:left + cut_width, ...] = 0.
  return img


@_register(NUM_DISCRETIZED_BINS)
def equalize(x: Image.Image, level: float) -> Image.Image:
  return Image.blend(x, ImageOps.equalize(x), level)


@_register(NUM_DISCRETIZED_BINS)
def invert(x: Image.Image, level: float) -> Image.Image:
  return Image.blend(x, ImageOps.invert(x), level)


@_register()
def identity(x: Image.Image) -> Image.Image:
  return x


@_register(POSTERIZE_LEVEL)
def posterize(x: Image.Image, level: float) -> Image.Image:
  level = 1 + int(level * 7.999)
  return ImageOps.posterize(x, level)


@_register(NUM_DISCRETIZED_BINS, SCALE_LEVEL)
def rescale(x: Image.Image, scale: float, method: float) -> Image.Image:
  s = x.size
  scale *= 0.25
  crop = (scale, scale, s[0] - scale * s[0], s[1] - scale * s[1])
  methods = (
      Image.Resampling.LANCZOS,
      Image.Resampling.BICUBIC,
      Image.Resampling.BILINEAR,
      Image.Resampling.BOX,
      Image.Resampling.HAMMING,
      Image.Resampling.NEAREST,
  )
  method = methods[int(method * 5.99)]
  return x.crop(crop).resize(x.size, method)


@_register(NUM_DISCRETIZED_BINS)
def rotate(x: Image.Image, angle: float) -> Image.Image:
  angle = int(np.round((2 * angle - 1) * 45))
  return x.rotate(angle)


@_register(NUM_DISCRETIZED_BINS)
def sharpness(x: Image.Image, sharpness_factor: float) -> Image.Image:
  return ImageEnhance.Sharpness(x).enhance(
      _get_enhance_factor(sharpness_factor))


@_register(NUM_DISCRETIZED_BINS)
def shear_x(x: Image.Image, shear: float) -> Image.Image:
  shear = (2 * shear - 1) * 0.3
  return x.transform(x.size, Image.Transform.AFFINE, (1, shear, 0, 0, 1, 0))


@_register(NUM_DISCRETIZED_BINS)
def shear_y(x: Image.Image, shear: float) -> Image.Image:
  shear = (2 * shear - 1) * 0.3
  return x.transform(x.size, Image.Transform.AFFINE, (1, 0, 0, shear, 1, 0))


@_register(NUM_DISCRETIZED_BINS)
def smooth(x: Image.Image, level: float) -> Image.Image:
  return Image.blend(x, x.filter(ImageFilter.SMOOTH), level)


@_register(NUM_DISCRETIZED_BINS)
def solarize(x: Image.Image, th: float) -> Image.Image:
  th = int(th * 255.999)
  return ImageOps.solarize(x, th)


@_register(NUM_DISCRETIZED_BINS)
def translate_x(x: Image.Image, delta: float) -> Image.Image:
  return x.transform(x.size, Image.Transform.AFFINE, (1, 0, delta, 0, 1, 0))


@_register(NUM_DISCRETIZED_BINS)
def translate_y(x: Image.Image, delta: float) -> Image.Image:
  return x.transform(x.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, delta))
