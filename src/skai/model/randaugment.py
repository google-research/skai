"""Adapted implementation of RandAugment from https://arxiv.org/abs/1909.13719 to support 6-channels images.

The code is adapted from https://github.com/wangermeng2021/FastClassification.
"""

import math
from typing import Any, List, Text, Tuple

import tensorflow as tf
import tensorflow_addons.image as image_ops

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0


def extend_channels(func):
  """Extends the image augmentation to handle 6-channels images."""
  def extended_channels_func(*args, **kwargs):
    image = args[0]
    if image.shape[-1] == 6:
      aug_before_image = func(image[..., :3], *args[1:], **kwargs)
      aug_after_image = func(image[..., 3:], *args[1:], **kwargs)

      aug_image = tf.concat([aug_before_image, aug_after_image], axis=-1)
      return aug_image
    elif image.shape[-1] == 3:
      return func(*args, **kwargs)
    else:
      raise ValueError('Incorrect number of channels')
    # Call the original function with the modified argument
  return extended_channels_func


def to_4d(image: tf.Tensor) -> tf.Tensor:
  """Converts an input Tensor to 4 dimensions.

  4D image => [N, H, W, C] or [N, C, H, W] 3D image => [1, H, W, C] or [1, C, H,
  W] 2D image => [1, H, W, 1]

  Args:
    image: The 2/3/4D input tensor.

  Returns:
    A 4D image tensor.
  Raises:
    `TypeError` if `image` is not a 2/3/4D tensor.
  """
  shape = tf.shape(image)
  original_rank = tf.rank(image)
  left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
  right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
  new_shape = tf.concat(
      [
          tf.ones(shape=left_pad, dtype=tf.int32),
          shape,
          tf.ones(shape=right_pad, dtype=tf.int32),
      ],
      axis=0,
  )
  return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
  """Converts a 4D image back to `ndims` rank."""
  shape = tf.shape(image)
  begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
  end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
  new_shape = shape[begin:end]
  return tf.reshape(image, new_shape)


def _convert_translation_to_transform(translations: List[int]) -> tf.Tensor:
  """Converts translations to a projective transform.

  The translation matrix looks like this:

    [[1 0 -dx]
     [0 1 -dy]
     [0 0 1]]
  Args:
    translations: The 2-element list representing [dx, dy], or a matrix of
      2-element lists representing [dx dy] to translate for each image. The
      shape must be static.

  Returns:
    The transformation matrix of shape (num_images, 8).
  Raises:
    `TypeError` if
      - the shape of `translations` is not known or
      - the shape of `translations` is not rank 1 or 2.
  """
  translations = tf.convert_to_tensor(translations, dtype=tf.float32)
  if translations.get_shape().ndims is None:
    raise TypeError('translations rank must be statically known')
  elif len(translations.get_shape()) == 1:
    translations = translations[None]
  elif len(translations.get_shape()) != 2:
    raise TypeError('translations should have rank 1 or 2.')
  num_translations = tf.shape(translations)[0]

  return tf.concat(
      values=[
          tf.ones((num_translations, 1), tf.dtypes.float32),
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          -translations[:, 0, None],
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          tf.ones((num_translations, 1), tf.dtypes.float32),
          -translations[:, 1, None],
          tf.zeros((num_translations, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def _convert_angles_to_transform(
    angles: tf.Tensor, image_width: tf.Tensor, image_height: tf.Tensor
) -> tf.Tensor:
  """Converts an angle or angles to a projective transform.

  Args:
    angles: A scalar to rotate all images, or a vector to rotate a batch of
      images. This must be a scalar.
    image_width: The width of the image(s) to be transformed.
    image_height: The height of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8).
  Raises:
    `TypeError` if `angles` is not rank 0 or 1.
  """
  angles = tf.convert_to_tensor(angles, dtype=tf.float32)
  if len(angles.get_shape()) == 0:  # pylint: disable=g-explicit-length-test
    angles = angles[None]
  elif len(angles.get_shape()) != 1:
    raise TypeError('Angles should have a rank 0 or 1.')
  x_offset = (
      (image_width - 1)
      - (
          tf.math.cos(angles) * (image_width - 1)
          - tf.math.sin(angles) * (image_height - 1)
      )
  ) / 2.0
  y_offset = (
      (image_height - 1)
      - (
          tf.math.sin(angles) * (image_width - 1)
          + tf.math.cos(angles) * (image_height - 1)
      )
  ) / 2.0
  num_angles = tf.shape(angles)[0]
  return tf.concat(
      values=[
          tf.math.cos(angles)[:, None],
          -tf.math.sin(angles)[:, None],
          x_offset[:, None],
          tf.math.sin(angles)[:, None],
          tf.math.cos(angles)[:, None],
          y_offset[:, None],
          tf.zeros((num_angles, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def transform(image: tf.Tensor, transforms) -> tf.Tensor:
  """Prepares input data for `image_ops.transform`."""
  original_ndims = tf.rank(image)
  transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
  if transforms.shape.rank == 1:
    transforms = transforms[None]
  image = to_4d(image)
  image = image_ops.transform(
      images=image, transforms=transforms, interpolation='nearest'
  )
  return from_4d(image, original_ndims)


def translate(image: tf.Tensor, translations) -> tf.Tensor:
  """Translates image(s) by provided vectors.

  Args:
    image: An image Tensor of type uint8.
    translations: A vector or matrix representing [dx dy].

  Returns:
    The translated version of the image.
  """
  transforms = _convert_translation_to_transform(translations)
  return transform(image, transforms=transforms)


def rotate(image: tf.Tensor, degrees: float) -> tf.Tensor:
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    The rotated version of image.
  """

  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = tf.cast(degrees * degrees_to_radians, tf.float32)

  original_ndims = tf.rank(image)
  image = to_4d(image)

  image_height = tf.cast(tf.shape(image)[1], tf.float32)
  image_width = tf.cast(tf.shape(image)[2], tf.float32)
  transforms = _convert_angles_to_transform(
      angles=radians, image_width=image_width, image_height=image_height
  )
  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = transform(image, transforms=transforms)
  return from_4d(image, original_ndims)


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
  """Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used. A value of
  1.0 means only image2 is used.  A value between 0.0 and 1.0 means we linearly
  interpolate the pixel values between the two images.  A value greater than 1.0
  "extrapolates" the difference between the two pixel values, and we clip the
  results to values between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  """
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1, tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


@extend_channels
def cutout(image: tf.Tensor, pad_size: int, replace: int = 0) -> tf.Tensor:
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to a random
  location within `img`. The pixel values filled in will be of the value
  `replace`. The located where the mask will be applied is randomly chosen
  uniformly over the whole image.

  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that is
      applied to the image. The mask will be of size (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has the
      cutout mask applied to it.

  Returns:
    An image Tensor that is of type uint8.
  """
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32
  )

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32
  )

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad),
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
  )
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image
  )
  return image


@extend_channels
def solarize(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)


@extend_channels
def solarize_add(
    image: tf.Tensor, addition: int = 0, threshold: int = 128
) -> tf.Tensor:
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = tf.cast(image, tf.int64) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.where(image < threshold, added_image, image)


@extend_channels
def color(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


@extend_channels
def contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


@extend_channels
def brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


@extend_channels
def posterize(image: tf.Tensor, bits: int) -> tf.Tensor:
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


@extend_channels
def wrapped_rotate(image: tf.Tensor, degrees: float, replace: int) -> tf.Tensor:
  """Applies rotation with wrap/unwrap."""
  image = rotate(wrap(image), degrees=degrees)
  return unwrap(image, replace)


@extend_channels
def translate_x(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
  """Equivalent of PIL Translate in X dimension."""
  image = translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


@extend_channels
def translate_y(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
  """Equivalent of PIL Translate in Y dimension."""
  image = translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


# TODO(mohammedelfatihsalah): replace should have a type of list with exactly 3 values.
@extend_channels
def shear_x(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = transform(
      image=wrap(image), transforms=[1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  )
  return unwrap(image, replace)


@extend_channels
def shear_y(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = transform(
      image=wrap(image), transforms=[1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0]
  )
  return unwrap(image, replace)


@extend_channels
def autocontrast(image: tf.Tensor) -> tf.Tensor:
  """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image: tf.Tensor) -> tf.Tensor:
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


@extend_channels
def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = (
      tf.constant(
          [[1, 1, 1], [1, 5, 1], [1, 1, 1]],
          dtype=tf.float32,
          shape=[3, 3, 1, 1],
      )
      / 13.0
  )
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1]
  )
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


@extend_channels
def equalize(image: tf.Tensor) -> tf.Tensor:
  """Implements Equalize function from PIL using TF ops."""

  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(
        tf.equal(step, 0),
        lambda: im,
        lambda: tf.gather(build_lut(histo, step), im),
    )

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


@extend_channels
def invert(image: tf.Tensor) -> tf.Tensor:
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


def wrap(image: tf.Tensor) -> tf.Tensor:
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], axis=2)
  return extended


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position, the rest of
  the three channels in that spatial dimension are grayed (set to 128).
  Operations like translate and shear on a wrapped Tensor will leave 0s in empty
  locations.  Some transformations look at the intensity of values to do
  preprocessing, and we want these empty pixels to assume the 'average' value,
  rather than pure black.

  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[:, 3], axis=-1)

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image,
  )

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
  return image


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 30.0
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level: float):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2.0 / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level: float):
  return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level: float, translate_const: float):
  level = (level / _MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _mult_to_arg(level: float, multiplier: float = 1.0):
  return (int((level / _MAX_LEVEL) * multiplier),)


def _apply_func_with_prob(func: Any, image: tf.Tensor, args: Any, prob: float):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool
  )
  augmented_image = tf.cond(
      should_apply_op, lambda: func(image, *args), lambda: image
  )
  return augmented_image


def select_and_apply_random_policy(policies: Any, image: tf.Tensor):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for i, policy in enumerate(policies):
    image = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image),
        lambda: image,
    )
  return image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
})


def level_to_arg(cutout_const: float, translate_const: float):
  """Creates a dict mapping image operation names to their arguments."""

  no_arg = lambda level: ()
  posterize_arg = lambda level: _mult_to_arg(level, 4)
  solarize_arg = lambda level: _mult_to_arg(level, 256)
  solarize_add_arg = lambda level: _mult_to_arg(level, 110)
  cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
  translate_arg = lambda level: _translate_level_to_arg(level, translate_const)

  args = {
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Invert': no_arg,
      'Rotate': _rotate_level_to_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'SolarizeAdd': solarize_add_arg,
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': cutout_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
  }
  return args


def _parse_policy_info(
    name: Text,
    prob: float,
    level: float,
    replace_value: List[int],
    cutout_const: float,
    translate_const: float,
) -> Tuple[Any, float, Any]:
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = level_to_arg(cutout_const, translate_const)[name](level)

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  return func, prob, args


class ImageAugment(object):
  """Image augmentation class for applying image distortions."""

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """Given an image tensor, returns a distorted image with the same shape.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.

    Returns:
      The augmented version of `image`.
    """
    raise NotImplementedError()


class RandAugment(ImageAugment):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  """

  def __init__(
      self,
      num_layers: int = 2,
      magnitude: float = 10.0,
      cutout_const: float = 40.0,
      translate_const: float = 100.0,
  ):
    """Applies the RandAugment policy to images.

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 10].
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.
    """
    super(RandAugment, self).__init__()

    self.num_layers = num_layers
    self.magnitude = float(magnitude)
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)
    self.available_ops = [
        'AutoContrast',
        'Equalize',
        'Invert',
        'Rotate',
        'Posterize',
        'Solarize',
        'Color',
        'Contrast',
        'Brightness',
        'Sharpness',
        'ShearX',
        'ShearY',
        'TranslateX',
        'TranslateY',
        'Cutout',
        'SolarizeAdd',
    ]

  def distort(self, image: tf.Tensor):
    """Applies the RandAugment policy to `image`.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.

    Returns:
      The augmented version of `image`.
    """
    input_image_type = image.dtype
    # print(input_image_type)
    if input_image_type != tf.uint8:
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)

    replace_value = [128] * 3
    min_prob, max_prob = 0.2, 0.8

    for _ in range(self.num_layers):
      op_to_select = tf.random.uniform(
          [], maxval=len(self.available_ops) + 1, dtype=tf.int32
      )

      branch_fns = []
      for i, op_name in enumerate(self.available_ops):
        prob = tf.random.uniform(
            [], minval=min_prob, maxval=max_prob, dtype=tf.float32
        )
        func, _, args = _parse_policy_info(
            op_name,
            prob,
            self.magnitude,
            replace_value,
            self.cutout_const,
            self.translate_const,
        )
        branch_fns.append((
            i,
            # pylint: disable=g-long-lambda
            lambda selected_func=func, selected_args=args: selected_func(
                image, *selected_args
            ),
            # pylint: enable=g-long-lambda
        ))

      image = tf.switch_case(
          branch_index=op_to_select,
          branch_fns=branch_fns,
          default=lambda: tf.identity(image),
      )

    image = tf.cast(image, dtype=input_image_type)
    return image
