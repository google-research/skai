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
"""Library to prepare datasets to run MixMatch and FixMatch.

Defines SSLDataset class, which is dataset with format as required by the
MixMatch class. Original reference found here:
learning/brain/research/red_team/semi_supervised/libml/data.py
The steps are as follows:
1. Loads the example files from the given sources.
2. Splits the examples into a labeled train, unlabeled train, and test set.
3. Parses and processes the examples, e.g. whitening and augmenting them.
4. Creates and returns a SSLDataset instance.

When masks are being used, we assume both the pre- and post-disaster imagery
are available.
"""

from typing import Iterable, List, Mapping, Optional

from absl import logging
import numpy as np
from skai.semi_supervised import utils
import tensorflow.compat.v1 as tf


NUM_CHANNELS_PRE_DISASTER = 3
NUM_CHANNELS_POST_DISASTER = 3
NUM_MASK_CHANNELS = 2  # 1 channel for each mask black and white png
PARALLEL_PARSE = 4  # Number of parallel calls to make to parse dataset
PARALLEL_AUGMENT = 4  # Number of parallel calls to make to augment dataset

IMAGE_KEY = 'image'
LABEL_KEY = 'label'
COORDS_KEY = 'coordinates'
PRE_IMAGE_PNG_KEY = 'pre_image_png'
POST_IMAGE_PNG_KEY = 'post_image_png'
PRE_SEGMENTATION_PNG_KEY = 'pre_image_segmentations'
POST_SEGMENTATION_PNG_KEY = 'post_image_segmentations'


def random_flip(x: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
  """Randomly flips given image.

  Args:
    x: Input image.
    seed: Random seed.

  Returns:
    Flipped image.
  """
  x = tf.image.random_flip_left_right(x, seed=seed)
  x = tf.image.random_flip_up_down(x, seed=seed)
  return x


def random_shift(x: tf.Tensor,
                 w: int = 4,
                 seed: Optional[int] = None) -> tf.Tensor:
  """Randomly shifts given image by specified amount.

  Args:
    x: Input image.
    w: Max number of pixels to shift in any one direction.
    seed: Random seed.

  Returns:
    Randomly shifted image.
  """
  padded_x = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
  return tf.random_crop(padded_x, tf.shape(x), seed=seed)


AUGMENTATIONS = [random_flip, random_shift]  # Default augmentations of MixMatch


class SSLDataset:
  """Dataset class with format required for SSL code.

  Attributes:
    name: Name of the dataset.
    train_labeled: The tf.data.Dataset of the labeled training data.
    train_unlabeled: The tf.data.Dataset of the unlabeled training data.
    test: The tf.data.Dataset of the test data.
    unlabeled_validation_examples: The sample of unlabeled examples reserved for
      validation.
    eval_labeled: A version of the labeled training data for evaluation.
    eval_unlabeled: A version of the unlabeled training data for evaluation.
    height: Height of the imagery.
    width: Width of the imagery.
    channels: Number of channels.
    nclass: Number of classes.
    mean: Mean across channels in entire dataset, for whitening.
    std: Standard deviation across channels in entire dataset, for whitening.
    use_pre_disaster_image: Boolean that indicates pre-disaster imagery is
      imagery is available.
    p_labeled: DO NOT USE. Required for MixMatch code but not actively used.
    p_unlabeled: DO NOT USE. Required for MixMatch code but not actively used.
  """

  def __init__(self,
               name: str,
               train_labeled: tf.data.Dataset,
               train_unlabeled: tf.data.Dataset,
               test: tf.data.Dataset,
               unlabeled_validation_examples: tf.data.Dataset,
               eval_labeled: tf.data.Dataset,
               eval_unlabeled: tf.data.Dataset,
               height: int,
               width: int,
               channels: int,
               nclass: int,
               mean: float,
               std: float,
               use_pre_disaster_image: Optional[bool] = True,
               p_labeled: Optional[float] = None,
               p_unlabeled: Optional[float] = None):
    self.name = name
    self.train_labeled = train_labeled
    self.train_unlabeled = train_unlabeled
    self.eval_labeled = eval_labeled
    self.eval_unlabeled = eval_unlabeled
    self.test = test
    self.unlabeled_validation_examples = unlabeled_validation_examples
    self.height = height
    self.width = width
    self.channels = channels
    self.nclass = nclass
    self.mean = mean
    self.std = std
    self.use_pre_disaster_image = use_pre_disaster_image
    self.p_labeled = p_labeled
    self.p_unlabeled = p_unlabeled


def _parse_record(
    serialized_example: str, use_mask: bool,
    use_pre_disaster_image: bool) -> Mapping[str, tf.Tensor]:
  """Parse a record and return a dict for dataset.

  Args:
    serialized_example: String that specifies location of record to be parsed.
    use_mask: Boolean indicating whether to parse segmentation mask features.
    use_pre_disaster_image: Boolean indicating that pre-disaster images are
      available.

  Returns:
    A dict with an image and label key, where the image is a multi-channel image
    that has combined the pre- and post-disaster images.
  """
  features_config = {
      POST_IMAGE_PNG_KEY: tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.float32),
      'coordinates': tf.FixedLenFeature([2], tf.float32)
  }
  if use_pre_disaster_image:
    features_config[PRE_IMAGE_PNG_KEY] = tf.FixedLenFeature([], tf.string)
    if use_mask:
      features_config[PRE_SEGMENTATION_PNG_KEY] = tf.FixedLenFeature([],
                                                                     tf.string)
      features_config[POST_SEGMENTATION_PNG_KEY] = tf.FixedLenFeature([],
                                                                      tf.string)

  features = tf.parse_single_example(
      serialized_example, features=features_config)

  post_png = tf.image.decode_image(
      features[POST_IMAGE_PNG_KEY])
  if use_pre_disaster_image:
    pre_png = tf.image.decode_image(features[PRE_IMAGE_PNG_KEY])
    image_channels = [pre_png, post_png]
    # TODO(jlee24): use enum constants to define order and indices of
    # channels.
    # Expectations about the order of channels are hardcoded into fixmatch.py
    # (for summary images) and ctaugment.py (for only augmenting the pre and
    # post images).
    if use_mask:
      pre_mask = tf.image.decode_image(features[PRE_SEGMENTATION_PNG_KEY])
      post_mask = tf.image.decode_image(features[POST_SEGMENTATION_PNG_KEY])
      image_channels.append(pre_mask)
      image_channels.append(post_mask)
    image = tf.concat(image_channels, axis=-1)
  else:
    image = post_png
  image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
  label = tf.cast(features['label'], tf.int64)
  coordinates = tf.cast(features['coordinates'], tf.float64)

  return {IMAGE_KEY: image, LABEL_KEY: label, COORDS_KEY: coordinates}


def _parse_dataset(
    filenames: Iterable[str],
    shuffle: bool = False,
    use_mask: bool = False,
    use_pre_disaster_image: bool = True) -> tf.data.Dataset:
  """Parallel parsing of dataset.

  Args:
    filenames: List of TfRecord filenames to be loaded into a TFRecordDataset.
    shuffle: Boolean to shuffle dataset when true and do nothing otherwise.
    use_mask: Boolean indicating whether to parse segmentation mask features.
    use_pre_disaster_image: Boolean indicating that pre-disaster images are
      available.

  Returns:
    A TFRecordDataset with parsed examples, shuffled if shuffle is True.
  """
  filenames = sorted(sum([tf.gfile.Glob(x) for x in filenames], []))
  if shuffle:
    np.random.shuffle(filenames)
  ds = tf.data.TFRecordDataset(filenames)
  para = 4 * max(1, len(utils.get_available_gpus())) * PARALLEL_PARSE
  return ds.map(
      lambda x: _parse_record(x, use_mask, use_pre_disaster_image),
      num_parallel_calls=para)


def _compute_mean_std(ds: tf.data.Dataset) -> Iterable[float]:
  """Compute mean and standard deviation across entire training dataset.

  Args:
    ds: The tf.data.Dataset for which to compute mean and standard deviation.

  Returns:
    The mean and standard deviation per channel.
  """
  ds = ds.map(lambda x: x[IMAGE_KEY]).batch(1024).prefetch(1)
  ds = ds.make_one_shot_iterator().get_next()
  count = 0
  stats = []
  with tf.Session(config=utils.get_config()) as sess:

    def iterator():
      while True:
        try:
          yield sess.run(ds)
        except tf.errors.OutOfRangeError:
          break

    logging.info('Computing dataset mean and std')
    for batch in iterator():
      ratio = batch.shape[0] / 1024.
      count += ratio
      stats.append((batch.mean((0, 1, 2)) * ratio, (batch**2).mean(
          (0, 1, 2)) * ratio))
  mean = sum(x[0] for x in stats) / count
  sigma = sum(x[1] for x in stats) / count - mean**2
  std = np.sqrt(sigma)
  logging.info('Mean %d  Std: %d', mean, std)
  return mean, std


def _memoize(ds: tf.data.Dataset) -> tf.data.Dataset:
  """Store the dataset in memory to speed up access.

  Args:
    ds: The tf.data.Dataset to memoize.

  Returns:
    Returns a dataset that has been memoized.
  """
  data = []
  with tf.Session(config=utils.get_config()) as session:
    ds = ds.prefetch(16)
    it = ds.make_one_shot_iterator().get_next()
    try:
      while 1:
        data.append(session.run(it))
    except tf.errors.OutOfRangeError:
      pass
  images = np.stack([x[IMAGE_KEY] for x in data])
  labels = np.stack([x[LABEL_KEY] for x in data])
  coordinates = np.stack([x[COORDS_KEY] for x in data])

  def tf_get(index):
    image, label, coordinate = tf.py_func(
        lambda i: (images[i], labels[i], coordinates[i]), [index],
        [tf.float32, tf.int64, tf.float64])
    return {IMAGE_KEY: image, LABEL_KEY: label, COORDS_KEY: coordinate}

  ds = tf.data.Dataset.range(len(data))
  return ds.map(tf_get)


def get_example_files(patterns: List[str]) -> List[str]:
  """Retrieve the examples given the file pattern.

  Args:
    patterns: The file pattern(s) that specifies where the examples are located.

  Returns:
    A list of example filenames.

  Raises:
    ValueError when no example files are found with the given pattern.
  """
  all_filenames = []
  for example_pattern in patterns:
    filenames = tf.gfile.Glob(example_pattern)
    if not filenames:
      raise ValueError(f'No example files found for {example_pattern}.')
    all_filenames.extend(filenames)
  if not all_filenames:
    raise ValueError(f'No example files found among {patterns}.')
  return all_filenames


def _stack_augment(x: Mapping[str, tf.Tensor], num_augmentations: int,
                   expected_channels: int) -> Mapping[str, tf.Tensor]:
  """Give unlabeled data nu-augmentations and stack them.

  Function originally from:
  google3/learning/brain/research/red_team/semi_supervised/libml/data.py

  Args:
    x: Example to augment.
    num_augmentations: Number of augmentations for class-consistency.
    expected_channels: Number of channels to expect in the image feature.

  Returns:
    A version of the augmentation function that stacks augmented images.
  """
  imgs_to_stack = []
  labels_to_stack = []
  coords_to_stack = []
  for _ in range(num_augmentations):
    img_to_stack = tf.ensure_shape(x[IMAGE_KEY],
                                   (None, None, expected_channels))
    for augment_function in AUGMENTATIONS:
      img_to_stack = augment_function(img_to_stack)
    imgs_to_stack.append(img_to_stack)
    labels_to_stack.append(x[LABEL_KEY])
    coords_to_stack.append(x[COORDS_KEY])

  return {
      IMAGE_KEY: tf.stack(imgs_to_stack),
      LABEL_KEY: tf.stack(labels_to_stack),
      COORDS_KEY: tf.stack(coords_to_stack)
  }


def _weak_augment(train_label: tf.data.Dataset, train_unlabel: tf.data.Dataset,
                  num_augmentations: int, num_expected_channels: int):
  """Weakly augments the training data with default transformations.

  Args:
    train_label: Dataset object containing labeled training data.
    train_unlabel: Dataset object containing unlabeled training data.
    num_augmentations: Number of augmentations to perform per image.
    num_expected_channels: Number of channels to expect in the image feature.

  Returns:
    Augmented versions of labeled dataset and unlabeled dataset.

  """
  num_parallel_calls = max(1, len(
      utils.get_available_gpus())) * PARALLEL_AUGMENT
  # TODO(jlee24): Consider allowing user to specify augmentations by dict
  # that maps augmentation string name to function
  for augment_function in AUGMENTATIONS:
    train_label = train_label.map(
        lambda x: {  # pylint: disable=g-long-lambda
            IMAGE_KEY:
                augment_function(  # pylint: disable=cell-var-from-loop
                    tf.ensure_shape(x[IMAGE_KEY],
                                    (None, None, num_expected_channels))),
            LABEL_KEY:
                x[LABEL_KEY],
            COORDS_KEY:
                x[COORDS_KEY]
        },
        num_parallel_calls)
  train_unlabel = train_unlabel.map(
      lambda x: _stack_augment(x, num_augmentations, num_expected_channels),
      num_parallel_calls)
  return train_label, train_unlabel


# TODO(jlee24): Create multi-class version that selects arbitrary number of
# labeled examples
def take_balanced(input_ds: tf.data.Dataset, num_positives: int,
                  num_negatives: int, buffer_size: int):
  """Take a specified number of positive and negative examples from a dataset.

  Args:
    input_ds: Input dataset. Should contain image, label, and coords tensors.
    num_positives: Maximum number of positive examples to take.
    num_negatives: Maximum number of negative examples to take.
    buffer_size: Number of examples to sample positives and negatives from.
      Should be at least num_positives + num_negatives.

  Returns:
    Dataset of positive and negative examples.
  """

  def sample_balanced(batch):
    labels = batch[LABEL_KEY]
    images = batch[IMAGE_KEY]
    coords = batch[COORDS_KEY]
    neg_indexes = tf.squeeze(tf.where(tf.math.equal(labels, 0)), axis=1)
    neg_indexes = tf.slice(tf.random.shuffle(neg_indexes), [0], [num_negatives])
    pos_indexes = tf.squeeze(tf.where(tf.math.equal(labels, 1)), axis=1)
    pos_indexes = tf.slice(tf.random.shuffle(pos_indexes), [0], [num_positives])
    all_indexes = tf.concat((neg_indexes, pos_indexes), axis=0)
    shuffled_indexes = tf.random.shuffle(all_indexes)
    new_labels = tf.gather(labels, shuffled_indexes)
    new_images = tf.gather(images, shuffled_indexes)
    new_coords = tf.gather(coords, shuffled_indexes)
    return {
        LABEL_KEY: new_labels,
        IMAGE_KEY: new_images,
        COORDS_KEY: new_coords
    }

  return input_ds.batch(buffer_size).take(1).map(sample_balanced).unbatch()


# TODO(jlee24): Allow taking specific number of labeled examples in
# multi-class case
def create_dataset(name: str,
                   train_label_filepatterns: List[str],
                   train_unlabel_filepatterns: List[str],
                   test_filepatterns: List[str],
                   num_classes: int,
                   height: int,
                   width: int,
                   shuffle: bool,
                   num_labeled_examples: Optional[int],
                   num_unlabeled_validation_examples: int,
                   num_augmentations: int,
                   inference_mode: bool = False,
                   whiten: bool = False,
                   do_memoize: bool = True,
                   num_labeled_positives: int = 0,
                   num_labeled_negatives: int = 0,
                   use_mask: bool = False,
                   use_pre_disaster_image: bool = True) -> SSLDataset:
  """Create datasets with formats required by MixMatch and FixMatch.

  Args:
    name: Name of dataset.
    train_label_filepatterns: File pattern for labeled train examples.
    train_unlabel_filepatterns: File pattern for unlabeled train examples.
    test_filepatterns: File pattern for test examples.
    num_classes: Number of classes.
    height: Height of imagery in dataset.
    width: Width of imagery in dataset.
    shuffle: Boolean that, if true, shuffles filepatterns. Else, does nothing.
    num_labeled_examples: Number of examples to take from the labeled training
      dataset.
    num_unlabeled_validation_examples: Number of examples to sample from the
      unlabeled training set to validate model's performance on unlabeled data.
    num_augmentations: Number of augmentations.
    inference_mode: Boolean for inference mode, which requires only test data.
    whiten: Boolean that indiciates whether or not to whiten data.
    do_memoize: Boolean that indicates whether or not to memoize data.
    num_labeled_positives: Number of positive labeled examples to read.
    num_labeled_negatives: Number of negative labeled examples to read.
    use_mask: Boolean for adding building segmentation mask channels.
    use_pre_disaster_image: Boolean that indicates pre-disaster image is
      available.

  Returns:
    A SSLDataset object.
  """
  logging.info('Creating dataset %s', (name))
  logging.info('Retrieving test data')
  test_files = get_example_files(test_filepatterns)
  num_channels_total = NUM_CHANNELS_POST_DISASTER
  if use_pre_disaster_image:
    num_channels_total += NUM_CHANNELS_PRE_DISASTER
    if use_mask:
      num_channels_total += NUM_MASK_CHANNELS
  test = _parse_dataset(
      test_files,
      shuffle=shuffle,
      use_mask=use_mask,
      use_pre_disaster_image=use_pre_disaster_image)

  if inference_mode:
    return SSLDataset(
        name,
        train_labeled=[],
        train_unlabeled=[],
        test=test,
        unlabeled_validation_examples=0,
        eval_labeled=[],
        eval_unlabeled=[],
        height=height,
        width=width,
        channels=num_channels_total,
        nclass=num_classes,
        mean=0,
        std=1,
        use_pre_disaster_image=use_pre_disaster_image)

  logging.info('Retrieving training data')
  train_label_files = get_example_files(train_label_filepatterns)
  train_unlabel_files = get_example_files(train_unlabel_filepatterns)
  train_unlabel_files += train_label_files

  logging.info('Parsing training examples')
  train_label = _parse_dataset(
      train_label_files,
      shuffle=shuffle,
      use_mask=use_mask,
      use_pre_disaster_image=use_pre_disaster_image)

  if num_labeled_examples:
    if num_labeled_positives > 0 or num_labeled_negatives > 0:
      # Sample a specific number of positive and negative labeled examples.
      # Note that you should probably set num_labeled_examples to 10x of
      # (num_labeled_positives + num_labeled_negatives) to ensure that the
      # sample is large enough to get the desired number of each.
      train_label = take_balanced(train_label, num_labeled_positives,
                                  num_labeled_negatives, num_labeled_examples)
    else:
      train_label = train_label.take(num_labeled_examples)
  train_unlabel_orig = _parse_dataset(
      train_unlabel_files,
      shuffle=shuffle,
      use_mask=use_mask,
      use_pre_disaster_image=use_pre_disaster_image)
  unlabeled_validation_examples = train_unlabel_orig.take(
      num_unlabeled_validation_examples)
  train_unlabel = train_unlabel_orig.skip(num_unlabeled_validation_examples)

  # Prepare the dataset following steps from MixMatch's data.py
  # 1. Calculate stats if whitening distribution later
  if whiten:
    logging.info('Calculating mean and std for whitening')
    mean, std = _compute_mean_std(train_label.concatenate(train_unlabel))
  else:
    mean, std = 0, 1

  # 2. Memoize
  if do_memoize:
    logging.info('Memoizing training data')
    train_label = _memoize(train_label)
    train_unlabel = _memoize(train_unlabel)

  # 3. Augment
  logging.info('Weakly augmenting training data')
  train_label, train_unlabel = _weak_augment(
      train_label,
      train_unlabel,
      num_augmentations=num_augmentations,
      num_expected_channels=num_channels_total)

  return SSLDataset(
      name,
      train_labeled=train_label,
      train_unlabeled=train_unlabel,
      test=test,
      unlabeled_validation_examples=unlabeled_validation_examples,
      eval_labeled=_parse_dataset(
          train_label_files,
          shuffle=shuffle,
          use_mask=use_mask,
          use_pre_disaster_image=use_pre_disaster_image),
      eval_unlabeled=train_unlabel_orig.skip(num_unlabeled_validation_examples),
      height=height,
      width=width,
      channels=num_channels_total,
      nclass=num_classes,
      mean=mean,
      std=std,
      use_pre_disaster_image=use_pre_disaster_image)
