"""Dataloader for SKAI Images.

Supports augmentation, loading generated data, subsampling datasets.
"""

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# from tensorflow.keras import layers
# from tensorflow.keras import Sequential

from google3.experimental.users.tarunkalluri.SKAI_training.configs import default

MEAN_RGB = default.MEAN_RGB
STDDEV_RGB = default.STDDEV_RGB
CROP_SIZE = default.CROP_SIZE
IMAGE_RESIZE = default.IMAGE_RESIZE


def convert_img_to_save(img):
  img = unnormalize_image(img)
  img = np.clip((img.numpy()*255).astype(int), 0, 255)
  return tf.image.encode_jpeg(img)


def normalize_image(image):
  nchannels = image.shape.as_list()[-1]
  if nchannels == 3:
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  elif nchannels == 6:
    image -= tf.constant(MEAN_RGB*2, shape=[1, 1, 6], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB*2, shape=[1, 1, 6], dtype=image.dtype)
  else:
    raise NotImplementedError()
  return image


def unnormalize_image(image):
  image *= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image += tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def resize_small(image: tf.Tensor,
                 size: int,
                 *,
                 antialias: bool = False) -> tf.Tensor:
  """Resizes the smaller side to `size` keeping aspect ratio.

  Args:
    image: Single image as a float32 tensor.
    size: an integer, that represents a new size of the smaller side of an input
      image.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.
  """
  h, w = tf.shape(image)[0], tf.shape(image)[1]

  # Figure out the necessary h/w.
  ratio = (tf.cast(size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
  h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
  w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
  image = tf.image.resize(image, [h, w], antialias=antialias)
  return image


def preprocess_for_train(image,
                         dtype=tf.float32,
                         image_size=CROP_SIZE,
                         image_resize=IMAGE_RESIZE,
                         augmentation=None
                         ):
  """Preprocesses the given image for training.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    dtype: data type of the image.
    image_size: image size.
    image_resize: image resize.
    augmentation: Any augmentation to apply

  Returns:
    A preprocessed image `Tensor`.
  """
  ## replace custom function with library random crop function
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  ## resize the smaller dimension to 256.
  image = resize_small(image, size=image_resize)
  ## don't random crop a patch, center crop instead.
  # image = tf.image.random_crop(image, [image_size, image_size, 6])
  image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
  if augmentation:
    image = augmentation(image)
  image = normalize_image(image)

  return image


def preprocess_for_eval(image,
                        dtype=tf.float32,
                        image_size=CROP_SIZE,
                        image_resize=IMAGE_RESIZE):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    dtype: data type of the image.
    image_size: image size.
    image_resize: image resize

  Returns:
    A preprocessed image `Tensor`.
  """
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  image = resize_small(image, size=image_resize)
  image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
  image = normalize_image(image)

  return image


def subsample_dataset(ds,
                      frac,
                      config,
                      ds_size,
                      class_balanced=False):
  """Subsample `frac` fraction of data from the dataset."""
  assert isinstance(frac, (int, float))

  if class_balanced:
    ds_positive = ds.filter(lambda sample: sample['label'] == 1)
    ds_positive = ds_positive.shuffle(
        3000, seed=config.rng_seed, reshuffle_each_iteration=False
    )
    new_n_pos = round(frac * ds_size / 2)
    ds_positive = ds_positive.take(new_n_pos)

    ds_negative = ds.filter(lambda sample: sample['label'] == 0)
    ds_negative = ds_negative.shuffle(
        3000, seed=config.rng_seed, reshuffle_each_iteration=False
    )
    new_n_neg = round(frac * ds_size / 2)
    ds_negative = ds_negative.take(new_n_neg)

    n_take = new_n_pos + new_n_neg
    ds = ds_positive.concatenate(ds_negative)
  else:
    ds = ds.shuffle(
        ds_size,
        seed=config.rng_seed,
        reshuffle_each_iteration=False,
    )
    n_take = round(frac * ds_size)
    ds = ds.take(n_take)

  print('Pruned dataset from {} to {}'.format(ds_size, n_take))

  return ds


def create_train_ds(
    ds,
    *,
    config,
    dtype=tf.float32,
    skai_name=None,
    subset_frac=None,
    return_imgid=True,
):
  """Perform preprocessing and preparing the data."""

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)
  # augmentation = Sequential([
  #     layers.RandomFlip(seed=config.rng_seed),
  #     layers.RandomRotation(factor=0.2, seed=config.rng_seed),
  # ])
  augmentation = None

  def decode_example(example):
    if 'generated' in example:  ## deal the generated image differently
      image = preprocess_for_train(
          example['input_feature'],
          dtype,
          image_resize=IMAGE_RESIZE * 2,
          augmentation=augmentation,
      )
      before_image = image[:, :, :3]
      generated_image = preprocess_for_train(
          example['generated_image'],
          dtype,
          image_resize=IMAGE_RESIZE * 2,
          augmentation=augmentation,
      )
      image = tf.concat([before_image, generated_image], axis=-1)
    else:
      image = preprocess_for_train(
          example['input_feature']['small_image'],
          dtype,
          augmentation=augmentation,
      )

    if skai_name:
      subgrp_lbl = skai_name
    else:
      subgrp_lbl = example['subgroup_label']

    img_dict = {
        'image_feature': image,
        'label': example['label'],
        'string_label': example['string_label'],
        'subgroup_label': subgrp_lbl,
    }

    if return_imgid:
      img_dict['example_id'] = example['example_id']
      img_dict['prompt'] = example.get('prompt', '')

    return img_dict

  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ## sample subset of data
  if subset_frac:  ## class balanced data pruning
    ds_size = ds.cardinality().numpy()
    ds = subsample_dataset(
        ds, subset_frac, config, ds_size, config.balanced_pruning
    )
  if config.balanced_sampling:
    ds = ds.rejection_resample(
        class_func=lambda x: x['label'],
        target_dist=[0.6, 0.4],
        seed=config.rng_seed,
    ).map(lambda class_func_result, data: data)

  return ds


def create_eval_ds(
    ds,
    *,
    batch_size,
    dtype=tf.float32,
    prefetch=10,
    skai_name=None,
    return_imgid=False,
    drop_last=False,
):
  """Perform preprocessing and preparing the evaluation data."""

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  def decode_example(example):
    image = preprocess_for_eval(example['input_feature']['small_image'], dtype)

    if skai_name:
      subgrp_lbl = skai_name
    else:
      subgrp_lbl = example['subgroup_label']

    img_dict = {
        'image_feature': image,
        'label': example['label'],
        'string_label': example['string_label'],
        'subgroup_label': subgrp_lbl,
    }

    if return_imgid:
      img_dict['example_id'] = example['example_id']
      img_dict['prompt'] = example.get('prompt', '')

    return img_dict

  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if batch_size:
    ds = ds.batch(batch_size, drop_remainder=drop_last).prefetch(prefetch)

  return ds


# Read the data back out.
def decode_fn(record_bytes):
  """Decode the tfrecords to tensors."""

  example = tf.io.parse_single_example(
      # Data
      record_bytes,
      # # Schema
      {
          'prompt': tf.io.FixedLenFeature([], dtype=tf.string),
          'example_id': tf.io.FixedLenFeature([], dtype=tf.string),
          'string_label': tf.io.FixedLenFeature(
              [], dtype=tf.string, default_value=''
          ),
          'label': tf.io.FixedLenFeature([], dtype=tf.int64),
          'after_image_raw': tf.io.FixedLenFeature([], dtype=tf.string),
          'before_image_raw': tf.io.FixedLenFeature([], dtype=tf.string),
          'generated_image_raw': tf.io.FixedLenFeature([], dtype=tf.string),
          'generated': tf.io.FixedLenFeature([], dtype=tf.int64),
          'mask': tf.io.FixedLenFeature([], dtype=tf.string),
      },
  )

  features = {'input_feature': {}}
  before_image = decode_image(example['before_image_raw'])
  after_image = decode_image(example['after_image_raw'])
  generated_image = decode_image(example['generated_image_raw'])

  image_concat = tf.concat([before_image, after_image], axis=-1)

  features['input_feature'] = image_concat
  features['generated_image'] = generated_image
  features['label'] = tf.cast(example['label'], tf.int64)
  features['example_id'] = example['example_id']
  features['string_label'] = example['string_label']
  features['prompt'] = example['prompt']
  features['generated'] = tf.cast(example['generated'], tf.bool)
  features['mask'] = example['mask']
  return features


def decode_image(image_bytes) -> tf.Tensor:
  return tf.io.decode_image(
      image_bytes,
      channels=3,
      expand_animations=False,
      dtype=tf.float32,
  )


def get_training_data(
    skai_name,
    batch_size,
    config,
    return_unlabeled=False,
    labeled_root='/cns/ok-d/home/skai-dev/tensorflow_datasets/ttl=1y',
    subset_frac=None,
    **kwargs,
):
  """Get annotated/labeled training data."""

  (train, test, unlabeled), info = tfds.load(
      'skai_dataset/hurricane_%s' % (skai_name),
      data_dir=labeled_root,
      try_gcs=False,
      split=['labeled_train', 'labeled_test', 'unlabeled'],
      with_info=True,
  )

  len_train = info.splits['labeled_train'].num_examples
  if subset_frac:
    if subset_frac == -1:  ## full data
      subset_frac = 1.0
    elif subset_frac > 1:  ## num_samples instead of fraction.
      subset_frac = float(subset_frac/len_train)
    len_train = round(len_train*subset_frac)

  ## don't batch now. batch after joining datasets.
  train_ds = create_train_ds(
      train,
      config=config,
      skai_name=skai_name,
      subset_frac=subset_frac,
      **kwargs,
  )

  train_ds = train_ds.cache().shuffle(
      config.shuffle_buffer_size, seed=config.rng_seed
  ).repeat()

  test_ds = create_eval_ds(
      test,
      batch_size=batch_size,
      skai_name=skai_name,
      return_imgid=True
  )

  len_test = info.splits['labeled_test'].num_examples

  if return_unlabeled:
    unlabeled_ds = create_train_ds(
        unlabeled, config=config, **kwargs
    )
    unlabeled_ds = unlabeled_ds.cache().shuffle(
        config.shuffle_buffer_size, seed=config.rng_seed
    ).batch(config.batch)
    len_unlabeled = info.splits['unlabeled'].num_examples
    return {
        'train': (train_ds, len_train),
        'test': (test_ds, len_test),
        'unlabeled': (unlabeled_ds, len_unlabeled),
    }
  else:
    return {
        'train': (train_ds, len_train),
        'test': (test_ds, len_test),
    }


def get_generated_data(
    skai_name,
    config,
    generated_data_root='/cns/dl-d/home/jereliu/public/tarunkalluri/skai_sample_images/',
    matching_exp='*.tfrecords',
    subset_frac=None,
    **kwargs,
):
  """Get generated/synthetic training data."""

  assert isinstance(skai_name, str)

  tag = None
  edit_mode = None
  if config.edit_mode == 'mask_based':
    tag = 'zs_bb'
    edit_mode = 'mask_based'
  elif config.edit_mode == 'adapter_tune':
    tag = 'at_bb'
    edit_mode = 'adapter_tune'
  elif config.edit_mode == 'at_target_unsup':
    tag = 'at_tgt_bb'
    edit_mode = 'adapter_tune'

  root_name = generated_data_root
  dataroot = root_name + '{}/{}/{}_binary'.format(skai_name, tag, edit_mode)
  assert tf.io.gfile.exists(dataroot), dataroot
  print('Loading data from {}'.format(dataroot))
  lof = tf.io.gfile.glob(os.path.join(dataroot, matching_exp))

  gen_ds = tf.data.TFRecordDataset(filenames=lof)

  gen_ds = gen_ds.map(decode_fn)

  ## don't batch now. batch after joining datasets.
  train_ds = create_train_ds(
      gen_ds,
      config=config,
      skai_name=skai_name,
      subset_frac=None,
      **kwargs,
  )
  len_train = 5000
  if subset_frac is not None:
    if subset_frac == -1:
      n_take = len_train
    elif subset_frac > 1:
      n_take = subset_frac
    elif 0 < subset_frac <= 1:
      n_take = round(len_train * subset_frac)
    else:
      raise NotImplementedError

    train_ds = subsample_dataset(
        train_ds,
        n_take/len_train,
        config,
        len_train,
        class_balanced=config.balanced_pruning
    )
    len_train = n_take

  train_ds = train_ds.cache().shuffle(
      config.shuffle_buffer_size, seed=config.rng_seed
  ).repeat()

  return {'train': (train_ds, len_train), 'test': None}  ## load test separately


def prepare_data_loo(config):
  """Prepare the datasets for multi-domain training."""

  domains = config.training_domains  ## training_domains
  ood_domains = (
      config.leave_one
      if isinstance(config.leave_one, list)
      else [config.leave_one]
  )
  if not config.leave_one:
    ood_domains = config.training_domains[0:1]
  train = []
  test = []

  if not config.use_aug_only:
    for domain in domains:
      if config.data_fraction:
        dfrac = config.data_fraction[domain]
      else:
        dfrac = None

      ds = get_training_data(
          domain,
          config.batch,
          config,
          return_unlabeled=False,
          subset_frac=dfrac,
      )
      train.append(ds['train'])
      print(
          'Loaded {} samples from {}'.format(train[-1][1], domain), flush=True
      )

  if config.use_aug:
    for domain in ood_domains:
      ood_dataset = get_generated_data(
          skai_name=domain,
          config=config,
          subset_frac=config.data_fraction
      )
      train.append(ood_dataset['train'])
      print('Loaded {} samples from {}'.format(train[-1][1], domain))

  for domain in ood_domains[0:1]:
    ds = get_training_data(
        domain,
        config.batch,
        config,
        return_unlabeled=False,
    )
    test.append(ds['test'][0])
    print('Loaded {} test samples from {}'.format(ds['test'][1], domain))

  return train, test


def prepare_data_single(config):
  """Prepare the datasets for single-domain training."""

  if isinstance(config.training_domains, list):
    domain = config.training_domains[0]
  else:
    domain = config.training_domains

  train = []
  test = []

  ds = get_training_data(
      domain,
      config.batch,
      config,
      return_unlabeled=False,
      subset_frac=config.data_fraction,
  )
  train.append(ds['train'])
  test.append(ds['test'][0])
  print(
      'Loaded {} samples from {}'.format(train[-1][1], domain), flush=True
  )
  print('Loaded {} test samples from {}'.format(ds['test'][1], domain))

  return train, test


def prepare_data_semisup(config):
  """Prepare the datasets for single-domain semi-supervised training."""

  if isinstance(config.training_domains, list):
    domain = config.training_domains[0]
  else:
    domain = config.training_domains

  train = []
  test = []

  if not config.use_aug_only:
    if config.data_fraction:
      dfrac = config.data_fraction[domain]
    else:
      dfrac = None

    ds = get_training_data(
        domain,
        config.batch,
        config,
        return_unlabeled=False,
        subset_frac=dfrac,
    )
    train.append(ds['train'])
    print(
        'Loaded {} samples from {}'.format(train[-1][1], domain), flush=True
    )
    test.append(ds['test'][0])
    print('Loaded {} test samples from {}'.format(ds['test'][1], domain))

  if config.use_aug:
    ood_dataset = get_generated_data(
        skai_name=domain,
        config=config,
        subset_frac=config.data_fraction
    )
    train.append(ood_dataset['train'])
    print('Loaded {} samples from {}'.format(train[-1][1], domain))

  return train, test


