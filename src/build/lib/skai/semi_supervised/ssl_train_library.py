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

r"""Defines library of functions for model training binary.
"""

from absl import flags
from absl import logging
from skai import ssl_flags
from skai.semi_supervised import model_initializer
from skai.semi_supervised import train
from skai.semi_supervised import utils
from skai.semi_supervised.dataloader import prepare_ssl_data


flags.adopt_module_key_flags(ssl_flags)
FLAGS = flags.FLAGS


def create_dataset(shuffle: bool) -> prepare_ssl_data.SSLDataset:
  """Checks that provided examples are valid and creates SSLDataset object.

  Shared code for both internal and external use to be called efficiently in
  the ssl_train.py binary of each version.

  Args:
    shuffle: Bool for whether or not to shuffle.

  Returns:
    SSLDataset object.
  """
  logging.info('Loading dataset')
  if not FLAGS.inference_mode and FLAGS.train_label_examples is None:
    raise ValueError('Must specify non-empty list of train_label_examples.')
  if FLAGS.test_examples is None:
    raise ValueError('Must specify non-empty list of test_examples.')
  return prepare_ssl_data.create_dataset(
      name=FLAGS.dataset_name,
      train_label_filepatterns=FLAGS.train_label_examples,
      train_unlabel_filepatterns=FLAGS.train_unlabel_examples,
      test_filepatterns=FLAGS.test_examples,
      num_classes=FLAGS.num_classes,
      height=FLAGS.height,
      width=FLAGS.width,
      shuffle=shuffle,
      num_labeled_examples=FLAGS.num_labeled_examples,
      num_unlabeled_validation_examples=FLAGS.num_unlabeled_validation_examples,
      num_augmentations=FLAGS.num_augmentations,
      inference_mode=FLAGS.inference_mode,
      whiten=FLAGS.do_whiten,
      do_memoize=FLAGS.do_memoize,
      num_labeled_positives=FLAGS.num_labeled_positives,
      num_labeled_negatives=FLAGS.num_labeled_negatives,
      use_pre_disaster_image=FLAGS.use_pre_disaster_image)


def set_experiment_hyperparams():
  """Sets the default experiment hyperparameter values based on method."""
  if FLAGS.method == 'mixmatch':
    FLAGS.set_default('num_augmentations', 2)  # No strong augment
  elif FLAGS.method == 'fixmatch':
    FLAGS.set_default('num_augmentations', 1)  # Both strong and weak augment
    FLAGS.set_default('weight_decay', 0.0005)
    FLAGS.set_default('lr', 0.03)
  elif FLAGS.method == 'fully_supervised':
    FLAGS.set_default('weight_decay', 0.002)
    FLAGS.set_default('lr', 0.002)


def create_model(dataset: prepare_ssl_data.SSLDataset) -> train.ClassifySemi:
  """Creates model based on method and loads dataset.

  Shared code for both internal and external use to be called efficiently in
  the ssl_train.py binary of each version.

  Args:
    dataset: Input dataset for model.

  Returns:
    Semi-supervised learning model.
  """
  logging.info('Creating model')
  log_width = utils.ilog2(dataset.width)
  return model_initializer.create_model(FLAGS.method, dataset, log_width,
                                        FLAGS.flag_values_dict())


def launch_train_or_eval(model: train.ClassifySemi):
  """Launches evaluation on a single checkpoint or training, depending on flags.

  Shared code for both internal and external use to be called efficiently in
  the ssl_train.py binary of each version. Loads checkpoint if finetuning.

  Args:
    model: Initialized model that will be trained or evaluated.
  """
  if FLAGS.eval_ckpt:
    logging.info('Evaluating checkpoint')
    model.eval_checkpoint(FLAGS.eval_ckpt)
  elif not FLAGS.inference_mode:
    logging.info('Training model')
    if FLAGS.finetune_ckpt:
      logging.info('Will load checkpoint and then resume training.')
    model.train(
        FLAGS.batch,
        FLAGS.train_nimg,
        FLAGS.save_nimg,
        FLAGS.keep_ckpt,
        finetune_ckpt=FLAGS.finetune_ckpt)
