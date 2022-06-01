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

r"""Flags shared by ssl_eval.py and ssl_train.py.

This library allows us to have a single file defining flags used across semi-
supervised learning binaries. The Dataset Params and Evaluation Job flags are
used in ssl_eval.py. All flags except Evaluation Job flags are used in
ssl_train.py.
"""

from absl import flags
from skai.semi_supervised import classifiers
from skai.semi_supervised import fixmatch
from skai.semi_supervised import layers
from skai.semi_supervised import model_initializer

FLAGS = flags.FLAGS

# Dataset Params
flags.DEFINE_string('dataset_name',
                    None,
                    'Save name of the dataset. For example: '
                    'train_label=A_train_unlabel=B_test=C')
flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_string('train_dir',
                    None,
                    'Location to save training results.')
flags.DEFINE_list(
    'train_label_examples',
    None,
    'Comma-delimited list of file patterns for labeled training examples.'
    'When specified, can leave train_unlabel_examples and test_examples blank, '
    'and this value will be used.')
flags.DEFINE_list(
    'train_unlabel_examples',
    None,
    'Comma-delimited list of file patterns for unlabeled training examples.')
flags.DEFINE_list(
    'test_examples',
    None,
    'Comma-delimited list of file patterns for test examples.')
flags.DEFINE_integer(
    'num_labeled_examples', None,
    'Number of examples to take from train_label_examples.'
    'If None, uses all examples.')
flags.DEFINE_integer(
    'num_labeled_positives', 0,
    'Number of positive examples to sample from labeled data.'
    'When 0, uses all.')
flags.DEFINE_integer(
    'num_labeled_negatives', 0,
    'Number of negative examples to sample from labeled data.'
    'When 0, uses all.')
flags.DEFINE_integer(
    'num_unlabeled_validation_examples', 1,
    'Number of examples to sample from train_unlabel_examples to validate '
    'model performance on unlabeled data.')
flags.DEFINE_integer('shuffle_seed', 1,
                     'Set random seed. No shuffle when None.')
flags.DEFINE_integer('height', 64, 'Height of images in examples.')
flags.DEFINE_integer('width', 64, 'Width of images in examples.')
flags.DEFINE_boolean('do_whiten', False, 'Calculate mean, std to whiten data.')
flags.DEFINE_boolean('do_memoize', True, 'Memoize data to speed up training.')
flags.DEFINE_boolean(
    'inference_mode', False, 'Runs model in inference mode, which only expects '
    'test data (no training). Saves predictions and corresponding coordinate '
    'locations to GeoJSON files.')

# Training Hyperparameters
# For all models
flags.DEFINE_float('lr', 0.03, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay.')
flags.DEFINE_enum('arch', classifiers.RESNET,
                  classifiers.SUPPORTED_ARCHITECTURES, 'Model architecture.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
flags.DEFINE_integer('scales', 0,
                     'Number of 2x2 downscalings in the classifier.')
flags.DEFINE_integer('conv_filter_size', 32, 'Filter size of convolutions.')
flags.DEFINE_integer('num_residual_repeat_per_stage', 4,
                     'Number of residual layers per stage.')
flags.DEFINE_boolean('use_pre_disaster_image', True, 'Use both pre-disaster '
                     'and post-disaster image when reading data and '
                     'training.')

# For MixMatch Only
flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
flags.DEFINE_bool('logit_norm', False, 'Whether to use logit normalization.')
flags.DEFINE_float('sharpening_temperature', 0.5,
                   'MixMatch sharpening temperature.')
flags.DEFINE_enum('mixup_mode', 'xxy.yxy', layers.MixMode.MODES, 'Mixup mode')
flags.DEFINE_integer('dbuf', 128,
                     'Distribution buffer size to estimate p_model.')
flags.DEFINE_float('w_match', 1.5, 'Weight for distribution matching loss.')
flags.DEFINE_integer('warmup_kimg', 1024,
                     'Warmup in kimg for the matching loss.')
flags.DEFINE_integer('num_augmentations', 2,
                     'Number of augmentations for class-consistency.')

# For FixMatch Only
flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
flags.DEFINE_float('pseudo_label_loss_weight', 1, 'Pseudo label loss weight.')
flags.DEFINE_integer('unlabeled_ratio', 7, 'Unlabeled batch size ratio.')
flags.DEFINE_integer('num_parallel_calls', 2,
                     'Number of parallel calls to make per machine when '
                     'augmenting data.')
strategy_options = [option.name for option in fixmatch.StrategyOptions]
flags.DEFINE_enum('augmentation_strategy', strategy_options[0],
                  strategy_options,
                  'Augmentation strategy to use when using FixMatch.')

# For Fully Supervised Only
flags.DEFINE_float('embedding_layer_dropout_rate', 0,
                   'Dropout rate of embedding layer.')
flags.DEFINE_float('smoothing', 0.001, 'Label smoothing.')

# Experiment Parameters
flags.DEFINE_integer('train_nimg', 1 << 24,
                     'Training duration in number of examples.')
flags.DEFINE_integer('save_nimg', 64 << 10,
                     'Save checkpoint period in number of examples.')
flags.DEFINE_integer('keep_ckpt', 10000, 'Number of checkpoints to keep. '
                     'Default has been set high to retain all checkpoints, '
                     'which also comes in handy when fine-tuning from a '
                     'specific point in training.')
flags.DEFINE_string(
    'eval_ckpt', '',
    'Checkpoint to evaluate. If provided, do not do training, just do eval.')
flags.DEFINE_string(
    'finetune_ckpt', '',
    'Checkpoint to load before resuming training. Used to finetune model.')
flags.DEFINE_enum('method', 'fixmatch',
                  model_initializer.MODELS.keys(),
                  'Method to use to train model.')

# Evaluation Job Parameters
flags.DEFINE_integer(
    'last_epoch', 0,
    'This job terminates after it has processed the checkpoint for this epoch.')
flags.DEFINE_boolean('save_predictions', True,
                     'Whether or not to save predictions in CSV files.')
