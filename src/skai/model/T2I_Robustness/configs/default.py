r"""Default config for training SKAI images using ViT backbone.
"""

import os

from vision_transformer.vit_jax.configs import augreg as augreg_config


IMAGE_RESIZE = 224
CROP_SIZE = 224
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def get_config(model_name):
  """Returns default parameters for finetuning ViT `model`."""

  # 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0'
  config = augreg_config.get_config(model_name)
  config = config.unlock()
  config.model_name = model_name

  # edit
  config.dataset = "skai"
  config.num_classes = 2
  config.batch = 64
  config.prefetch = 10
  config.warmup_iters = 100
  config.rng_seed = 1234
  config.num_training_iters = 5000
  config.iters_per_eval = 500
  # config.iters_per_checkpoint = 100
  config.iters_per_log = 100
  config.base_lr = 2e-6
  config.weight_decay = 1e-4
  config.root = "/cns/dl-d/home/jereliu/public/tarunkalluri/vit_base/"
  config.lr_multiplier = [1., 10.]  #  for backbone and classifier
  config.experiment_name = ""  # TBD by the config file.
  config.workdir = os.path.join(config.root, config.experiment_name)

  ## data
  config.image_resize = IMAGE_RESIZE
  config.crop_size = CROP_SIZE
  config.mean_rgb = MEAN_RGB
  config.stddev_rgb = STDDEV_RGB
  config.shuffle_buffer_size = 2048
  config.balanced_pruning = False
  config.balanced_sampling = False
  config.dataset_reweight = False

  return config
