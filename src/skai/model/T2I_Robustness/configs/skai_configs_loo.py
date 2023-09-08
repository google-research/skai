r"""Config file for leave-one-out-training.
"""

import os
import tensorflow as tf
from google3.experimental.users.tarunkalluri.SKAI_training.configs import default


MODEL_NAME = "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0"


def get_config_loo(
    loo_dataset="",
    pretrained_path="",
    model_name=MODEL_NAME,
    last_layer=False,
    use_aug_data=False,
    use_aug_only=False,
    load_checkpoint=False,
    suffix=""
):
  """Returns default parameters for finetuning ViT `model`."""

  config = default.get_config(model_name)
  config.train_setting = "loo"

  config.training_domains = ["ian", "michael", "maria", "laura"]
  # config.training_domains += ["florence"]
  config.data_fraction = None
  # {
  #     "maria": 1.0,
  #     "laura": 0.5,
  #     "ian": 0.2,
  #     "michael": 0.01,
  # }
  config.edit_mode = "mask_based"
  config.load_checkpoint = load_checkpoint
  config.last_layer_only = last_layer
  config.use_aug = use_aug_data
  config.use_aug_only = use_aug_only
  if config.use_aug_only:
    assert config.use_aug

  config.leave_one = loo_dataset
  if config.leave_one:
    config.training_domains.remove(config.leave_one)
    config.experiment_name = (
        "vit_skai_test_LOO_%s_AUG_%s_lastLayer_%s_fromPT_%s_TgtDataOnly_%s"
        % (
            config.leave_one,
            config.use_aug,
            config.last_layer_only,
            config.load_checkpoint,
            config.use_aug_only
        )
    )
  else:
    config.experiment_name = (
        "vit_skai_test_all_aug_%s_lastLayer_%s_fromPT_%s_TgtDataOnly"
        % (config.use_aug, config.last_layer_only, config.load_checkpoint)
    )

  config.experiment_name += suffix
  config.workdir = os.path.join(config.root, config.experiment_name)

  if tf.io.gfile.exists(config.workdir):
    config.load_checkpoint = True

  config.pretrained_path = pretrained_path
  if config.pretrained_path:
    assert tf.io.gfile.exists(config.pretrained_path)

  return config
