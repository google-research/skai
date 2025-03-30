"""COnfigurations for external VLM models.

Those models can be found here:

https://colab.sandbox.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb
"""

from typing import Optional

import ml_collections as mlc


_PUBLIC_WEBLI_MODELS = {
    ('B/16', 224): ('webli_en_b16_224_63724782.npz', 'B', 768, 64, 32_000),
    ('B/16', 256): ('webli_en_b16_256_60500360.npz', 'B', 768, 64, 32_000),
    ('B/16', 384): ('webli_en_b16_384_68578854.npz', 'B', 768, 64, 32_000),
    ('B/16', 512): ('webli_en_b16_512_68580893.npz', 'B', 768, 64, 32_000),
    ('L/16', 256): ('webli_en_l16_256_60552751.npz', 'L', 1024, 64, 32_000),
    ('L/16', 384): ('webli_en_l16_384_63634585.npz', 'L', 1024, 64, 32_000),
    ('So400m/14', 224): (
        'webli_en_so400m_224_57633886.npz',
        'So400m',
        1152,
        16,
        32_000,
    ),
    ('So400m/14', 384): (
        'webli_en_so400m_384_58765454.npz',
        'So400m',
        1152,
        64,
        32_000,
    ),
    ('B/16-i18n', 256): (
        'webli_i18n_b16_256_66117334.npz',
        'B',
        768,
        64,
        250_000,
    ),
}

TOKENIZERS = {
    32_000: 'c4_en',
    250_000: 'mc4',
}

_FOLDER = 'gs://big_vision/siglip/'


def _get_checkpoint_path(checkpoint_path):
  return _FOLDER + checkpoint_path


def get_model_config(
    model_type: str,
    image_variant: str,
    image_size: int,
    geofm_savedmodel_path: Optional[str],
) -> mlc.ConfigDict:
  """Returns the config for the given VLM model.

  Args:
    model_type: Represents model type.
    image_variant: Represents model variants.
    image_size: The size of the image.
    geofm_savedmodel_path: Path to the GeoFM exported SavedModel.

  Returns:
  Configuration dict that specifys how the model would be loaded. The
      configuration dict should have the following fields:
        - init_shapes: Tuple specifys the expected shape of the image and the
        tokenized text.
        - model: Dict specifys which model architecture to load.
        - model_init: Dict specifys model checkpoints.
        - evals: Dict specifys preprocessing functions for the image and text.
  """
  if model_type == 'siglip':
    config = get_siglip_config(image_variant, image_size)
  else:
    config = get_geofm_config(geofm_savedmodel_path)
  config.model_type = model_type
  return config


def get_siglip_config(
    image_variant: str,
    image_size: int
) -> mlc.ConfigDict:
  """Returns the config for the SigLIP model.
  """
  try:
    checkpoint, text_variant, embedding_dim, sequence_length, vocab_size = (
        _PUBLIC_WEBLI_MODELS[(image_variant, image_size)]
    )
  except KeyError as ex:
    raise ValueError(
        'The provided image_variant and image_size tuple are not supported: '
        f'{image_variant}, {image_size}'
    ) from ex

  checkpoint_path = _get_checkpoint_path(checkpoint)
  config = mlc.ConfigDict()
  config.model = mlc.ConfigDict()
  model_cfg = config.model
  model_cfg.image_model = 'vit'
  model_cfg.text_model = 'proj.image_text.text_transformer'
  model_cfg.image = dict(variant=image_variant, pool_type='map')
  model_cfg.text = dict(variant=text_variant, vocab_size=vocab_size)
  model_cfg.out_dim = (None, embedding_dim)  # (image_out_dim, text_out_dim)
  model_cfg.bias_init = -10.0
  model_cfg.temperature_init = 10.0

  config.evals = mlc.ConfigDict()
  config.evals.pp_img = f'resize({image_size})|value_range(-1, 1)'
  config.evals.pp_txt = (
      f'tokenize(max_len={sequence_length}, model="{TOKENIZERS[vocab_size]}",'
      ' eos="sticky", pad_value=1, inkey="texts")'
  )

  image_shape = (1, image_size, image_size, 3)
  text_shape = (1, sequence_length)
  config.init_shapes = (image_shape, text_shape)

  config.init_types = ('float32', 'int32')
  config.model_init = checkpoint_path
  return config


def get_geofm_config(savedmodel_path: str) -> mlc.ConfigDict:
  """Returns the config for the GeoFM model.
  """
  if not savedmodel_path:
    raise ValueError('GeoFM savedmodel_path cannot be empty.')
  config = mlc.ConfigDict()
  config.model = mlc.ConfigDict()
  config.savedmodel_path = savedmodel_path
  return config
