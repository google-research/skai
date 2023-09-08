"""Functions to load model for training."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class TwoStreamNet(nn.Module):
  """Two-tower model for binary classification for disaster assessment.
  
  Parallel encoders -> concatenate embeddings -> classify using MLP.
  """

  hidden_dim: int
  num_classes: int
  backbone: nn.Module
  last_layer_only: bool = False

  @nn.compact
  def __call__(self, inputs, *, train):

    before_image, after_image = jnp.split(inputs, 2, axis=-1)

    feature_before = self.backbone(before_image, train=train)
    feature_after = self.backbone(after_image, train=train)

    concat_feature = jnp.concatenate((feature_before, feature_after), axis=-1)

    if self.last_layer_only:
      concat_feature = jax.lax.stop_gradient(concat_feature)

    output = nn.Sequential([
        nn.Dense(features=self.hidden_dim, name='mlp'),
        nn.relu,
        nn.Dense(
            features=self.num_classes,
            name='output_head',
            # kernel_init=nn.initializers.zeros,
            # bias_init=nn.initializers.constant(self.backbone.head_bias_init)
        ),
    ])(concat_feature)

    return output
