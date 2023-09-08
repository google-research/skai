"""Train Utils."""

from absl import logging
import flax
from flax.training import checkpoints as flax_checkpoints
from flax.training import train_state
import jax
from jax.example_libraries.optimizers import clip_grads  ## pylint: disable=g-importing-member
import jax.numpy as jnp
import ml_collections
from ml_collections import config_dict
import optax
import orbax
from packaging import version
from vision_transformer.vit_jax import checkpoint


def create_learning_rate_fn(config, base_learning_rate):
  """Create Learning Rate.
  
  Args:
    config: configuration file
    base_learning_rate: base learning rate to use

  Returns:
    optim: optimizer fucntion to use for flax training
  """

  if config.get('lr_multiplier', None):
    assert isinstance(config.lr_multiplier, list)
    schedule_fns = []
    for lrm in config.lr_multiplier:
      base_lr = base_learning_rate * lrm

      warmup_fn = optax.linear_schedule(
          init_value=0, end_value=base_lr, transition_steps=config.warmup_iters
      )

      cosine_iters = max(1, config.num_training_iters - config.warmup_iters)
      cosine_schedule = optax.cosine_decay_schedule(
          init_value=base_lr,
          decay_steps=cosine_iters,
      )
      schedule_fns.append(
          optax.join_schedules(
              schedules=[warmup_fn, cosine_schedule],
              boundaries=[config.warmup_iters],
          )
      )

    return schedule_fns

  else:
    warmup_fn = optax.linear_schedule(
        init_value=0,
        end_value=base_learning_rate,
        transition_steps=config.warmup_iters,
    )

    cosine_iters = max(1, config.num_training_iters - config.warmup_iters)
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_iters,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_schedule], boundaries=[config.warmup_iters]
    )

    return schedule_fn


class TrainState(train_state.TrainState):
  dropout_rng: jax.random.KeyArray


def create_train_state(
    config: ml_collections.ConfigDict,
    model,
    learning_rate_fn,
    image_size,
    ckpt_params=None,
):
  """Create a custom training state.
  """
  rng = jax.random.PRNGKey(config.rng_seed)
  key1, key2, key3 = jax.random.split(rng, 3)

  # random initialization
  variables = model.init(
      {'params': key1}, jax.random.normal(key2, image_size), train=False
  )
  params = flax.core.freeze(variables['params'])

  ## create an optimizer

  if isinstance(learning_rate_fn, list):
    tx_adam_backbone = optax.adam(learning_rate=learning_rate_fn[0])
    tx_adam_classifier = optax.adam(learning_rate=learning_rate_fn[1])
    partition_optimizers = {
        'backbone': tx_adam_backbone,
        'classifier': tx_adam_classifier,
    }
    param_partitions = flax.core.freeze(
        flax.traverse_util.path_aware_map(
            lambda path, v: 'backbone' if 'backbone' in path else 'classifier',
            params,
        )
    )
    tx = optax.multi_transform(partition_optimizers, param_partitions)
  else:
    tx = optax.adam(learning_rate=learning_rate_fn)

  if ckpt_params is not None:
    params = flax.core.freeze(ckpt_params)

  state = TrainState.create(
      apply_fn=model.apply, params=params, tx=tx, dropout_rng=key3
  )

  return state


def compute_metrics(logits, labels):
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'accuracy': accuracy,
  }
  metrics = flax.jax_utils.pmean(metrics, axis_name='batch')
  return metrics


def one_train_step(image, label, state, learning_rate_fn, config):
  """One training step."""
  dkey = jax.random.fold_in(key=state.dropout_rng, data=state.step)

  # define loss function
  def loss_fn(params):
    logits = state.apply_fn(
        {'params': params}, image, train=True, rngs={'dropout': dkey}
    )

    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, label)
    )
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_l2 = sum(
        [0.5 * jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1]
    )

    weight_decay_loss = config.weight_decay * weight_l2
    loss = loss + weight_decay_loss

    return loss, logits

  ## compute the learning rate for the step
  if isinstance(learning_rate_fn, list):
    lr = learning_rate_fn[-1](state.step)
  else:
    lr = learning_rate_fn(state.step)

  ## compute the grad function - every step
  grad_fn = jax.value_and_grad(
      loss_fn, has_aux=True
      )
  (loss, (logits)), grad = grad_fn(state.params)

  grad = flax.jax_utils.pmean(grad, axis_name='batch')
  loss = flax.jax_utils.pmean(loss, axis_name='batch')

  if config.get('max_grad_norm') is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  new_state = state.apply_gradients(grads=grad, dropout_rng=dkey)

  ## compute the metrics.
  metrics = compute_metrics(logits, label)
  metrics.update({'learning_rate': lr, 'loss': loss})

  return new_state, metrics


def one_eval_step(state, image, label):
  """Just do evaluation, do not modify state."""

  logits = state.apply_fn(
      {'params': state.params}, image, train=False, mutable=False
  )
  metrics = compute_metrics(logits, label)
  loss = jnp.mean(
      optax.softmax_cross_entropy_with_integer_labels(logits, label)
  )
  loss = flax.jax_utils.pmean(loss, axis_name='batch')
  metrics['loss'] = loss
  return metrics, logits


def only_eval(state, image):
  """Just do evaluation, do not modify state."""

  logits = state.apply_fn(
      {'params': state.params}, image, train=False, mutable=False
  )
  return logits


def save_checkpoint(state, workdir, config):
  """Utils for saving and loading from pretrained checkpoints."""

  assert isinstance(config, config_dict.config_dict.ConfigDict)

  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
  step = int(state.step)
  logging.info('Saving checkpoint step %d.', step)

  save_obj = {'state': state, 'config': config, 'step': step}

  async_checkpointer = orbax.checkpoint.AsyncCheckpointer(
      orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=50
  )

  flax_checkpoints.save_checkpoint_multiprocess(
      workdir,
      save_obj,
      step=step,
      overwrite=True,
      keep=4,
      orbax_checkpointer=async_checkpointer,
  )

  async_checkpointer.wait_until_finished()


def load_checkpoint(workdir, state=None, step=None):
  # target = {'state': state, 'config': config_dict.ConfigDict(), 'step': 0}
  target = {'state': state, 'config': None, 'step': 0}

  ckpt = flax_checkpoints.restore_checkpoint(
      ckpt_dir=workdir, target=target, step=step
  )
  return ckpt


def load_pretrained(*, pretrained_path, init_params, model_config):
  """Loads/converts a pretrained checkpoint for fine tuning.

  Args:
    pretrained_path: File pointing to pretrained checkpoint.
    init_params: Parameters from model. Will be used for the head of the model
      and to verify that the model is compatible with the stored checkpoint.
    model_config: Configuration of the model. Will be used to configure the head
      and rescale the position embeddings.

  Returns:
    Parameters like `init_params`, but loaded with pretrained weights from
    `pretrained_path` and adapted accordingly.
  """

  restored_params = checkpoint.inspect_params(
      params=checkpoint.load(pretrained_path),
      expected=init_params,
      fail_if_extra=False,
      fail_if_missing=False)

  # The following allows implementing fine-tuning head variants depending on the
  # value of `representation_size` in the fine-tuning job:
  # - `None` : drop the whole head and attach a nn.Linear.
  # - same number as in pre-training means : keep the head but reset the last
  #    layer (logits) for the new task.
  if model_config.get('representation_size') is None:
    if 'pre_logits' in restored_params:
      logging.info('load_pretrained: drop-head variant')
      restored_params['pre_logits'] = {}
  if 'head' in init_params:
    restored_params['head']['kernel'] = init_params['head']['kernel']
    restored_params['head']['bias'] = init_params['head']['bias']
  if 'posembed_input' in restored_params.get('Transformer', {}):
    # Rescale the grid of position embeddings. Param shape is (1,N,1024)
    posemb = restored_params['Transformer']['posembed_input']['pos_embedding']
    posemb_new = init_params['Transformer']['posembed_input']['pos_embedding']
    if posemb.shape != posemb_new.shape:
      logging.info('load_pretrained: resized variant: %s to %s', posemb.shape,
                   posemb_new.shape)
      posemb = checkpoint.interpolate_posembed(
          posemb, posemb_new.shape[1], model_config.classifier == 'token')
      restored_params['Transformer']['posembed_input']['pos_embedding'] = posemb

  if version.parse(flax.__version__) >= version.parse('0.3.6') or flax.__version__ == 'google3-head':  # pylint: disable=line-too-long
    restored_params = checkpoint._fix_groupnorm(restored_params)  ## pylint: disable=protected-access

  return flax.core.freeze(restored_params)


def load_from_pretrained_vit(config, state):
  """Loads from a pretrained vit."""

  path = f'{config.pretrained_dir}/{config.model_name}.npz'
  init_params = flax.core.unfreeze(state.params)
  pt_params = load_pretrained(
      pretrained_path=path,
      init_params=init_params['backbone'],
      model_config=config.model)

  pt_params, _ = pt_params.pop('head')  ## we replace head with custom module
  init_params['backbone'] = pt_params
  state = state.replace(params=flax.core.freeze(init_params))
  del pt_params
  return state


