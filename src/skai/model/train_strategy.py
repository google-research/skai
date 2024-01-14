"""Train Strategy file.

This creates the strategy for specified accelerator, cpu, gpu or tpu.
"""
from typing import Union
import tensorflow as tf


_Strategy = Union[
    tf.distribute.Strategy,
    tf.distribute.MirroredStrategy,
    tf.distribute.TPUStrategy
]


def get_tpu_resolver(tpu: str|None = 'local'):
  """Create cluster resolver for Cloud TPUs.

  Args:
    tpu: TPU to use - name, worker address or 'local'.

  Returns:
    TPUClusterResolver for Cloud TPUs.
  """
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  return resolver


def get_strategy(
    accelerator_type: str, tpu_bns: str|None = 'local'
) -> _Strategy:
  """Gets distributed training strategy for accelerator type.

  Args:
    accelerator_type: The accelerator type which is one of cpu, gpu and tpu.
    tpu_bns: A string of the Headless TPU Worker's BNS address.

  Returns:
    MirroredStrategy for gpu accelerator,
        TPUStrategy for tpu,
        and default Strategy for cpu.
  """
  if accelerator_type == 'gpu':
    return tf.distribute.MirroredStrategy()
  elif accelerator_type == 'tpu':
    resolver = get_tpu_resolver(tpu_bns)
    return tf.distribute.TPUStrategy(resolver)
  return tf.distribute.get_strategy()
