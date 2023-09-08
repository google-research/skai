"""Training framework."""

import functools
import time

from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.training import common_utils
import jax
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import notebook
from vision_transformer.vit_jax import models_vit

# from data import prepare_data_loo
from google3.experimental.users.tarunkalluri.SKAI_training import data
from google3.experimental.users.tarunkalluri.SKAI_training import metrics
from google3.experimental.users.tarunkalluri.SKAI_training import model
from google3.experimental.users.tarunkalluri.SKAI_training import train_utils


def shard_data(x):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _shard(x):
    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)

    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_util.tree_map(_shard, x)


def train_and_evaluate(
    config
):
  """The training and evaluations script.
  """

  ## first, load the datasets
  train = []
  test = []

  if config.train_setting == "loo":
    train, test = data.prepare_data_loo(config)
  elif config.train_setting == "single":
    train, test = data.prepare_data_single(config)
  elif config.train_setting == "semisup":
    train, test = data.prepare_data_semisup(config)

  assert train, "Atleast one domain required for training."

  n_train_samples = [d[1] for d in train]

  if config.dataset_reweight:
    weights = [t / sum(n_train_samples) for t in n_train_samples]
  else:
    weights = [1 / len(train)] * len(train)
  print("Reweighting the dataset with weights {}".format(weights))

  train_ds = tf.data.Dataset.sample_from_datasets(
      [d[0] for d in train],
      weights,
      seed=config.rng_seed,
      stop_on_empty_dataset=True,
  )

  train_ds = tfds.as_numpy(
      train_ds.batch(config.batch, drop_remainder=True).prefetch(
          config.prefetch
      )
  )

  if test:
    test_ds = tfds.as_numpy(test[0])
  else:
    test_ds = None

  network = model.TwoStreamNet(
      hidden_dim=config.model.hidden_size,
      num_classes=2,
      backbone=models_vit.VisionTransformer(num_classes=None, **config.model),
      last_layer_only=config.last_layer_only,
  )

  ## then, create the optimizers
  learning_rate_fn = train_utils.create_learning_rate_fn(config, config.base_lr)

  # ## then create a train state
  state = train_utils.create_train_state(
      config,
      network,
      learning_rate_fn,
      image_size=(1, config.crop_size, config.crop_size, 6),
  )

  state = train_utils.load_from_pretrained_vit(config, state)

  # load checkpoints, if any.
  step_offset = state.step
  if config.load_checkpoint:
    try:
      print("Loading from given checkpoint")
      assert config.pretrained_path
      assert tf.io.gfile.exists(config.pretrained_path)
      ckpt = train_utils.load_checkpoint(config.pretrained_path, state=state)
      pt_params = ckpt["state"].params
      state = train_utils.create_train_state(
          config,
          network,
          learning_rate_fn,
          image_size=(1, config.crop_size, config.crop_size, 6),
          ckpt_params=pt_params,
      )
      del pt_params, ckpt
    except OSError as _:
      print("Loading from existing checkpoint")
      ckpt = train_utils.load_checkpoint(config.workdir, state=state)
      state = ckpt["state"]
      step_offset = state.step

  ## then replicate this onto all devices
  state = jax_utils.replicate(state)

  ## start the training loop
  p_train_step = jax.pmap(
      functools.partial(
          train_utils.one_train_step,
          learning_rate_fn=learning_rate_fn,
          config=config,
      ),
      axis_name="batch",
  )
  p_eval_step = jax.pmap(train_utils.one_eval_step, axis_name="batch")

  train_metrics = []
  hooks = []
  train_metrics_last_t = time.time()

  best_auprc = 0
  best_state = state

  writer = metric_writers.create_default_writer(
      logdir=config.workdir, just_logging=jax.process_index() != 0
  )

  if jax.process_index() == 0:
    hooks += [
        periodic_actions.Profile(num_profile_steps=5, logdir=config.workdir),
    ]

  for step, batch in zip(
      range(step_offset, config.num_training_iters), train_ds
      ):

    if step == step_offset: print("Initial compilation completed")

    batch = shard_data(batch)

    image, label = batch["image_feature"], batch["label"]
    state, batch_metrics = p_train_step(image, label, state)

    for h in hooks:
      h(step)

    train_metrics.append(batch_metrics)

    if (step + 1) % config.iters_per_log == 0:
      train_metrics = common_utils.get_metrics(train_metrics)
      summary = {
          f"train_{k}": v
          for k, v in jax.tree_util.tree_map(
              lambda x: x.mean(), train_metrics
          ).items()
      }
      summary["steps_per_second"] = config.iters_per_log / (
          time.time() - train_metrics_last_t
      )
      writer.write_scalars(step + 1, summary)

      print(
          "Train: step: (%d/%d), loss: %.4f, accuracy: %.2f, steps_per_second:"
          " %.2f, learning_rate: %.7f"
          % (
              step + 1,
              config.num_training_iters,
              summary["train_loss"],
              summary["train_accuracy"] * 100,
              summary["steps_per_second"],
              summary["train_learning_rate"],
          ),
          flush=True,
      )

      ## reset stats
      train_metrics = []
      train_metrics_last_t = time.time()

    if ((step + 1) % (config.iters_per_eval) == 0) or (
        (step + 1) == config.num_training_iters
    ):
      eval_metrics = []
      all_scores = []
      all_labels = []
      ## collect state from all devices at once
      for batch in test_ds:
        image = batch["image_feature"]
        label = batch["label"]

        batch_eval_metrics, batch_logits = flax.jax_utils.pad_shard_unpad(
            p_eval_step
        )(state, image, label)
        scores = (
            jax.nn.softmax(batch_logits, axis=-1)[:, 1].reshape(-1).tolist()
        )

        eval_metrics.append(batch_eval_metrics)
        all_scores.extend(scores)
        all_labels.extend(label.tolist())

      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)

      df = pd.DataFrame({"score": all_scores, "label": all_labels})
      f1, auprc, cls_avg = metrics.binary_metrics(
          df, threshold=0.5, print_summary=False
      )
      summary.update({"f1": f1, "auprc": auprc, "cls_avg": cls_avg})

      if auprc > best_auprc:
        best_auprc = auprc
        train_utils.save_checkpoint(state, config.workdir, config)
        best_state = state

      print(
          "Eval: Iter: %d, loss: %.4f, accuracy: %.2f, f1: %.2f, auprc: %.2f, cls_avg: %.2f, best_auprc: %.2f"  # pylint: disable=line-too-long
          % (
              step + 1,
              summary["loss"],
              summary["accuracy"] * 100,
              summary["f1"] * 100,
              summary["auprc"] * 100,
              summary["cls_avg"] * 100,
              best_auprc * 100,
          ),
          flush=True,
      )

      writer.write_scalars(
          step + 1, {f"eval_{k}": v for k, v in summary.items()}
      )
      writer.flush()

  return best_auprc, jax_utils.unreplicate(best_state)


def evaluate(cfg, ds_name, load_dir=None, step=None, show=False):
  """Evaluation module.
  
  Args:
    cfg: config file.
    ds_name: evaluation dataset
    load_dir: directory to load from.
    step: [Optional] step to load checkpoint from.
    show: print results

  Returns:
    best_metrics: evaluation metrics.
  """
  eval_ds = tfds.load(
      "skai_dataset/hurricane_%s" % (ds_name),
      data_dir="/cns/ok-d/home/skai-dev/tensorflow_datasets/ttl=1y",
      try_gcs=False,
      split="labeled_test",
  )
  eval_ds = tfds.as_numpy(
      data.create_eval_ds(
          eval_ds,
          batch_size=cfg.batch,
          skai_name=ds_name,
          return_imgid=True,
          drop_last=False,
      )
  )

  print("Loaded data from %s" % (ds_name), flush=True)

  network = model.TwoStreamNet(
      hidden_dim=cfg.model.hidden_size,
      num_classes=2,
      backbone=models_vit.VisionTransformer(num_classes=None, **cfg.model),
      last_layer_only=cfg.last_layer_only,
  )

  learning_rate_fn = train_utils.create_learning_rate_fn(cfg, cfg.base_lr)

  # ## then create a train state
  state = train_utils.create_train_state(
      cfg,
      network,
      learning_rate_fn,
      image_size=(1, cfg.crop_size, cfg.crop_size, 6),
  )

  all_scores = []
  all_labels = []
  pre_images = []
  post_images = []
  example_ids = []

  p_only_eval = jax.pmap(train_utils.only_eval, axis_name="batch")

  if step:
    load_step = step
  else:
    load_step = None

  if not load_dir:
    load_dir = cfg.workdir

  ckpt = train_utils.load_checkpoint(load_dir, state=state, step=load_step)
  state = ckpt["state"]

  state = jax_utils.replicate(state)

  for batch in notebook.tqdm(eval_ds):

    images = batch["image_feature"]
    labels = batch["label"].tolist()
    imgids = batch["example_id"].tolist()

    logits = flax.jax_utils.pad_shard_unpad(p_only_eval)(state, images)

    scores = jax.nn.softmax(logits, axis=-1)[:, 1].reshape(-1).tolist()

    all_scores.extend(list(scores))
    all_labels.extend(list(labels))
    pre_imagery, post_imagery = np.split(batch["image_feature"], 2, axis=-1)
    pre_images.extend([data.convert_img_to_save(img) for img in pre_imagery])
    post_images.extend([data.convert_img_to_save(img) for img in post_imagery])
    example_ids.extend([b.decode() for b in imgids])

  df = pd.DataFrame({
      "score": all_scores,
      "label": all_labels,
      "pre_image": pre_images,
      "post_image": post_images,
      "example_id": example_ids,
  })

  f1, auprc, avg_acc = metrics.binary_metrics(
      df, threshold=0.5, print_summary=False
  )

  acc_vals = {"f1": f1, "auprc": auprc, "avg_acc": avg_acc}

  if show:
    print(
        "{}: {:.2f}/{:.2f}/{:.2f}".format(
            state.step[0],
            acc_vals["auprc"] * 100,
            acc_vals["f1"] * 100,
            acc_vals["avg_acc"] * 100,
        ),
        end="\t",
    )

  return df, acc_vals



