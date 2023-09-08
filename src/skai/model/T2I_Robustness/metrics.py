"""Functions to compute mertics for binary classification."""

import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics
import tensorflow as tf


def print_response(response):
  for answer, score in zip(response.outputs, response.scores):
    answer = answer.decode('utf-8')
    answer = answer.removeprefix('<extra_id_0> ')
    print(f'{score:.4f}  {answer}')


def visualize_example(pre_image, post_image):
  fig, axes = plt.subplots(1, 2)
  before_image = tf.io.decode_image(pre_image)
  after_image = tf.io.decode_image(post_image)
  axes[0].imshow(before_image)
  axes[1].imshow(after_image)
  plt.show(fig)


def parse_model_output(answers, scores):
  """Parse answers."""

  yes_score = None
  no_score = None
  maybe_score = None
  for answer, score in zip(answers, scores):
    answer = answer.decode('utf-8')
    answer = answer.removeprefix('<extra_id_0> ')
    if answer == 'yes':
      yes_score = score
    elif answer == 'no':
      no_score = score
    elif answer == 'maybe':
      maybe_score = score
  return yes_score, no_score, maybe_score


def compute_accuracy(df):
  num_correct = np.count_nonzero(df.label == df.prediction)
  return num_correct / len(df.label)


def classavg_accuracy(df):
  gt_labels = df.label.unique().tolist()
  acc_label = {}
  for gl in gt_labels:
    df_label = df[df.label == gl]
    num_correct = np.count_nonzero(df_label.label == df_label.prediction)
    acc_label[gl] = num_correct / len(df_label.label)
  return sum(acc_label.values()) / len(acc_label)


def compute_accuracy_at_threshold(df, threshold):  ## makes sense for binary.
  num_correct = np.count_nonzero(
      df.label == (df.score >= threshold).astype(float)
  )
  return num_correct / len(df.label)


def plot_subgroup_accuracies(df: pd.DataFrame, column: str):
  values = df[column].unique()
  thresholds = np.arange(0, 1, 0.05)
  results = pd.DataFrame({'threshold': thresholds})
  for value in values:
    subgroup = df[df[column] == value]
    subgroup_size = len(subgroup)
    results[value] = [
        np.count_nonzero(subgroup.label == (subgroup.score > t)) / subgroup_size
        for t in thresholds
    ]
  results.set_index('threshold').plot()


def compute_subgroup_stats(df: pd.DataFrame, column: str):
  values = df[column].unique()
  cols = collections.defaultdict(list)
  for value in values:
    cols['value'].append(value)
    subgroup_df = df.loc[df[column] == value]
    cols['size'].append(len(subgroup_df))
    cols['positives'].append(sum(subgroup_df.label == 1.0))
    cols['negatives'].append(sum(subgroup_df.label == 0.0))
    cols['auprc'].append(compute_auprc(subgroup_df))
  return pd.DataFrame(cols)


# def plot_precision_recall(df):
#   sklearn_metrics.PrecisionRecallDisplay.from_predictions(df.label, df.score)
#   plt.title('PR Curve')
#   plt.ylim((0, 1))
#   plt.grid()
#   plt.show()

#   precision, recall, thresholds = sklearn_metrics.precision_recall_curve(
#       df.label, df.score
#   )
#   x = pd.DataFrame({
#       'threshold': thresholds,
#       'precision': precision[:-1],
#       'recall': recall[:-1],
#   })
#   sns.lineplot(data=x.set_index('threshold'))
#   plt.title('Precision/Recall vs. Threshold')
#   plt.xlim((0, 1))
#   plt.ylim((0, 1))
#   plt.grid()
#   plt.show()


def compute_auroc(df):
  return sklearn_metrics.roc_auc_score(df.label, df.score)


def compute_auprc(df):
  precision, recall, _ = sklearn_metrics.precision_recall_curve(
      df.label, df.score
  )
  return sklearn_metrics.auc(recall, precision)


def compute_f1(df):
  return sklearn_metrics.f1_score(df.label, df.prediction)


def compute_precision(df):
  return sklearn_metrics.precision_score(df.label, df.prediction)


def compute_recall(df):
  return sklearn_metrics.recall_score(df.label, df.prediction)


def compute_precision_at_recall(
    df: pd.DataFrame, target_recall: float
) -> float:
  precision, recall, thresholds = sklearn_metrics.precision_recall_curve(
      df.label, df.score
  )
  for i, r in enumerate(recall):
    if r < target_recall:
      t = thresholds[i - 1]
      return precision[i - 1], t, compute_accuracy_at_threshold(df, t)
  return None


def compute_recall_at_precision(
    df: pd.DataFrame, target_precision: float
) -> float:
  precision, recall, thresholds = sklearn_metrics.precision_recall_curve(
      df.label, df.score
  )
  for i, p in enumerate(precision):
    if p > target_precision:
      t = thresholds[i - 1]
      return recall[i - 1], t, compute_accuracy_at_threshold(df, t)
  return None


def compute_precision_recall_at_cross(df: pd.DataFrame) -> float:
  precision, recall, thresholds = sklearn_metrics.precision_recall_curve(
      df.label, df.score
  )
  diff = np.abs(precision - recall)
  min_i = np.argmin(diff)
  return (
      thresholds[min_i],
      precision[min_i],
      recall[min_i],
      compute_accuracy_at_threshold(df, thresholds[min_i]),
  )


def plot_confusion_matrix(df):
  sklearn_metrics.ConfusionMatrixDisplay.from_predictions(
      df.label, df.prediction
  )
  plt.title('Confusion matrix')
  plt.show()

  plt.title('Confusion matrix')
  plt.show()


# def plot_score_distribution(df):
#   sns.displot(data=df, x='score', hue='string_label')
#   plt.yscale('log')
#   plt.show()


def plot_example_images(df, n, cols=3):
  """Plot example images."""
  n = min(len(df), n)
  rows = (n // cols) + int(n % cols > 0)
  print(rows)
  size_factor = 3
  fig_size = (cols * 2 * size_factor, rows * size_factor)
  fig, axes = plt.subplots(rows, cols * 2, figsize=fig_size)
  for row in range(rows):
    for col in range(cols):
      if rows > 1:
        ax1 = axes[row, 2 * col]
        ax2 = axes[row, 2 * col + 1]
      else:
        ax1 = axes[2 * col]
        ax2 = axes[2 * col + 1]
      ax1.axis('off')
      ax2.axis('off')
      i = col + row * cols
      if i < n:
        pre_image = tf.image.decode_image(df.iloc[i]['pre_image']).numpy()
        post_image = tf.image.decode_image(df.iloc[i]['post_image']).numpy()
        ax1.imshow(pre_image)
        ax1.set_title('before')
        ax2.imshow(post_image)
        score = df.iloc[i]['score']
        ax2.set_title(f'after ({score:.3f})')
  plt.show(fig)


def plot_positives(df, n, threshold):
  positives = df[df.score >= threshold]
  print(f'{len(positives)} positives in dataset (out of {len(df)})')
  plot_example_images(positives.sample(min(len(positives), n)), n)


def plot_negatives(df, n, threshold):
  negatives = df[df.score < threshold]
  print(f'{len(negatives)} negatives in dataset (out of {len(df)})')
  plot_example_images(negatives.sample(min(len(negatives), n)), n)


def plot_false_positives(df, n, threshold, choose_worst=False):
  fp = df[(df.label == 0) & (df.score >= threshold)]
  print(f'{len(fp)} false positives in dataset (out of {len(df)})')
  fp = fp.sort_values(by='score', ascending=False)
  if choose_worst:
    sample = fp.iloc[:n]
  else:
    sample = fp.sample(min(len(fp), n))
  plot_example_images(sample, n)


def plot_false_negatives(df, n, threshold, choose_worst=False):
  fn = df[(df.label == 1) & (df.score < threshold)]
  print(f'{len(fn)} false negatives in dataset (out of {len(df)})')
  fn = fn.sort_values(by='score', ascending=True)
  if choose_worst:
    sample = fn.iloc[:n]
  else:
    sample = fn.sample(min(len(fn), n))
  plot_example_images(sample, n)


def binary_metrics(df: pd.DataFrame, threshold: float, print_summary=True):
  """Compute binary metrics."""

  num_positives = sum(df.label == 1.0)
  num_negatives = sum(df.label == 0.0)

  df['prediction'] = df.score > threshold

  cm = sklearn_metrics.confusion_matrix(df.label, df.prediction)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  neg_accuracy, pos_accuracy = cm.diagonal()

  cls_avg_accuracy = (neg_accuracy + pos_accuracy) / 2.0

  if print_summary:

    print(f'Num positive labels: {num_positives}')
    print(f'Num negative labels: {num_negatives}')
    print()

    print(f'Accuracy:  {compute_accuracy(df):.4g}')
    print(f'Pos. Acc:  {pos_accuracy:.4g}')
    print(f'Neg. Acc:  {neg_accuracy:.4g}')
    print(f'F1-score:  {compute_f1(df):.4g}')
    print(f'Precision: {compute_precision(df):0.4g}')
    print(f'Recall:    {compute_recall(df):0.4g}')
    p, t, a = compute_precision_at_recall(df, 0.7)
    print(f'Precision@R70: p{p:.3f} t{t:.3f} a{a:.3f}')
    r, t, a = compute_recall_at_precision(df, 0.7)
    print(f'Recall@P70:    r{r:.3f} t{t:.3f} a{a:.3f}')
    t, p, r, a = compute_precision_recall_at_cross(df)
    print(f'Cross point:   p{p:.3f} r{r:.3f} t{t:.3f} a{a:.3f}')

    if num_positives > 0 and num_negatives > 0:
      print(f'AUROC:     {compute_auroc(df):.4g}')
      print(f'AUPRC:     {compute_auprc(df):.4g}')
    else:
      print('AUROC and AUPRC skipped as there is only one label class.')

    # plot_precision_recall(df)
    # plot_subgroup_accuracies(df, 'string_label')
    # plot_confusion_matrix(df)
    # plot_score_distribution(df)

  f1 = compute_f1(df)
  auprc = compute_auprc(df)

  return f1, auprc, cls_avg_accuracy
