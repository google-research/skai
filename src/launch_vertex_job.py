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

"""Launcher script to launch training and evaluation jobs on Vertex AI.

Run this script `python launch_vertex_train.py` from the command line to launch
a training job.
"""

import sys
from typing import Any, Dict, List

from absl import app
from absl import flags
from google.cloud import aiplatform
from skai import ssl_flags
from skai import utils


flags.adopt_module_key_flags(ssl_flags)
FLAGS = flags.FLAGS

_JOB_TYPE = flags.DEFINE_enum(
    "job_type",
    default="train",
    enum_values=["train", "eval"],
    help="Specify whether to launch a training job or evaluation job.")

# Training Worker Specs
_TRAIN_IMAGE_URI = flags.DEFINE_string(
    "train_docker_image_uri_path",
    default="gcr.io/disaster-assessment/ssl-train-uri",
    help="Path to the Docker image for model training in GCR.")
_TRAIN_WORKER_MACHINE_TYPE = flags.DEFINE_string(
    "train_worker_machine_type",
    default="n1-highmem-2",
    help="Machine type for workers used in training.")
_TRAIN_WORKER_ACCELERATOR_TYPE = flags.DEFINE_string(
    "train_worker_accelerator_type",
    default="NVIDIA_TESLA_P100",
    help="Accelerator type for workers used in training.")
_TRAIN_WORKER_ACCELERATOR_COUNT = flags.DEFINE_integer(
    "train_worker_accelerator_count",
    default=2,
    help="Number of accelerators to use per worker in training.")

# Eval Worker Specs
_EVAL_IMAGE_URI = flags.DEFINE_string(
    "eval_docker_image_uri_path",
    default="gcr.io/disaster-assessment/ssl-eval-uri",
    help="Path to the Docker image for model evaluation in GCR.")
_EVAL_WORKER_MACHINE_TYPE = flags.DEFINE_string(
    "eval_worker_machine_type",
    default="n1-standard-4",
    help="Machine type for workers used in evaluation.")
_EVAL_WORKER_ACCELERATOR_TYPE = flags.DEFINE_string(
    "eval_worker_accelerator_type",
    default="NVIDIA_TESLA_P100",
    help="Accelerator type for workers used in evaluation.")
_EVAL_WORKER_ACCELERATOR_COUNT = flags.DEFINE_integer(
    "eval_worker_accelerator_count",
    default=1,
    help="Number of accelerators to use per worker in evaluation.")

# Job Specs
_DISPLAY_NAME = flags.DEFINE_string(
    "display_name",
    default="ssl-train",
    help="Display name of this job.")
_SERVICE_ACCOUNT = flags.DEFINE_string(
    "service_account",
    default=None,
    help="Email address of the service account. Example: "
    "user@project.iam.gserviceaccount.com")
_PROJECT = flags.DEFINE_string(
    "project",
    default=None,
    help="Google Cloud Project in which job is run. For guidance, user likely "
    "ran `gcloud init` earlier in the pipeline; use the project that was "
    "specified previously.")
_LOCATION = flags.DEFINE_string(
    "location",
    default=None,
    help="Location of physical computing resources used to run job. Must be "
    "the same as location of data. For list of all options, see: "
    "https://cloud.google.com/vertex-ai/docs/general/locations")


def _get_worker_pool_specs(flags_to_pass: List[str]) -> List[Dict[str, Any]]:
  """Creates the worker pool specs depending on the job type.

  Args:
    flags_to_pass: Flags that are passed to the job.

  Returns:
    Dictionary of worker pool specs wrapped by list, as expected by CustomJob.
  """
  if _JOB_TYPE.value == "train":
    return [{
        "machine_spec": {
            "machine_type": _TRAIN_WORKER_MACHINE_TYPE.value,
            "accelerator_type": _TRAIN_WORKER_ACCELERATOR_TYPE.value,
            "accelerator_count": _TRAIN_WORKER_ACCELERATOR_COUNT.value
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": _TRAIN_IMAGE_URI.value,
            "args": flags_to_pass,
        },
    }]
  else:
    return [{
        "machine_spec": {
            "machine_type": _EVAL_WORKER_MACHINE_TYPE.value,
            "accelerator_type": _EVAL_WORKER_ACCELERATOR_TYPE.value,
            "accelerator_count": _EVAL_WORKER_ACCELERATOR_COUNT.value
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": _EVAL_IMAGE_URI.value,
            "args": flags_to_pass
        }
    }]


def main(unused_argv) -> None:
  flags_list = FLAGS.key_flags_by_module_dict()[sys.argv[0]]
  flags_to_pass = utils.reformat_flags(flags_list)

  # The spec of the worker pools including machine type and Docker image
  worker_pool_specs = _get_worker_pool_specs(flags_to_pass)

  job = aiplatform.CustomJob(
      display_name=_DISPLAY_NAME.value,
      worker_pool_specs=worker_pool_specs,
      project=_PROJECT.value,
      location=_LOCATION.value,
      base_output_dir=FLAGS.train_dir,
      staging_bucket=FLAGS.train_dir)

  if not _SERVICE_ACCOUNT.value:
    raise ValueError("Must specify a service account.")

  job.run(service_account=_SERVICE_ACCOUNT.value)

if __name__ == "__main__":
  flags.mark_flags_as_required(["dataset_name", "train_dir", "test_examples"])
  app.run(main)
