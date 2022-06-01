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

r"""Trains models using semi-supervised learning methods.

Creates a dataloder and semi-supervised model. Launches training and/or
evaluation depending on flags.
"""

from absl import app
from absl import flags
import numpy as np
from skai import ssl_flags
from skai.semi_supervised import ssl_train_library
from skai.semi_supervised import utils

flags.adopt_module_key_flags(ssl_flags)
FLAGS = flags.FLAGS


def main(unused_argv):
  np.random.seed(FLAGS.shuffle_seed)
  dataset = ssl_train_library.create_dataset(
      shuffle=FLAGS.shuffle_seed is not None)
  ssl_train_library.set_experiment_hyperparams()
  model = ssl_train_library.create_model(dataset)
  ssl_train_library.launch_train_or_eval(model)


if __name__ == '__main__':
  flags.mark_flags_as_required(['dataset_name', 'train_dir', 'test_examples'])
  utils.setup_tf()
  app.run(main)
