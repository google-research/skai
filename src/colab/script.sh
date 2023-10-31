# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


export GOOGLE_APPLICATION_CREDENTIALS=/root/service-account-private-key.json
export GOOGLE_CLOUD_BUCKET_NAME=skai-data

cd /content/skai
xmanager launch src/skai/model/xm_launch_single_model_vertex.py -- --xm_wrap_late_bindings --xm_upgrade_db=True --project_path=/content/skai --accelerator_count=1 --config=src/skai/model/configs/skai_two_tower_config.py --config.data.tfds_dataset_name=skai_dataset --config.data.tfds_data_dir=gs://skai-data/hurricane_ian --config.output_dir=gs://skai-data/experiments/skai_train_vizier --config.training.num_epochs=10 --accelerator=V100 --experiment_name=skai_train_vizier
