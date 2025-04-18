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

"""Functions to generate docker build instructions."""

import pathlib

from xmanager import xm

GPU_ACCELERATORS = ['P100', 'V100', 'P4', 'T4', 'A100']
TPU_ACCELERATORS = ['TPU_V2', 'TPU_V3']

CPU_BASE_IMAGE = 'tensorflow/tensorflow:2.14.0'
GPU_BASE_IMAGE = 'tensorflow/tensorflow:2.14.0-gpu'
TPU_BASE_IMAGE = 'ubuntu:22.04'

SKAI_DOCKER_INSTRUCTIONS = [
    'COPY skai /skai',
    'RUN pip install -r /skai/requirements.txt',
    'RUN pip install /skai/src/.',
]

BIG_VISION_DOCKER_INSTRUCTIONS = [
    'RUN apt-get -y install git',
    'RUN git clone https://github.com/google-research/big_vision.git',
    'RUN pip install -r /big_vision/big_vision/requirements.txt',
    (
        'RUN echo "from setuptools import setup, find_packages" >>'
        ' /big_vision/setup.py'
    ),
    (
        'RUN echo \'setup(name="big_vision"'
        ' ,version="1.0",packages=find_packages())\' >> /big_vision/setup.py'
    ),
    'RUN touch /big_vision/big_vision/models/proj/__init__.py',
    'RUN touch big_vision/big_vision/models/proj/image_text/__init__.py',
    'RUN touch big_vision/big_vision/datasets/__init__.py',
    'RUN touch big_vision/big_vision/datasets/imagenet/__init__.py',
    'RUN pip uninstall big_vision',
    'RUN pip install /big_vision/.',
    (
        'RUN pip install jax[tpu] -f'
        ' https://storage.googleapis.com/jax-releases/libtpu_releases.html'
    ),
]


def tpuvm_docker_instructions() -> list[str]:
  """Returns a list of docker commands necessary to use TensorFlow on TPUs.

  Returns:
    Docker container build commands.
  """
  tf_wheel_name = (
      'tensorflow-2.14.0-cp310-cp310-manylinux_2_17_x86_64.' +
      'manylinux2014_x86_64.whl'
  )
  tf_wheel_url = (
      'https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/' +
      'tensorflow/tf-2.14.0/' + tf_wheel_name
  )
  tpu_shared_object_url = (
      'https://storage.googleapis.com/' +
      'cloud-tpu-tpuvm-artifacts/libtpu/1.8.0/libtpu.so'
  )
  return [
      f'RUN wget {tpu_shared_object_url} -O /lib/libtpu.so',
      'RUN chmod 700 /lib/libtpu.so',
      f'RUN wget {tf_wheel_url}',
      f'RUN pip3 install {tf_wheel_name}',
      f'RUN rm {tf_wheel_name}',
  ]


def get_docker_instructions(accelerator: str) -> tuple[str, list[str]]:
  """Returns the docker instructions and base image for `accelerator`.

  Args:
    accelerator: The type of accelerator to build the image for (cpu, gpu, tpu).

  Returns:
    A tuple of the base image and the docker instructions.
  """
  if (accelerator == 'tpu') or (accelerator in TPU_ACCELERATORS):
    # Required by TPU vm.
    base_image = TPU_BASE_IMAGE
    # Build can get stuck for a while without this line.
    docker_instructions = [
        'ENV DEBIAN_FRONTEND=noninteractive',
    ]
    # Make sure python executable is python3.
    docker_instructions += [
        'RUN apt-get update && apt-get install -y python3-pip wget'
    ]
    docker_instructions += tpuvm_docker_instructions()

  elif (accelerator == 'gpu') or (accelerator in GPU_ACCELERATORS):
    # Select a base GPU image. Other options can be found in
    # https://cloud.google.com/deep-learning-containers/docs/choosing-container
    base_image = GPU_BASE_IMAGE
    docker_instructions = [
        'RUN apt-get update && apt-get install -y ' +
        'libcairo2-dev libjpeg-dev libgif-dev'
    ]

  else:
    # Select a base CPU image. Other options can be found in
    # https://cloud.google.com/deep-learning-containers/docs/choosing-container
    base_image = CPU_BASE_IMAGE
    docker_instructions = [
        'RUN apt-get update && apt-get install -y python3-pip wget',
    ]
  docker_instructions += [
      'RUN apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev ' +
      'libglib2.0-0 python-is-python3'
  ]
  docker_instructions.extend(BIG_VISION_DOCKER_INSTRUCTIONS)
  docker_instructions.extend(SKAI_DOCKER_INSTRUCTIONS)
  return base_image, docker_instructions


def get_geofm_docker_instructions() -> tuple[str, list[str]]:
  """Returns the docker instructions and base image for `accelerator`.

  Returns:
    A tuple of the base image and the docker instructions.
  """
  base_image = 'tensorflow/tensorflow:2.19.0'
  docker_instructions = [
      'RUN apt-get update && apt-get install -y python3-pip wget',
      'RUN pip install ml-collections',
      'RUN pip install pandas>=2',
      'RUN pip install geopandas>=0.8',
      'RUN pip install pillow',
      'RUN pip install rasterio==1.3.9',
      'RUN pip install tqdm',
      'COPY skai /skai',
      'RUN pip install /skai/src/.',
    ]
  return base_image, docker_instructions


def get_xm_executable_spec(accelerator: str):
  """Returns a Xmanager executable spec that can be used to build docker images.

  The image has a default entrypoint that launches a Python script. The script
  must be specified as the first argument when running the image, followed by
  all flags that the script expects. Making the script an argument allows the
  built image to be used to launch any script in the SKAI repo.

  The script path should start with "/skai/src". For example, the following
  script path is used to run model training: "/skai/src/skai/model/train.py"

  Args:
    accelerator: The type of accelerator to build the image for (cpu, gpu, tpu,
      geofm-cpu).

  Returns:
    Xmanager executable spec.
  """
  source_path = str(pathlib.Path(__file__).parents[3])  # SKAI root directory.
  if accelerator == 'geofm-cpu':
    base_image, instructions = get_geofm_docker_instructions()
  else:
    base_image, instructions = get_docker_instructions(accelerator)
  return xm.PythonContainer(
      path=source_path,
      base_image=base_image,
      docker_instructions=instructions,
      entrypoint=xm.CommandList([
          'cd /skai',
          'eval python $@',  # "eval" is needed to strip a level of quotes off.
      ]),
      use_deep_module=True,
  )
