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


CPU_BASE_IMAGE = 'tensorflow/tensorflow:2.14.0'
GPU_BASE_IMAGE = 'tensorflow/tensorflow:2.19.0-gpu'
TPU_BASE_IMAGE = 'ubuntu:22.04'

SKAI_DOCKER_INSTRUCTIONS = [
    'COPY skai /skai',
    'RUN pip install -r /skai/requirements.txt',
    'RUN pip install /skai/src/.',
]

BIG_VISION_DOCKER_INSTRUCTIONS = [
    (
        # The commit hash 6d6c28a9634fd2f48f0f505f112d063dfc9bdf96 is tested to
        # work with the current version of SKAI.
        'RUN git clone https://github.com/google-research/big_vision.git'
        ' /big_vision && cd /big_vision && git checkout'
        ' 0127fb6b337ee2a27bf4e54dea79cff176527356'
    ),
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
    'RUN touch big_vision/big_vision/pp/proj/paligemma/__init__.py',
    'RUN touch big_vision/big_vision/pp/proj/__init__.py',
    'RUN pip uninstall big_vision',
    'RUN pip install /big_vision/.',
    'RUN cd ..',
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


def get_docker_instructions(image_type: str) -> tuple[str, list[str]]:
  """Returns the docker instructions and base image for `accelerator`.

  Args:
    image_type: The type of image to build.

  Returns:
    A tuple of the base image and the docker instructions.
  """
  if image_type == 'siglip-tpu':
    return get_siglip_tpu_docker_instructions()
  elif image_type == 'siglip-gpu':
    return get_siglip_gpu_docker_instructions()
  elif image_type == 'geofm-cpu':
    return get_geofm_docker_instructions('geofm-cpu')
  elif image_type == 'geofm-gpu':
    return get_geofm_docker_instructions('geofm-gpu')
  elif image_type == 'gpu':
    # Select a base GPU image. Other options can be found in
    # https://cloud.google.com/deep-learning-containers/docs/choosing-container
    base_image = GPU_BASE_IMAGE
    docker_instructions = [
        'RUN apt-get update && apt-get install -y ' +
        'libcairo2-dev libjpeg-dev libgif-dev'
    ]
  elif image_type == 'cpu':
    # Select a base CPU image. Other options can be found in
    # https://cloud.google.com/deep-learning-containers/docs/choosing-container
    base_image = CPU_BASE_IMAGE
    docker_instructions = [
        'RUN apt-get update && apt-get install -y python3-pip wget',
    ]
  else:
    raise ValueError(f'Unsupported image type: {image_type}')
  docker_instructions += [
      'RUN apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev ' +
      'libglib2.0-0 python-is-python3'
  ]
  docker_instructions.extend(SKAI_DOCKER_INSTRUCTIONS)
  return base_image, docker_instructions


def get_siglip_gpu_docker_instructions() -> tuple[str, list[str]]:
  """Returns the docker instructions and base image for to run SigLIP on TPU."""
  docker_instructions = [
      'ENV DEBIAN_FRONTEND=noninteractive',
      (
          'RUN apt-get update && apt-get install -y wget git libgl1-mesa-glx'
          ' libsm6 libxext6 libxrender-dev libglib2.0-0 python-is-python3'
          ' libcairo2-dev libjpeg-dev libgif-dev'
      ),
  ]
  # For skai packages.
  docker_instructions.extend([
      'COPY skai /skai',
      'RUN pip install /skai/src/.',
      'RUN pip install -r /skai/siglip_requirements.txt',
  ])
  docker_instructions.extend(BIG_VISION_DOCKER_INSTRUCTIONS)
  docker_instructions.append('RUN pip install jax[cuda12]')
  return 'tensorflow/tensorflow:2.18.0-gpu', docker_instructions


def get_siglip_tpu_docker_instructions() -> tuple[str, list[str]]:
  """Returns the docker instructions and base image for to run SigLIP on TPU."""
  docker_instructions = [
      'ENV DEBIAN_FRONTEND=noninteractive',
      'RUN apt-get update && apt-get install -y python3-pip wget',
  ]
  docker_instructions.extend(tpuvm_docker_instructions())
  docker_instructions.append(
      'RUN apt-get -y install git',
  )
  docker_instructions += [
      'RUN apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev '
      + 'libglib2.0-0 python-is-python3'
  ]
  # For skai packages.
  docker_instructions.extend([
      'COPY skai /skai',
      'RUN pip install /skai/src/.',
      'RUN pip install -r /skai/siglip_requirements.txt',
  ])
  docker_instructions.extend([
      (
          'RUN pip install jax[tpu] -f'
          ' https://storage.googleapis.com/jax-releases/libtpu_releases.html'
      ),
  ])
  docker_instructions.extend(BIG_VISION_DOCKER_INSTRUCTIONS)

  return TPU_BASE_IMAGE, docker_instructions


def get_geofm_docker_instructions(accelerator: str) -> tuple[str, list[str]]:
  """Returns the docker instructions and base image for `accelerator`.

  Args:
    accelerator: The type of accelerator to build the image for (geofm-cpu,
      geofm-gpu).

  Returns:
    A tuple of the base image and the docker instructions.
  """
  assert accelerator in ['geofm-cpu', 'geofm-gpu'], (
      'Unsupported accelerator: %s' % accelerator
  )
  base_image = 'tensorflow/tensorflow:2.19.0'
  docker_instructions = [
      'RUN apt-get update && apt-get install -y python3-pip wget',
      'RUN pip install ml-collections',
      'RUN pip install pandas>=2',
      'RUN pip install geopandas>=0.8',
      'RUN pip install pillow',
      'RUN pip install rasterio==1.3.9',
      'RUN pip install tqdm',
  ]
  if accelerator == 'geofm-gpu':
    docker_instructions.extend([
        'RUN pip install "tensorflow[and-cuda]"',
    ])
    base_image = 'tensorflow/tensorflow:2.19.0-gpu'
  docker_instructions.extend([
      'COPY skai /skai',
      'RUN pip install /skai/src/.',
  ])
  return base_image, docker_instructions


def get_xm_executable_spec(docker_image_name: str) -> xm.PythonContainer:
  """Returns a Xmanager executable spec that can be used to build docker images.

  The image has a default entrypoint that launches a Python script. The script
  must be specified as the first argument when running the image, followed by
  all flags that the script expects. Making the script an argument allows the
  built image to be used to launch any script in the SKAI repo.

  The script path should start with "/skai/src". For example, the following
  script path is used to run model training: "/skai/src/skai/model/train.py"

  Args:
    docker_image_name: The name of the image that is going to be built, which
      includes the type of accelerator, e.g. siglip-tpu, geofm-gpu.

  Returns:
    A PythonContainer to be used as the XManager executable spec.
  """
  source_path = str(pathlib.Path(__file__).parents[3])  # SKAI root directory.
  base_image, instructions = get_docker_instructions(docker_image_name)
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
