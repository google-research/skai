r"""XManager launch script for zero_shot eval on Vertex AI.

This script launches an XManager experiment for zero-shot inference. It launches
the zero-shot inference job on a single TPU.

Usage:

xmanager launch src/skai/model/xm_vlm_zero_shot_vertex.py -- \
    --model_variant='So400m/14' \
    --batch_size=128 \
    --image_size=224 \
    --dataset_names='hurricane_ian' \
    --example_patterns='/path/to/hurricane_ian_dataset' \
    --output_dir='/path/to/output_dir' \
    --source_dir='/tmp/skai'

"""

from absl import flags
from xmanager import xm
from xmanager import xm_local


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory.', required=True
)

_EXAMPLE_PATTERNS = flags.DEFINE_list(
    'example_patterns',
    None,
    'List of file patterns to the input datasets.',
    required=True,
)

_DATASET_NAMES = flags.DEFINE_list(
    'dataset_names', None, 'List of dataset names.'
)

_POSITIVE_LABELS_FILEPATH = flags.DEFINE_string(
    'positive_labels_filepath',
    'gs://skai-public/VLM/damaged_labels.txt',
    'File path to a text file containing positive labels.',
)

_NEGATIVE_LABELS_FILEPATH = flags.DEFINE_string(
    'negative_labels_filepath',
    'gs://skai-public/VLM/undamaged_labels.txt',
    'File path to a text file containing negative labels.',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 128, 'Batch size for the inference.'
)

_IMAGE_FEATURE = flags.DEFINE_string(
    'image_feature',
    'post_image_png_large',
    'Feature to use as the input image.',
)

_MODEL_VARIANT = flags.DEFINE_string(
    'model_variant',
    'So400m/14',
    'Specifys model variant. Available model variants'
    ' are "B/16", "L/16", "So400m/14" and "B/16-i18n". Note model_variant'
    ' supports a specific set of image sizes.',
)

_IMAGE_SIZE = flags.DEFINE_integer('image_size', 224, 'Image size.')

_SOURCE_DIR = flags.DEFINE_string(
    'source_dir', None, 'Path to the dirctory containg the skai source code.'
)


TPU_BASE_IMAGE = 'ubuntu:22.04'

SKAI_DOCKER_INSTRUCTIONS = [
    'RUN apt-get -y install git',
    'COPY skai/ /skai',
    'RUN pip install -r /skai/requirements.txt --timeout 1000',
    'RUN pip install /skai/src/.',
]

BIG_VISION_DOCKER_INSTRUCTIONS = [
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


# copied from skai.model.docker_instructions.
# TODO(mohammedelfatihsalah): Refactor below function to a different location.
def tpuvm_docker_instructions():
  """Returns a list of docker commands necessary to use TensorFlow on TPUs.

  Returns:
    Docker container build commands.
  """
  docker_instructions = [
      'ENV DEBIAN_FRONTEND=noninteractive',
  ]
  # Make sure python executable is python3.
  docker_instructions += [
      'RUN apt-get update && apt-get install -y python3-pip wget'
  ]
  tf_wheel_name = (
      'tensorflow-2.14.0-cp310-cp310-manylinux_2_17_x86_64.'
      + 'manylinux2014_x86_64.whl'
  )
  tf_wheel_url = (
      'https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/'
      + 'tensorflow/tf-2.14.0/'
      + tf_wheel_name
  )
  tpu_shared_object_url = (
      'https://storage.googleapis.com/'
      + 'cloud-tpu-tpuvm-artifacts/libtpu/1.8.0/libtpu.so'
  )
  docker_instructions.extend([
      f'RUN wget {tpu_shared_object_url} -O /lib/libtpu.so',
      'RUN chmod 700 /lib/libtpu.so',
      f'RUN wget {tf_wheel_url}',
      f'RUN pip3 install {tf_wheel_name}',
      f'RUN rm {tf_wheel_name}',
  ])
  docker_instructions.extend([
      'RUN apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev ' +
      'libglib2.0-0 python-is-python3'
  ])
  return docker_instructions


def main(_) -> None:
  experiment_name = []
  experiment_name.append(_MODEL_VARIANT.value)
  experiment_name.append(str(_IMAGE_SIZE.value))
  if _DATASET_NAMES.value:
    experiment_name.extend(_DATASET_NAMES)

  experiment_name = '_'.join(experiment_name)

  with xm_local.create_experiment(
      experiment_title=experiment_name
  ) as experiment:
    instructions = tpuvm_docker_instructions()
    instructions.extend(SKAI_DOCKER_INSTRUCTIONS)
    instructions.extend(BIG_VISION_DOCKER_INSTRUCTIONS)
    executable_spec = xm.PythonContainer(
        path=_SOURCE_DIR.value,
        base_image=TPU_BASE_IMAGE,
        docker_instructions=instructions,
        entrypoint=xm.CommandList([
            'python /skai/src/skai/model/vlm_zero_shot_vertex.py $@',
        ]),
        use_deep_module=True,
    )
    resources_args = {
        'TPU_V3': 8,
        'RAM': 64 * xm.GiB,
        'CPU': 8 * xm.vCPU,
    }
    executor = xm_local.Vertex(
        requirements=xm.JobRequirements(
            service_tier=xm.ServiceTier.PROD, **resources_args
        ),
    )
    [train_executable] = experiment.package([
        xm.Packageable(
            executable_spec=executable_spec,
            executor_spec=xm_local.Vertex.Spec()
        ),
    ])

    experiment.add(
        xm.Job(
            executable=train_executable,
            args={
                'model_variant': _MODEL_VARIANT.value,
                'image_size': _IMAGE_SIZE.value,
                'dataset_names': _DATASET_NAMES.value,
                'example_patterns': _EXAMPLE_PATTERNS.value,
                'output_dir': _OUTPUT_DIR.value,
                'negative_labels_filepath': _NEGATIVE_LABELS_FILEPATH.value,
                'positive_labels_filepath': _POSITIVE_LABELS_FILEPATH.value,
                'batch_size': _BATCH_SIZE.value,
                'image_feature': _IMAGE_FEATURE.value,
            },
            executor=executor,
        )
    )
