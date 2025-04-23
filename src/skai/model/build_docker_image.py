r"""Builds docker images for ML jobs.

The image will be uploaded to "gcr.io/<project name>/skai-ml-<accelerator>".
You must pass the command-line flag "--xm_build_image_locally=False" when
running this script, so that the image will be built on Cloud and automatically
uploaded to gcr.io.

Example command:

python skai/model/build_docker_image.py \
  --project=disaster-assessment \
  --image_type=tpu \
  --xm_build_image_locally=False

"""

from absl import app
from absl import flags
from skai.model import docker_instructions
from xmanager import xm
import xmanager.cloud.build_image


_PROJECT = flags.DEFINE_string(
    'project', None, 'Cloud project to build image for.', required=True
)

_IMAGE_TYPE = flags.DEFINE_enum(
    'image_type',
    None,
    ['cpu', 'gpu', 'tpu', 'geofm-cpu', 'geofm-gpu'],
    'Type of image.',
    required=True,
)


def main(_) -> None:
  if flags.FLAGS.xm_build_image_locally:
    raise ValueError(
        'Local builds are not supported. You must pass'
        ' --xm_build_image_locally=False'
    )

  xmanager.cloud.build_image.build(
      docker_instructions.get_xm_executable_spec(_IMAGE_TYPE.value),
      args=xm.SequentialArgs(),
      env_vars={},
      image_name=f'gcr.io/disaster-assessment/skai-ml-{_IMAGE_TYPE.value}',
      project='disaster-assessment',
      bucket='disaster-assessment',
  )


if __name__ == '__main__':
  app.run(main)
