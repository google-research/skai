#!/bin/sh
#
# Builds new docker images for SKAI's dataflow pipeline.

set -e

PYTHON_VERSIONS=('3.11' '3.10')
BEAM_VERSION='2.54.0'
TENSORFLOW_VERSION='2.14.0'

PYTHON_DEPS=(
  earthengine-api
  fiona
  gcsfs
  "geopandas>=0.8"
  ml-collections
  numpy
  opencv-python
  openlocationcode
  "pandas>=2"
  pillow
  pyarrow
  pyproj
  rasterio
  rtree
  "shapely>=2.0.0"
  "tensorflow==${TENSORFLOW_VERSION}"
  tensorflow-addons
  tensorflow-datasets
  tf-models-official
  tqdm
)

function gen_cpu_docker_file() {
  PYTHON_VERSION="$1"
  cat <<EOF
FROM apache/beam_python${PYTHON_VERSION}_sdk:${BEAM_VERSION}
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --upgrade pip
RUN pip install ${PYTHON_DEPS[@]}
EOF
}


function gen_gpu_docker_file() {
  PYTHON_VERSION="$1"
  cat <<EOF
FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}-gpu
RUN apt-get update && apt-get install -y libcairo2-dev libjpeg-dev libgif-dev
RUN pip install --upgrade pip
RUN pip install ${PYTHON_DEPS[@]}
RUN pip install apache-beam[gcp]==${BEAM_VERSION}
COPY --from=apache/beam_python${PYTHON_VERSION}_sdk:${BEAM_VERSION} /opt/apache/beam /opt/apache/beam
ENTRYPOINT [ "/opt/apache/beam/boot" ]
EOF
}

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}
do
  for ACCELERATOR in cpu gpu
  do
    DOCKER_DIR="$(mktemp -d)"

    echo "Building docker image for Python ${version} in ${DOCKER_DIR}"
    cd $DOCKER_DIR

    case "$ACCELERATOR" in
      cpu)
        gen_cpu_docker_file $PYTHON_VERSION >>Dockerfile
        ;;
      gpu)
        gen_gpu_docker_file $PYTHON_VERSION >>Dockerfile
        ;;
    esac
    gcloud builds submit --tag "gcr.io/disaster-assessment/dataflow_${ACCELERATOR}_${PYTHON_VERSION}_image:${TIMESTAMP}"

    echo "Removing ${DOCKER_DIR}"
    rm -rf ${DOCKER_DIR}
  done
done
