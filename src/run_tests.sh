#!/bin/bash
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

#
# This script will run all Python unit tests in SKAI. You should be able to run
# this script on Linux as long as you have a recent Python version (>=3.7) and
# virtualenv installed.

set -e
set -x

if [ -n "$KOKORO_ROOT" ]
then
  # Setup for invoking from continuous integration test framework.
  VIRTUALENV_PATH="${KOKORO_ROOT}/skai_env"
  SKAI_DIR="${KOKORO_ARTIFACTS_DIR}/github/skai"

  PY_VERSION=3.11.4
  pyenv uninstall -f $PY_VERSION
  # Have to install lzma library first, otherwise will get an error
  # "ModuleNotFoundError: No module named '_lzma'".
  sudo apt-get install liblzma-dev
  pyenv install -f $PY_VERSION
  pyenv global $PY_VERSION
else
  # Setup for manually triggered runs.
  VIRTUALENV_PATH=/tmp/skai_env
  SKAI_DIR=`dirname $0`/..
fi

function setup {
  if ! which python && which python3
  then
    PYTHON=python3
  else
    PYTHON=python
  fi

  which $PYTHON
  $PYTHON --version
  $PYTHON -m venv "${VIRTUALENV_PATH}"
  source "${VIRTUALENV_PATH}/bin/activate"
  pushd "${SKAI_DIR}"
  which pip
  pip --version
  pip install -r requirements.txt
}

function teardown {
  popd
  deactivate
  rm -rf "${VIRTUALENV_PATH}"
}

function run_tests {
  pushd src
  export PYTHONPATH=.:${PYTHONPATH}
  for test in `find skai -name '*_test.py'`
  do
    python "${test}" || exit 1
  done
}

setup
run_tests
teardown
