#!/bin/bash
#
# This script will run all Python unit tests in SKAI. You should be able to run
# this script on Linux as long as you have a recent Python version (>=3.7) and
# virtualenv installed.

if [ -n "$KOKORO_ROOT" ]
then
  # Setup for invoking from continuous integration test framework.
  VIRTUALENV_PATH="${KOKORO_ROOT}/skai_env"
  SKAI_DIR="${KOKORO_ARTIFACTS_DIR}/github/skai"
  pyenv install --skip-existing 3.7.10  # Closest to version used in Colab (3.7.12).
  pyenv global 3.7.10
  which python
  python --version
else
  # Setup for manually triggered runs.
  VIRTUALENV_PATH=/tmp/skai_env
  SKAI_DIR=`dirname $0`/..
fi

set -e
set -x

function setup {
  virtualenv "${VIRTUALENV_PATH}"
  source "${VIRTUALENV_PATH}/bin/activate"
  pushd "${SKAI_DIR}"
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
