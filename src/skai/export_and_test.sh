#!/bin/sh

# Script to run copybara, setup the python virtualenv, and run tests.
# Primarily to workaround the fact that rasterio isn't available in third_party.

# Takes zero or one test files (relative to the src/skai dir) to the tests you'd like
# to run. Defaults to running all *_test.py files.

# Example Usage:
#   source export_and_test.sh [detect_buildings_test.py]
# source is needed because otherwise bash forks a subshell that doesn't
# affect the parent env and we can't source our python venv.

start_time=$(date +%s)

# Store current location, to return to.
current_dir=$PWD

# Go to root, copybara expects to be run from root.
cd $(p4 g4d)

# This will overwrite any prior instances of /tmp/skai.
/google/data/ro/teams/copybara/copybara third_party/py/skai/copy.bara.sky local_install .. --folder-dir=/tmp/skai

# For some reason, venv doesn't work at google3 root.
cd $current_dir

# Create new virtualenv, we use a directory that won't get wiped by copybara so
# we don't have to reinstall each time.
python3 -m venv /tmp/skaienv
source /tmp/skaienv/bin/activate

# Install dependencies to virtualenv.
python -m pip install -r /tmp/skai/requirements.txt

cd /tmp/skai/src/

# Measure how long copybara + setup takes.
end_time=$(date +%s)
echo "Setup took: $((end_time - start_time)) seconds"
tests=$@

# Add skai. prefix
test_modules=( "${tests[@]/#/skai.}" )

# Either test a specific file or test all files.

# TODO: Right now, these commands will fail for
# detect_buildings_test.py, since Tensorflow attempts to parse the -s and -p
# with abslflags. For now, detect_buildings_test must be run as
# python -m skai.detect_buildings_test.

if [ $# -eq 0 ]; then
  python -m unittest discover -s skai -p "*_test.py"
elif [ $# -eq 1 ]; then
  # python -m unittest discover -s skai -p $1
  # For single tests, run in a way that works with detect_buildings_test
  python -m "skai.${1%.py}"
else
  echo "Tool only accepts a single test file as an argument, if left empty the" \
  "script will run all tests."
fi

# Pop back into the working directory used to run test.sh
cd $current_dir
