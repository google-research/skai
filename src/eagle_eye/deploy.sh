#!/bin/bash

# Copyright 2024 Google LLC
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

# Script for deploying Eagle Eye via Google Cloud Run.
#
# Usage example:
#
# $ bash deploy.sh /path/to/my/config.json

if [ ! -f "$1" ]
then
  echo "Usage: $0 config.json <database>"
  exit 1
fi

if [ -n "$2" ]
then
  ENV_VAR_SETTINGS="--set-env-vars FIRESTORE_DB=$2"
fi

TEMP_DIR=$(mktemp -d)
SOURCE_DIR=`dirname $0`

cp -r "$SOURCE_DIR"/* "$TEMP_DIR"
cp "$1" "$TEMP_DIR/app/config.json"
gcloud run deploy eagleeye --source "$TEMP_DIR/app" $ENV_VAR_SETTINGS
rm -rf "$TEMP_DIR"
