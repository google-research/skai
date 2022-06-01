#!/bin/sh

# Simple script which handles setting authentication variables for GDAL.

echo "Authorization: Bearer $(gcloud auth application-default print-access-token)" >/tmp/cloud_header
export GDAL_HTTP_HEADER_FILE=/tmp/cloud_header
