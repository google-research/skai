# SKAI Damage Assessment Instructions

Last update: June 6, 2024

Before running these instructions, please make sure that your Google Cloud
project and Linux environment have been set up by following these
[instructions](setup.md).


## Step 1: Set environment variables

Before starting the assessment, please set a few environment variables to streamline running future commands.


```
$ export PROJECT=<your cloud project>
$ export LOCATION=<cloud location, e.g. us-central1>
$ export SERVICE_ACCOUNT=<service account email>
$ export PYTHONPATH=<skai source directory>:$PYTHONPATH
```



## Step 2: Prepare images

Your satellite images must be [Cloud Optimized GeoTIFFs](https://www.cogeo.org/)
for SKAI to process them. If you are not sure if your GeoTIFFs are valid Cloud
Optimized GeoTIFFs, you can check using this command (make sure the SKAI python
virtualenv is activated):

```
$ rio cogeo validate /path/to/images/before.tif

before.tif is a valid cloud optimized GeoTIFF
```


If your images are not COGs, you can use the following command to convert it:


```
$ rio cogeo create /path/to/images/before.tif
```


For more information, see [here](https://cogeotiff.github.io/rio-cogeo/Is_it_a_COG/).

Your satellite images should be uploaded to your Cloud storage bucket so that they can be accessed by the Dataflow preprocessing pipeline. If the images are on your workstation, you can use gsutil to upload them to the bucket.


```
$ gsutil cp /path/to/images/before.tif gs://$BUCKET/images/before.tif
$ gsutil cp /path/to/images/after.tif gs://$BUCKET/images/after.tif
```

You can provide SKAI with multiple before and after images. It will
automatically determine which before and after images pair together. However, if
a building is only covered by a before image but not an after image, or
vice-versa, SKAI will not assess that building.

## Step 3: Choose an Area of Interest (Optional)

You need to choose an area of interest (AOI) where SKAI will perform the assessment.
By default, SKAI will consider the entire area covered by your after images to
be the area of interest. If you want to restrict the assessment to a smaller
area, then you must specify the bounds of that area in the format described
below.

The custom AOI should be recorded in a GIS file format, such as [GeoJSON](https://geojson.org/) (preferred) or [Shapefile](https://en.wikipedia.org/wiki/Shapefile). The easiest way to do this is to use a GIS program such as [QGIS](https://www.qgis.org/) that lets you draw a polygon on a map, then save that polygon as a GeoJSON file. This [QGIS tutorial](https://docs.qgis.org/3.22/en/docs/training_manual/create_vector_data/create_new_vector.html) walks through how to do this.


## Step 4: Generate Unlabeled Examples

The next step is to extract examples of buildings in the AOI from the before and
after images, and save them in SKAI's training example format. Run the following
command to do that.

```
$ python generate_examples_main.py \
  --cloud_project=$PROJECT \
  --cloud_region=$LOCATION \
  --dataset_name=<dataset name> \
  --before_image_patterns=<before image paths> \
  --after_image_patterns=<after image paths> \
  --aoi_path=<aoi-path> \
  --output_dir=gs://$BUCKET/test_run \
  --buildings_method=<building method> \
  --use_dataflow \
  --worker_service_account=$SERVICE_ACCOUNT
```

`<dataset name>` is an arbitrary name that the dataflow job will take on.
It should only contain alphanumeric characters and hyphen ("-").

`<before image patterns>` is a comma-separated list of path patterns to your
pre-disaster images. For example,
`gs://$BUCKET/images/before1.tif,gs://$BUCKET/images/before2.tif`.

`<after image patterns>` is a comma-separated list of path patterns to your
post-disaster images.

`<aoi-path>` is the path to the Area of Interest file, as discussed in the
previous section. If you omit this flag, the entire area covered by your after
image(s) will be considered the area of interest.

`<building method>` specifies how building centroids will be fetched.
This is discussed in more detail in the Building Detection subsection below.

After running this command, you should be able to see a new Dataflow job in the
[Cloud console](https://console.cloud.google.com/dataflow/jobs). Clicking on the
job will show a real-time monitoring page of the job's progress.

When the Dataflow job finishes, it should have generated two sets of sharded
TFRecord files, one set under the "unlabeled" directory, and one set under the
"unlabeled-large" directory. Both file sets contain unlabeled examples of
all buildings found in the Area Of Interest. The difference is that the images
in the "unlabeled-large" file set include more context around the building and
are larger (by default, 256x256). These examples will be used for labeling. The
images in the "unlabeled" file set are smaller (by default, 64x64), and will be
used for model training and inference.

### Example Generation Configuration File

In addition to specifying input parameters using command line flags, the
`generate_examples_main.py` script can also read a configuration file. The
configuration file is more convenient when running SKAI on many images at a
time, and also makes pipelines more reproducible.

The configuration file uses
[JSON](https://www.w3schools.com/js/js_json_syntax.asp) format. The options
have the same names as the command line flags.

An example configuration files:

```
{
  "dataset_name": "<dataset name>",
  "aoi_path": "<aoi-path>",
  "output_dir": "gs://my-bucket/test_run",
  "buildings_method": "file",
  "buildings_file": "gs://my-bucket/buildings.csv",
  "use_dataflow": true,
  "cloud_project": "<cloud project>",
  "cloud_region": "<cloud region>",
  "worker_service_account": "<service account>",
  "max_dataflow_workers": 100,
  "output_shards": 100,
  "output_metadata_file": true,
  "before_image_patterns": [
    "<before image 1 pattern>",
    "<before image 2 pattern>",
    "<before image 3 pattern>"
  ],
  "after_image_patterns": [
    "<after image 1 pattern>",
    "<after image 2 pattern>",
    "<after image 3 pattern>"
  ]
}
```

Create a text file like the above and store it on your workstation. Then, to run
example generation, you only need to type:

```
$ python generate_examples_main.py --configuration_path=<path to config file>
```

### Building Detection

SKAI can ingest building centroids from the following open source databases:

* [Open Buildings](https://sites.research.google/open-buildings/)
* [OpenStreetMap](https://www.openstreetmap.org/)

The accuracy and coverage of these two databases are very different, so please
research which database is best for the region of the world most relevant to
your assessment.

Alternatively, you can provide a CSV file containing the longitude, latitude
coordinates of the buildings in the area of interest.

The example generation pipeline controls the building detection method using
two flags. The `--buildings_method` flag specifies where to look for buildings.
It can be set to `open_buildings`, `open_street_map`, or `file`, each
corresponding to one of the sources described above.

If `--buildings_method` is set to `file`, then the flag `--buildings_file` must
be set to the path of the CSV file that contains the building centroids.

## Step 5: Create example labeling task

To train a SKAI model, a small number of examples generated in the previous step
must be manually labeled. Run this command to collect a sample for labeling,
and generate PNG images suitable for use in the labeling task.

```
$ python create_labeling_examples.py \
  --examples_pattern=<examples pattern> \
  --output_dir=<output dir> \
  --max_images=2000
```

`<examples pattern>` is the file pattern matching the TFRecord containing
*large* unlabeled examples generated in the previous step. It should look
something like
`gs://$BUCKET/test_run/examples/unlabeled-large/unlabeled-*.tfrecord`.

`<output dir>` is the directory to write labeling images to. This can be
set to any Google Cloud Storage path. For example,
`gs://$BUCKET/test_run/examples/labeling_images`. After the command is finished,
you can see the images generated for labeling in this directory.

This command will also generate a CSV file called `image_metadata.csv` in the
output directory. This CSV contains information about each example in the
labeling sample, including the path of the labeling image and other metadata.
This CSV can be used by labeling tools to create labeling tasks.

## Step 6: Label examples

Set up the Eagle Eye labeling tool in your GCP environment by following these
[instructions](https://github.com/google-research/skai/blob/main/src/eagle_eye/README.md).
Then create a new project in Eagle Eye and upload the CSV generated in the
previous step to begin labeling the examples.

## Step 7: Merge Labels into Dataset

When a sufficient number of examples are labeled, download the labels from the
labeling tool as a CSV file and then merged into the TFRecords we are training
on.

```
$ python create_labeled_dataset.py \
  --examples_pattern=gs://$BUCKET/examples/test_run/examples/unlabeled-large/*.tfrecord \
  --label_file_paths=gs://$BUCKET/examples/test_run/labels.csv \
  --string_to_numeric_labels='bad_example=0,no_damage=0,minor_damage=1,major_damage=1,destroyed=1' \
  --train_output_path=gs://$BUCKET/examples/labeled_train_examples.tfrecord \
  --test_output_path=gs://$BUCKET/examples/labeled_test_examples.tfrecord
```

The `--label_file_paths` flag should point to the CSV files you downloaded from
the labeling tool.

The flag `--string_to_numeric_labels` controls how string label values such as
"no\_damage" and "destroyed" are mapped to numeric label values (either 0 or 1).
The default mapping is:

*   no\_damage, bad\_example --> 0
*   minor\_damage, major\_damage, destroyed --> 1

This command will output two TFRecord files, one containing examples for
training and one containing examples for testing. By default, 20% of labeled
examples are put into the test set, and the rest go into the training set. This
can be changed with the `--test_fraction` flag in the above command.


## Step 8: Train the Model

**Start the training job:**

Run the following command to start the training job.
Please edit the flag values as appropriate for your setup.

```
$ xmanager launch src/skai/model/xm_launch_single_model_vertex.py -- \
  --config=src/skai/model/configs/skai_two_tower_config.py \
  --config.data.adhoc_config_name=my_dataset \
  --config.data.labeled_train_pattern=gs://$BUCKET/examples/labeled_train_examples.tfrecord \
  --config.data.unlabeled_train_pattern=gs://$BUCKET/examples/labeled_train_examples.tfrecord \
  --config.data.validation_pattern=gs://$BUCKET/examples/labeled_test_examples.tfrecord \
  --config.output_dir=gs://$BUCKET/models/train_dir \
  --config.training.num_epochs=50 \
  --experiment_name=skai_model_training \
  --cloud_location=$LOCATION \
  --project_path"=$(pwd)"
```

If you wish to use GPU accelerators for training (recommended), add the
following two flags to the end of the command:

`--accelerator=T4 --accelerator_count=1`

Other possible GPU accelerators are: `P100`, `V100`, `P4`, `T4`, `A100`.

If you wish to use TPU accelerators for training, add the following two flags to
the end of the command.

`--accelerator=TPU_V2 --accelerator_count=8`

Once this command completes, point your web browser to the Vertex AI custom
training jobs
[console](https://console.cloud.google.com/vertex-ai/training/custom-jobs).
You should see your job listed here.

## Step 9: Generate damage assessment file

Run inference to get the model’s predictions on all buildings in the area of interest.

```
$ python skai/model/inference.py \
  --examples_pattern=gs://$BUCKET/examples/test_run/examples/unlabeled/*.tfrecord \
  --output_path=gs://$BUCKET/model_output.csv \
  --model_dir=gs://$BUCKET/models/train_dir/epoch-50-aucpr-0.812 \
  --use_dataflow \
  --cloud_project=$PROJECT \
  --cloud_region=$LOCATION \
  --dataflow_temp_dir='gs://$BUCKET/dataflow-temp' \
  --worker_service_account=$SERVICE_ACCOUNT
```

**Please note that you will need to change the model_dir flag to the appropriate path based on the contents of your output directory.**

This will launch a DataFlow job that runs model inference.
You can see the job in the [Cloud console](https://console.cloud.google.com/dataflow/jobs).

If you wish to use GPU accelerators to run inference, add the following flags
to the above command.

```
  --max_dataflow_workers=4 \
  --worker_machine_type=n1-highmem-8 \
  --accelerator=nvidia-tesla-t4 \
  --accelerator_count=1
```

TPUs are currently not available for inference as they are not supported by
Dataflow.

## Feedback

If you have any feedback on these instructions or SKAI in general, we'd love to hear from you. Please reach out to the developers at skai-developers@googlegroups.com, or create an issue in the Github issue tracker.
