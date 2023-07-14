# SKAI Damage Assessment Instructions

Last update: July 14, 2023

Before running these instructions, please make sure that your Google Cloud
project and Linux environment have been set up by following these
[instructions](setup.md).


## Step 1: Set environment variables

Before starting the assessment, please set a few environment variables to streamline running future commands.


```
$ export PROJECT=<your cloud project>
$ export LOCATION=<cloud location, e.g. us-central1>
$ export SERVICE_ACCOUNT=<service account email>
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
after images, and save them in SKAI's training example format. Run the following command to do that. Simultaneously, the command will also create a Cloud example labeling task


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
must be manually labeled. We use the
[Vertex AI labeling tool](https://cloud.google.com/vertex-ai/docs/datasets/data-labeling-job)
to do this. Run this command to create a labeling task in Vertex AI, and assign
the task to a number of human labelers.


```
$ python create_cloud_labeling_task.py \
  --cloud_project=$PROJECT \
  --cloud_location=$LOCATION \
  --dataset_name=<dataset name> \
  --examples_pattern=<examples pattern> \
  --images_dir=<images dir> \
  --cloud_labeler_emails=<labeler emails>
```

`<dataset name>` is a name you assign to the dataset to identify it.

`<examples pattern>` is the file pattern matching the TFRecord containing
*large* unlabeled examples generated in the previous step. It should look
something like
`gs://$BUCKET/test_run/examples/unlabeled-large/unlabeled-*.tfrecord`.

`<images dir>` is a temporary directory to write labeling images to. This can be
set to any Google Cloud Storage path. For example,
`gs://$BUCKET/test_run/examples/labeling_images`. After the command is finished,
you can see the images generated for labeling in this directory.

`<labeler-emails>` is a comma-delimited list of the emails of people who will be
labeling example images. They must be Google email accounts, such as GMail or
GSuite email accounts.

An example labeling task will also be created in Vertex AI, and instructions for
how to label examples will be sent to all email accounts provided in the `--cloud_labeler_emails` flag.

**Note:** It takes a while for VertexAI to send the out email containing the
link to the labeling interface. Please be patient.

## Step 6: Label examples

All labelers should follow the [labeling instructions](https://storage.googleapis.com/skai-public/labeling_instructions_v2.pdf) in their emails to manually label a number of building examples. Labeling at least 250 examples each of damaged/destroyed and undamaged buildings should be sufficient. Labeling more examples may improve model accuracy.

**Note:** The labeling task is currently configured with 5 choices for each example:

*   no\_damage
*   minor\_damage
*   major\_damage
*   destroyed
*   bad\_example

These text labels are mapped into a binary label, 0 or 1, when generating examples.
The mapping is as follows:

*   no\_damage, bad\_example --> 0
*   minor\_damage, major\_damage, destroyed --> 1

Future versions of the SKAI model will have a separate class for bad examples,
resulting in 3 classes total.

## Step 7: Merge Labels into Dataset

When a sufficient number of examples are labeled, the labels need to be downloaded and merged into the TFRecords we are training on.

Find the `cloud_dataset_id `of your newly labeled dataset by visiting the
[Vertex AI datasets console](https://console.cloud.google.com/vertex-ai/datasets), and looking at
the "ID" column of your recently created dataset.

![Dataset ID screenshot](https://storage.googleapis.com/skai-public/documentation_images/dataset_id.png "Find dataset ID")


```
$ python create_labeled_dataset.py \
  --cloud_project=$PROJECT \
  --cloud_location=$LOCATION \
  --cloud_dataset_ids=<id>
  --cloud_temp_dir=gs://$BUCKET/temp \
  --examples_pattern=gs://$BUCKET/examples/test_run/examples/unlabeled/*.tfrecord \
  --train_output_path=gs://$BUCKET/examples/labeled_train_examples.tfrecord \
  --test_output_path=gs://$BUCKET/examples/labeled_test_examples.tfrecord
```


This will generate two TFRecord files, one containing examples for training and one containing examples for testing. By default, 20% of labeled examples are put into the test set, and the rest go into the training set. This can be changed with the `--test_fraction` flag in the above command.


## Step 8: Train the Model

**Create a Tensorboard resource instance**:


```
$ gcloud ai tensorboards create –display-name <Tensorboard name>

Using endpoint [https://us-central1-aiplatform.googleapis.com/]
Waiting for operation [999391182737489573628]...done.
Created Vertex AI Tensorboard: projects/123456789012/locations/us-central1/tensorboards/874419473951.
```


The last line of the output is the tensorboard resource name. Pass this value into the flag `--tensorboard_resource_name `flag in the commands below.

**Start the training job:**

Give this experiment a name by passing it through the `--dataset_name `flag and replacing `train_dir_name` in the `--train_dir `flag.


```
$ python launch_vertex_job.py \
--project=$PROJECT \
--location=$LOCATION \
--job_type=train \
--display_name=train_job \
--train_docker_image_uri_path=gcr.io/disaster-assessment/ssl-train-uri \
--tensorboard_resource_name=<tensorboard resource name> \
--service_account=$SERVICE_ACCOUNT \
--dataset_name=dataset_name \
--train_dir=gs://$BUCKET/models/train_dir_name \
--train_label_examples=gs://$BUCKET/examples/labeled_train_examples.tfrecord \
--train_unlabel_examples=gs://$BUCKET/examples/test_run/examples/unlabeled/*.tfrecord \
--test_examples=gs://$BUCKET/examples/labeled_test_examples.tfrecord
```


**Start the eval job:**

This job will continuously evaluate the model on the test dataset and visualize the metrics in the tensorboard.


```
$ python launch_vertex_job.py \
--project=$PROJECT \
--location=$LOCATION \
--job_type=eval \
--display_name=eval_job \
--eval_docker_image_uri_path=gcr.io/disaster-assessment/ssl-eval-uri \
--service_account=$SERVICE_ACCOUNT \
--dataset_name=dataset_name \
--train_dir=gs://$BUCKET/models/train_dir_name \
--train_label_examples=gs://$BUCKET/examples/labeled_train_examples.tfrecord \
--train_unlabel_examples=gs://$BUCKET/examples/test_run/examples/unlabeled/*.tfrecord \
--test_examples=gs://$BUCKET/examples/labeled_test_examples.tfrecord
```

Once you see the evaluation job running, you can monitor the training progress on Tensorboard.

Point your web browser to the [Vertex AI custom training jobs console](https://console.cloud.google.com/vertex-ai/training/custom-jobs). You should see your job listed here. Click on the job, then click the “Open Tensorboard” button at the top.

![Open Tensorboard Screenshot](https://storage.googleapis.com/skai-public/documentation_images/open_tensorboard.png "Open Tensorboard")

**Note:** Tensorboards cost money to maintain. See [this page](https://cloud.google.com/vertex-ai/pricing) under the section "Vertex AI TensorBoard" for the actual cost. To save on Cloud billing, you should remove old Tensorboard instances once the model is trained. Tensorboard instances can be found and deleted on this [Cloud console page](https://console.cloud.google.com/vertex-ai/experiments/tensorboard-instances).


## Step 9: Generate damage assessment file

Run inference to get the model’s predictions on all buildings in the area of interest.


```
$ python3 launch_vertex_job.py \
  --project=$PROJECT \
  --location=$LOCATION \
  --job_type=eval \
  --display_name=inference \
  --eval_docker_image_uri_path=gcr.io/disaster-assessment/ssl-eval-uri \
  --service_account=$SERVICE_ACCOUNT \
  --dataset_name=dataset_name \
  --train_dir=gs://$BUCKET/models/train_dir_name \
  --test_examples=gs://$BUCKET/examples/labeled_test_examples.tfrecord \
  --inference_mode=True \
  --save_predictions=True
```


**Note:** If you would like to run inference using a specific checkpoint, use the `--eval_ckpt `flag. Example: `--eval_ckpt=gs://$BUCKET/models/train_dir_name/checkpoints/model.ckpt-00851968`. Do NOT include the extension, e.g. ‘.meta’, ‘.data’, or ‘.index’, and only use the prefix.

The predictions will be saved in a directory called `gs://$BUCKET/models/train_dir_name/predictions `as GeoJSON files. The number in each filename refers to the epoch of the checkpoint.


## Feedback

If you have any feedback on these instructions or SKAI in general, we'd love to hear from you. Please reach out to the developers at skai-developers@googlegroups.com, or create an issue in the Github issue tracker.


## Appendix: Build Docker containers for training and eval jobs

In the command for step 8, we used SKAI's default Docker containers for the training and eval jobs. If you make any changes to the training code, such as the model architecture, you must build and push your own Docker containers to the Container Registry, and then launch your training and eval jobs with those containers.

After you have modified the SKAI model source code, use this command to build a local custom container and launch a local training job with it to ensure that it works:


```
$ cd skai/src  # Make sure you're in the SKAI src directory.
$ gcloud beta ai custom-jobs local-run \
--base-image=gcr.io/deeplearning-platform-release/tf2-gpu.2-6 \
--python-module=ssl_train \
--requirements=tensorflow-probability==0.12.2 \
--work-dir=. \
--output-image-uri=gcr.io/$PROJECT/ssl-train-uri \
-- --dataset_name=dataset_name \
--train_dir=gs://$BUCKET/models/train_dir_name \
--train_label_examples=gs://$BUCKET/examples/train_labeled_examples*.tfrecord \
--train_unlabel_examples=gs://$BUCKET/examples/train_unlabeled_examples*.tfrecord \
--test_examples=gs://$BUCKET/examples/test_examples*.tfrecord \
--augmentation_strategy=CTA \
--shuffle_seed=1 \
--num_parallel_calls=2 \
--keep_ckpt=0 \
--train_nimg=0
```


If the previous command doesn't return any errors, use the following command to push the newly built container to the registry:


```
$ docker push gcr.io/$PROJECT/ssl-train-uri
```


Then build the evaluation Docker container, and push it to the registry.


```
$ docker build ./ -f SslEvalDockerfile -t ssl-eval-uri
$ docker tag ssl-eval-uri:latest gcr.io/$PROJECT/ssl-eval-uri
$ docker push gcr.io/$PROJECT/ssl-eval-uri
```


Now you can launch the training and eval jobs on Cloud using the new containers by setting the `--train_docker_image_uri_path` and `--eval_docker_image_uri_path` flags.
