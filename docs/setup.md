# SKAI Setup Instructions

Last update: December 15, 2022


## Cloud Setup

SKAI was designed primarily to run on Google Cloud. It relies on several Cloud components, including [Dataflow](https://cloud.google.com/dataflow) for data pre-processing and [Vertex AI](https://cloud.google.com/vertex-ai) for labeling examples and training models. Follow the instructions below to set up a cloud project.


### Create Google Cloud Project

Please follow these [instructions](https://cloud.google.com/resource-manager/docs/creating-managing-projects) to create a Google Cloud project.


### Enable Service APIs

SKAI uses the following Cloud services, so they must be [enabled](https://cloud.google.com/service-usage/docs/enable-disable). Visit each of the following links and click "Enable".



*   [Dataflow](https://console.cloud.google.com/apis/library/dataflow.googleapis.com) - Needed for pre-processing data.
*   [Vertex AI](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) - Needed for example labeling, model training, and model inference.
*   [Compute Engine](https://console.cloud.google.com/apis/library/compute.googleapis.com) - Needed for running virtual machines.


### Enable Google Private Access in Subnets

By default, every Dataflow worker machine is given an external IP. This external
IP is not needed for SKAI data processing jobs, and is actually problematic
because there's a quota on the number of external IPs your project can have in
any cloud region. By default, this quota is 69. This means that the number of
Dataflow workers your job can launch will be capped to 69 by default, even if
you set max_workers to a higher number.

To work around this problem, SKAI dataflow jobs are configured to run without
external IPs by default. However, this means that you need to enable
"Google Private Access" on the subnets your jobs will run on, so that the jobs
can still access resources such as Google Cloud Store. Follow these
[instructions](https://cloud.google.com/vpc/docs/configure-private-google-access#config-pga)

See [here](https://medium.com/google-cloud/eliminate-auto-scaling-bottlenecks-by-using-private-ips-for-dataflow-workers-23a8a73cecd5) for more details.

### Create Cloud Storage Bucket

A [Google Cloud Storage](https://cloud.google.com/storage) bucket is needed to store all satellite images, training examples, intermediate output, and assessment results during an assessment. Please follow these [instructions](https://cloud.google.com/storage/docs/creating-buckets) to create a bucket.

Defaults can be used for all choices except the bucket's [location](https://cloud.google.com/storage/docs/locations). The bucket should be located in the same region where you plan to run your Dataflow and Vertex AI training jobs to minimize data travel distance and latency.

When asked to choose a location, you should choose "Region" for location type, and one of the following regions:



*   If you are located in the United States: us-central1
*   If you are located in Europe: europe-west1
*   If you are located in Asia: asia-east1
*   For other locations, please read the following documentation to choose a suitable region:
    *   https://cloud.google.com/storage/docs/locations
    *   https://cloud.google.com/vertex-ai/docs/general/locations
    *   Pay attention to the "Feature availability" and "Using accelerators" sections. Choose a region that ideally has most of the Vertex AI features and Nvidia P100 GPU accelerators.

Once you have chosen the bucket's location, you should use that location for all SKAI pipelines you run.


### Create Service Account

A [service account](https://cloud.google.com/iam/docs/service-accounts) is an identity used by Cloud jobs such as Dataflow pipelines and ML training pipelines. Please create a service account for use with SKAI pipelines by following these [instructions](https://cloud.google.com/iam/docs/creating-managing-service-accounts).

Please grant the service account the "Owner" role, which will give it all necessary permissions to run all pipelines.

After the service account is created, please generate a private key for it to simplify authentication following these [instructions](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating). Download the key to the workstation where you will run the SKAI pipeline.

**Note**: Anyone who has this key file will be able to authenticate as the service account without needing to know your email or password. So please keep the key private and safe.


### Enable Earth Engine API (Optional)

If you want to visualize your satellite images, building footprints, and assessment results, you need to enable the Earth Engine API by following these [instructions](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup).

Please ensure that the Earth Engine API is enabled in your project by visiting this [page](https://console.cloud.google.com/apis/library/earthengine.googleapis.com).


### Create Linux VM (Optional)

SKAI runs in a Linux environment. If you don't have a Linux workstation, you can create a Linux virtual machine (VM) in your Cloud project and run SKAI from there. Please follow these [instructions](https://cloud.google.com/compute/docs/create-linux-vm-instance) to do that.



*   The VM will not have to run any heavy computation, so you can choose a configuration that minimizes the cost of the VM. The lowest-spec machine type, "n1-standard-1", should be sufficient.
*   The VM will also not have to hold much data, as most data will be stored in your GCS bucket. For the boot disk of the VM, you can choose a 10 GB balanced persistent disk with the most recent Debian GNU/Linux image.

After the virtual machine is created, you can log in with [SSH in your browser](https://cloud.google.com/compute/docs/ssh-in-browser). See the next section for instructions for setting up the Linux environment.


## Linux Setup Instructions

SKAI runs in a Linux environment. If you don't have a Linux workstation, you can create a Linux virtual machine (VM) in your Cloud project (explained above) and run SKAI from there.

Follow these steps to set up your Linux workstation to use SKAI.


### Clone SKAI git repo


```
$ git clone https://github.com/google-research/skai.git
```



### Install Python 3

SKAI is implemented in Python 3, which is installed by default on many popular Linux distributions.

In Debian Linux, you also need to run the following commands to install Virtualenv and OpenCV:


```
$ sudo apt-get install python3-venv python3-opencv
```


**Note about Python version:** As of December 21, 2022, the latest version of Dataflow supports Python versions 3.7-3.10. Please make sure that your workstation's Python installation is one of these versions. If not, you will have to manually install a compatible version of Python. See [here](https://cloud.google.com/dataflow/docs/support/beam-runtime-support) for more information.


### Install Google Cloud SDK

https://cloud.google.com/sdk/docs/install

Configure gcloud command line tool to use your project


```
$ gcloud init
$ gcloud config set project <Your project name>
$ gcloud auth application-default login

# Set environment variables
$ PROJECT=<Your project name>
$ LOCATION=<Region of project, e.g. us-central1 (NOT us-central1-a)>
$ BUCKET=<Name of bucket you created in Cloud Setup Instructions above>
```



### Set up virtualenv


```
$ python -m venv skai-env
$ source skai-env/bin/activate
$ pip install --upgrade pip
$ cd <skai-source-directory>
$ pip install -r requirements.txt
```


### Authenticate with Earth Engine (Optional)
* Some features in SKAI use earth engine, if you don't have an account, please sign-up [here](https://signup.earthengine.google.com/).

* After signing up, run the command `earthengine authenticate` in a terminal.

## Feedback

If you have any feedback on these instructions or SKAI in general, we'd love to hear from you. Please reach out to the developers at [skai-developers@googlegroups.com](mailto:skai-developers@googlegroups.com), or create an issue in the Github issue tracker.
