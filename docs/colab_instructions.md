# Instruction Document: SKAI Colab Notebooks

Last update: July 24, 2023

Before proceeding with these instructions, ensure that your Google Cloud project has been properly set up by following the steps outlined in [setup.md](setup.md).

This instruction document will guide you through the usage of two essential Colab notebooks:

1. [**Initialize_SKAI_Colab_Kernel.ipynb**](/src/colab/Initialize_SKAI_Colab_Kernel.ipynb): This notebook covers the initial setup and kernel initialization process required to start using SKAI on Google Colab.
2. [**Run_SKAI_Colab_Pipeline.ipynb**](/src/colab/Run_SKAI_Colab_Pipeline.ipynb): This notebook demonstrates how to run the SKAI automated machine learning pipeline on Google Colab.

We recommend making a copy of these Colab notebooks on your personal Github or your local Google Drive for easy access and customization.

## Notebook 1: Initialize_SKAI_Colab_Kernel.ipynb

### Step 1: Deploy Your Colab Virtual Machine

Before starting the assessment, configure and deploy your [Colab VM](https://console.cloud.google.com/marketplace/product/colab-marketplace-image-public/colab) from the Google Cloud Platform (GCP) Marketplace on Compute Engine.

Recommended Configuration:
* Machine type: N1 machine (e.g. standard-4)
* GPUs: No GPU is needed
* Boot Disk: SSD (e.g. 200GB)

Additionally, create a [service account](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fcloud.google.com%2Fiam%2Fdocs%2Fcreating-managing-service-account-keys%23iam-service-account-keys-create-console&link_redirector=1) to customize your initial Colab VM configuration.

### Step 2: Customize Your Colab Kernel

Open the Colab Notebook and connect to the previously created Colab VM using the "Connect to custom GCE VM" option.

In the first section **Settings** of the Colab Notebook, specify the following parameters and upload the service account key file:

1. Install the GDAL requirements.
2. Upload and activate the service account key file.
3. Install the Python dependencies required to run the SKAI Colab kernel.
4. Install the Python dependencies needed to run the SKAI source code, including cloning the Github repository and creating a virtual Python environment.

## Notebook 2: Run_SKAI_Colab_Pipeline.ipynb

This section provides guidance on running the SKAI pipeline using the Colab Notebook. If needed, you can always refer to [SKAI Damage Assessment Instructions](/docs/assessment_instructions.md) for more details on the underlying commands.

### Step 1: Upload Your Images and Other Input Files to GCP Project Bucket

Create a bucket folder in your GCP project and upload the following inputs for the SKAI pipeline:

1. Pre-disaster and post-disaster satellite image files.
2. Area Of Interest (AOI) GIS files (optional).
3. Building Footprints CSV file (optional).

Assuming the directory bucket path into your images and input variables, as set up in section Step 1., is `gs://bucket_path/`.

### Step 2: Set Pipeline Parameters

#### Step 2.1: Disaster Use Case Characteristics and GCP Project Parameters

Configure the directory of the workspace, specify information regarding the disaster and project:

* **`Disaster`**: Type of the disaster you are assessing (e.g., Cyclone).
* **`Year`**: Year of when the disaster happened (e.g., 2023).
* **`Month`**: Month of when the disaster happened (e.g., 05).
* **`Name`**: Name of the disaster you are assessing (e.g., Mocha).
* **`Country`**: Name of the country or area you are assessing (e.g., BGD).
* **`Organisation`**: Name of your organization performing the assessment (e.g., WFP).
* **`Run`** (optional): Version of the running pipeline (e.g., v0).

* **`GCP_PROJECT`**, **`GCP_LOCATION`**, **`GCP_SERVICE_ACCOUNT`**: Parameters from your cloud project (e.g., `skai-project`, `europe-west1`, and `skai-colab@skai-project.iam.gserviceaccount.com`).

* **`BCKT_VERSION`** (optional): Version of the bucket created (e.g., `ops`).

* **`Author`** (optional): Author's name running the pipeline (e.g., `amba`).

Ultimately, the workspace directory will be automatically created in the GCP Cloud Storage: `bucket/folder`: `skai-project_colab_bucket_ops_amba/wfp-cyclone-mocha-bdg-202305_v0`.

#### Step 2.2: Images Inputs Variables

* **`FILE_IMAGE_BEFORE`**: Directory path to the pre-disaster images in the GCP Cloud Storage.
* **`FILE_IMAGE_AFTER`**: Directory path to the post-disaster images in the GCP Cloud Storage.
* **`IMAGE_PREFIX_BEFORE`** (optional): Comma-separated list of the prefix names of the pre-disaster images in GCP Cloud Storage.
* **`IMAGE_PREFIX_AFTER`** (optional): Comma-separated list of prefix names of the post-disaster images in GCP Cloud Storage.

You can choose from three options to specify the image paths:
##### Option 1: Using Explicit Path, Single Pre and Post-Disaster Images
* **`FILE_IMAGE_BEFORE`** : `gs://bucket_path/pre_image.tif`
* **`FILE_IMAGE_AFTER`** : `gs://bucket_path/post_image.tif`

Leave **`IMAGE_PREFIX_BEFORE`** and **`IMAGE_PREFIX_AFTER`** blank.

##### Option 2: Using Pattern Path, All Pre and Post-Disaster Images
If you have multiple images which are named/indexed with the same pattern, and want to select all of them, e.g. `pre_image_0,pre_image_1` and `pre_image_0,pre_image_1`  then:
* **`FILE_IMAGE_BEFORE`** : `gs://bucket_path/pre_image_*.tif`
* **`FILE_IMAGE_AFTER`** : `gs://bucket_path/post_image_*.tif`

Leave **`IMAGE_PREFIX_BEFORE`** and **`IMAGE_PREFIX_AFTER`** blank.

##### Option 3: Using Pattern Path, Selected Pre and Post-Disaster Images

If you have multiple images which are named/indexed with the same pattern, and want to select only some of them, e.g. `pre_image_0,pre_image_10, ...` and `pre_image_2,pre_image_8, ...`  then:
* **`FILE_IMAGE_BEFORE`** : `gs://bucket_path/pre_image_*.tif`
* **`FILE_IMAGE_AFTER`** : `gs://bucket_path/post_image_*.tif`

* **`IMAGE_PREFIX_BEFORE`** : `0,10`
* **`IMAGE_PREFIX_AFTER`** : `2,8`


#### Step 2.3: Other Inputs Variables

* **`FILE_IMAGE_AOI`** (optional): Path to the Area of Interest file. If omitted, the entire area covered by your post-image(s) will be considered the area of interest.

You have also the option to upload a file containing a labeled dataset for training and evaluation.
* **`FILE_IMAGE_LABELED`** (optional): File path to the dataset with labeled examples.
* **`KEY_IMAGE_LABELED`** (optional): Key property to use as a label.
* **`MAPPING_IMAGE_LABELED`** (optional): Comma-separated list of the mapping of labels from the dataset and classes for the model.


* **`BUILDING_DETECTION_METHOD`**: Type of the method to perform building detection and fetching. This is discussed in more detail in the Building Detection section of the [SKAI Damage Assessment Instructions](/docs/assessment_instructions.md).
* **`BUILDINGS_CSV`** (optional): File path to the file containing the building centroids if **`BUILDING_DETECTION_METHOD`** is set to `file`.

#### Step 2.4: Labeling Information

* **`EMAIL_MANAGER`**: Email of the owner and responsible for the labeling task.

* **`EMAIL_ANNOTATORS`**: Comma-delimited list of the emails of people who will be labeling example images. They must be Google email accounts, such as GMail or GSuite email accounts.

### Step 3: Run Pipeline Assessment

Each step of the pipeline assessment is fully detailed and described in the Colab Notebook itself. The pipeline follows the same steps as described in [SKAI Damage Assessment Instructions](/docs/assessment_instructions.md), but with additional steps related to specific Colab parametrization and visualizations.

We will cover only these last specificities.

#### Step 3.1: Data Labeling

As a first step, you can visualize the pre- and post-images selected during the setup step for the assessment.

Then, by selecting the parameter **`GENERATING_JOB`**, you can choose to generate `unlabeled` examples or `labeled` examples (from the **`FILE_IMAGE_LABELED`** previously specified).

For the creation of the **labeling** task (in case of unlabeled generated examples), you can choose the maximum number of images to sample, e.g., `1000`.

Finally, by selecting the parameter **`LABELING_JOB`**, you can monitor the **progress** of this task, either by default taking the last labeling task created by the Notebook (`runtime_saved`), or by specifying the `id` of the task with the parameter **`JOB_ID`**.

#### Step 3.2: Building Training and Evaluation Dataset

Once your previous labeling task is completed, by selecting the parameter **`LABELING_DATASET`**, you can generate your training and evaluation datasets, either by default taking the last labeled dataset created by the Notebook (`runtime_saved`), or by specifying the `id` of the task with the parameter **`DATASET_ID`**.

The Notebook also offers you the opportunity to visualize a sample of your training and evaluation datasets. By default, a maximum of 100 labeled examples is outputted for inspection.

#### Step 3.3: Model Training and Performance Evaluation

The training and evaluation of the model are automatically performed sequentially after running the cell. You can select two parameters:
* **`LEARNING_METHOD`**: The learning technique to apply for the training, either `semi_supervised` (by default, fixmatch algorithm) or `fully_supervised`.
* **`LOAD_TENSORBOARD`**: Create a Tensorboard resource instance to monitor the progress and performance of the training and evaluation jobs.

An **experimentation** job will be created for each execution of the cell, including both training and evaluation jobs.

Once both jobs are running, you can close the notebook and reopen it later; the jobs keep running in the background, and you will then have access to the last status for monitoring.

#### Step 3.4: Inference Prediction

During the training or once your training is completed, run the model to infer predictions on all the examples generated in section Step 3.1. For this objective, specify the following parameters:

* **`EXPER_JOB`**: The experimentation job, either the last job launched and saved by the Notebook (`runtime_saved`), or specify the `name` with the parameter **`JOB_NAME`**.
* **`EVAL_JOB`**: The evaluation job, either the last job launched and saved by the Notebook (`runtime_saved`), or specify the `name` or `id` with the parameter **`JOB_ID_NAME`**.

These parameters will enable you to select the epoch you want to use for the **inference** job. You can choose from the following options for **`MODEL_CHECKPOINT`**:
1. `most_recent`: The last checkpoint created.
2. `top_auc_test`: The checkpoint with the best AUC on the evaluation dataset.
3. `top_acc_test`: The checkpoint with the best accuracy on the evaluation dataset.
4. `index_number`: Specifying a checkpoint number with the parameter **`INDEX_NUMBER`**.

Finally, you are able to visualize the pre and post-disaster images and the predictions from the inference by running the last cell.

## Feedback

If you have any feedback on these instructions or SKAI in general, we'd love to hear from you. Please reach out to the developers at skai-developers@googlegroups.com, or create an issue in the Github issue tracker.
