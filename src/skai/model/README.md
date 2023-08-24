## Running Experiment on Vertex AI

### Prequisites
1. login to GCP.

You must already have a GCP project. If you need to setup a new one, please skip
step 1 and follow the more detailed XManager GCP setup
[here](https://github.com/deepmind/xmanager/tree/main#create-a-gcp-project-optional).

```
    export GCP_PROJECT=<GCP PROJECT ID>
    gcloud auth login
    gcloud auth application-default login
    gcloud config set project $GCP_PROJECT
    export GOOGLE_CLOUD_BUCKET_NAME=<GOOGLE_CLOUD_BUCKET_NAME>
```

Replace `<GCP PROJECT ID>` with your GCP project ID and same as
`<GOOGLE_CLOUD_BUCKET_NAME>` with the name of the GCP bucket where data and
other necessary files/folders are stored.

2. Clone the repo, create a virtual environment and install necessary packages.
Note that XManager requires a python version >= 3.9.

```
   git clone https://github.com/panford/skai.git
   cd skai
   conda create -n skai-env python==3.10   #python version should be >= 3.9
   conda activate skai-env
   pip install src/. xmanager ml-collections   #ensure you are in the main "skai" directory
```

3. Launch training using XManager. The following terminal commands allow you to
launch training on Vertex AI from your local terminal.
  

```
xmanager launch src/skai/model/xm_launch_single_model_vertex.py -- \
    --xm_wrap_late_bindings \
    --xm_upgrade_db=True \
    --config=src/skai/model/configs/skai_two_tower_config.py \
    --config.data.tfds_dataset_name=skai_dataset \
    --config.data.tfds_data_dir=gs://skai-data/hurricane_ian \
    --config.output_dir=gs://skai-data/experiments/test_skai \
    --experiment_name=test_skai \
    --project_path=~/path/to/skai \
    --accelerator=V100 \
    --accelerator_count=1
```
### A little more details on flags
 a. `--config.data.tfds_data_dir=gs://skai-project/hurricane_ian` - Directory
 should contain the skai data to train/ evaluate on. This path should have a
 tree structure as shown below;
```
gs://path/to/dataset 
    └── skai_dataset
         └── skai_dataset
             └── 1.0.0
                 ├── dataset_info.json
                 ├── features.json
                 ├── *.tfrecord*
                     .
                     .
                     .
                 └── *.tfrecord*
```

 b. `--project_path=~/path/to/skai` - Here, provide the entrypoint to the cloned
 skai project. for instance `/home/user/my_projects/skai`.

 c. The `--accelerator` and `--accelerator_count` flags provides the options to
 choose an accelerator type and the number of this accelerator to run
 experiments on.

 ```
 # Example 1
    # Each trial runs on 1 V100 GPU
 ...
    --accelerator=V100 
    --accelerator_count=1
 ...


 # Example 2
   """
    Each trial runs on 8 v2 TPUs. 
    Note that the supported "accelerator_count" for TPU_V2 and TPU_V3 is 8.
        ref: https://github.com/deepmind/xmanager/blob/main/docs/executors.md
   """
 ...
    --accelerator=TPU_V2
    --accelerator_count=8
 ```

Some accelerator options are listed below.

    GPU ACCELERATORS:  P100, V100, P4, T4, A100
    TPU ACCELERATORS:  TPU_V2, TPU_V3
