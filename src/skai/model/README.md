## Running Experiment on Vertex AI

### Prequisites
1. login to GCP
You must already have a GCP project. If you need to setup a new one, please skip step 1 and follow the more detailed xmanager GCP setup [here](https://github.com/deepmind/xmanager/tree/main#create-a-gcp-project-optional).
```
    export GCP_PROJECT=<GCP PROJECT ID>
    gcloud auth login
    gcloud auth application-default login
    gcloud config set project $GCP_PROJECT
    export GOOGLE_CLOUD_BUCKET_NAME=<GOOGLE_CLOUD_BUCKET_NAME>
```
Replace `<GCP PROJECT ID>` with your GCP project ID and same as `<GOOGLE_CLOUD_BUCKET_NAME>` with the name of the GCP bucket where data and other necessary files/folders are stored.


2. Clone the repo, create a virtual environment and install necessary packages. 
Notice that xmanager requires a python version >= 3.9. 

```
   git clone https://panford/skai.git
   cd skai
   conda create -n skai-env python==3.10   #python version should be >= 3.9
   pip install src/. xmanager ml-collections   #ensure you are in the main "skai" directory
```

3. Launch training using xmanager.
The following terminal commands allow you to launch training on Vertex AI from your local terminal. 
  

```
   xmanager launch src/skai/model/xm_launch_single_model_vertex.py -- \
      --xm_wrap_late_bindings \
      --xm_upgrade_db=True \
      --config=src/skai/model/configs/skai_config.py \ #path to data-specific configs
      --config.data.tfds_dataset_name=skai_dataset \  
      --config.data.tfds_data_dir=gs://skai-project/hurricane_ian \ 
      --config.output_dir=gs://skai-project/experiments/test_skai \
      --experiment_name=test_skai \
      --project_path=~/path/to/skai 
```
### A little more details on flags
`--config.data.tfds_data_dir=gs://skai-project/hurricane_ian` - Directory should contain the skai data to train/ evaluate on. This path should have a tree structure as shown below;
└── gs://path/to/dataset
   └── skai_dataset
       └── skai_dataset
           ├── 1.0.0
               ├── dataset_info.json
               ├── features.json
               ├── *.tfrecord*
                    .
                    .
                    .
               └── *.tfrecord*

`--project_path=~/path/to/skai` - Here, provide the entrypoint to the cloned skai project. for instance `/home/user/my_projects/skai`.