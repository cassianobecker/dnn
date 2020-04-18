# DTI-DNN

Short description here.

### Setting up

##### Conda environment
* Run `conda env create -f environment.yml` in the directory.
  * To download `conda`, visit [anaconda.com](https://www.anaconda.com/distribution/).
* To activate environment in the command line, run `conda activate dnn`.
* To activate environment in PyCharm
  * Open PyCharm. Open `dnn` directory as a project.
  * Then, select "Preferences" > "Project: dnn" > ⚙️ > "Add..." > "Conda Environment" > "Existing environment" > "/Users/username/anaconda3/envs/dti-enn/bin/python"
 
 ##### HCP database 
* Copy `dataset/hcp/conf/hcp_database(example).ini` and paste in the same directory as `dataset/hcp/conf/hcp_database.ini`.
* Replace fields in `[Directories]` with values for your environment
  * When on `CBICA`, set `local_server_directory = '/cbica/projects/HCP_Data_Releases'`.
* Download covariates ([link](https://db.humanconnectome.org/REST/search/dict/Subject%20Information/results?format=csv&removeDelimitersFromFieldValues=true&restricted=0&project=HCP_1200)), and save as `dataset/hcp/res/hcp_covariates.csv`. 
* For AWS access, get AWS credentials (see [link](https://wiki.humanconnectome.org/plugins/viewsource/viewpagesrc.action?pageId=67666030)).
  * Save to `~/.aws/credentials` as
    ```
    [default]
	aws_access_key_id = 
	aws_secret_access_key = 
    ```
  * Using [CLI](https://aws.amazon.com/cli/), run `aws configure`.

##### DTI
* Download and install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).

### Getting started

##### HCP Data

1. To download & process the data, at the root folder of the project, `python -m experiments.hcp.scripts.process.process_dti`. 
    * List the subjects to download in `dataset/hcp/scripts/process/conf/subjects.txt`.
2. To train and test a CNN on the data, at the root folder of the project, select the arg.ini file describing your experiment, and run `python schedule (local/debug/cbica) experiments/hcp/conf/args.ini`, for example. 
    * List the subjects for training and testing in `experiments/hcp/conf/(test|train)_subjects.txt`.
