# DTI-DNN

Short description here.

### Setting up

##### Conda environment
* To create a conda environment, `cd` to the directory.], and
  * run `conda env create -f environment.yml` on a local machine and
  * run `conda env create -f environment_cbica.yml` on the CBICA cluster.
    * To download `conda`, visit [anaconda.com](https://www.anaconda.com/distribution/).
* To activate environment in the command line,
  * run `conda activate dnn` for a local machine and 
  * run `conda activate dnn-cbica` on the cluster.
* To activate environment in PyCharm
  * Open PyCharm. Open `dnn` directory as a project.
  * Then, select "Preferences" > "Project: dnn" > ⚙️ > "Add..." > "Conda Environment" > "Existing environment" > "/Users/username/anaconda3/envs/dti-enn/bin/python"
 
 ##### HCP database setup
* Download the HCP covariates file ([link](https://db.humanconnectome.org/REST/search/dict/Subject%20Information/results?format=csv&removeDelimitersFromFieldValues=true&restricted=0&project=HCP_1200)), and save as `dataset/hcp/res/hcp_covariates.csv`. 
* Get AWS credentials to HCP (see [link](https://wiki.humanconnectome.org/plugins/viewsource/viewpagesrc.action?pageId=67666030)).
  * Save to file `~/.aws/credentials` in you home folder with conents 
    ```
    [default]
	aws_access_key_id = (your HCP access key here) 
	aws_secret_access_key = (your HCP secret access key here)
    ```
  * Using [CLI](https://aws.amazon.com/cli/), run `aws configure`.

##### DTI
* Download and install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).
* Download and install [ANTs](https://github.com/ANTsX/ANTs)

### Getting started

##### HCP Data

1. To download & process the data, at the root folder of the project, run `python -m dataset.hcp.scripts.dti.process`. 
    * List the subjects to download in `dataset/hcp/scripts/dti/conf/subjects.txt`.
    * The downloaded and post-processed files will be stored in '~/.dnn/datasets/hcp', accessible from your home folder.
2. To train and test a Diffusion-CNN on the data, at the root folder of the project, select the arg.ini file describing your experiment, and run `python schedule (local/debug/cbica) experiments/hcp/conf/args.ini`. 
    * The list of subjects for training and testing is located in  `experiments/hcp/conf/(test|train)_subjects.txt`.
    * The results of you experiments will be stored in '~/.dnn/results' under a folder named according to the data and time when you then experiment was run.
