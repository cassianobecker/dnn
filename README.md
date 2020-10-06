# DNN

Diffusion neural network

## Setting up

#### Conda environment
* To create a conda environment, `cd` to the directory, and
  * run `conda env create -f environment.yml` on a local machine and
  * run `conda env create -f environment_cbica.yml` on the CBICA cluster.
    * To download `conda`, visit [anaconda.com](https://www.anaconda.com/distribution/).
* To activate environment in the command line,
  * run `conda activate dnn` for a local machine and 
  * run `conda activate dnn-cbica` on the cluster.
* To activate environment in PyCharm
  * Open PyCharm. Open `dnn` directory as a project.
  * Then, select "Preferences" > "Project: dnn" > ⚙️ > "Add..." > "Conda Environment" > "Existing environment" > "/Users/username/anaconda3/envs/dti-enn/bin/python"
 
#### HCP database setup
* Download the HCP covariates file ([link](https://db.humanconnectome.org/REST/search/dict/Subject%20Information/results?format=csv&removeDelimitersFromFieldValues=true&restricted=0&project=HCP_1200)), and save as `dataset/hcp/res/hcp_covariates.csv`. 
* Get AWS credentials to HCP (see [link](https://wiki.humanconnectome.org/plugins/viewsource/viewpagesrc.action?pageId=67666030)).
  * Save to file `~/.aws/credentials` in you home folder with conents 
    ```
    [default]
	aws_access_key_id = (your HCP access key here) 
	aws_secret_access_key = (your HCP secret access key here)
    ```
  * Using [CLI](https://aws.amazon.com/cli/), run `aws configure`.

#### DTI
* Download and install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).
* Download and install [ANTs](https://github.com/ANTsX/ANTs)

## Getting started

#### Downloading & processing HCP Data
* List the subjects to download and process in `dataset/hcp/dti/scripts/conf/subjects.txt`.
* Go the root folder of the project.
* Run `python schedule (local/debug/cbica_cpu) dataset/hcp/dti/scripts/conf/args.ini`. 
	* The downloaded and post-processed files will be stored in `~/.dnn/datasets/hcp`, accessible from your home folder.
	* On the UPenn CBICA cluster, you can use Cassiano's processed data in `/cbica/home/beckerc/.dnn/datasets/hcp/processing`. Set `local_processing_directory` in your `.ini` experiment configuration file in `experiments/hcp/conf`.

#### Training a Diffusion-CNN
* Select the `arg.ini` file describing your experiment.
* Go to the root folder of the project.
* Run `python schedule (local/debug/cbica_cpu/cbica_gpu) experiments/hcp/conf/args.ini`.
	* The list of subjects for training and testing is located in  `experiments/hcp/conf/(test|train)_subjects.txt`.
    	* The results of you experiments will be stored in '~/.dnn/results' under a folder named according to the data and time when you then experiment was run.
	* See list of covariates [here](https://wiki.humanconnectome.org/display/PublicData/HCP+Data+Dictionary+Public-+Updated+for+the+1200+Subject+Release#HCPDataDictionaryPublic-Updatedforthe1200SubjectRelease-Instrument:FluidIntelligence(PennProgressiveMatrices)).

## Running [MITK](https://www.mitk.org/wiki/The_Medical_Imaging_Interaction_Toolkit_(MITK)) on Docker

See README.md on [dataset/synth](https://github.com/cassianobecker/dnn/tree/master/dataset/synth]. 
