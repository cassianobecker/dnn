# Synthetic Dataset Generation

The following framework allows the user to generate synthetic white matter phantom datasets, using Fiberfox. [1]

The general stucture of the pipeline is:
```
  1. Create tractogram files
  2. Setup Diffusion Weighted Imaging (DWI) parameters
  3. Simulate DWI
  4. Transfer the files from the container
  5. Fit Diffusion Tensor Imaging (DTI)
  6. Fit Orientation Distribution Functions (ODF)
```

The overall pipeline can be run using:
```
sample_id = 'subject_01'
dataset = SynthProcessor()
dataset.process_subject(sample_id) 
``` 

Each processing step of the `process_subject` command is explored in detail below:

### 1. Create tractogram files

Usage:
```
sample_id = 'subject_01'
dataset = SynthProcessor()
dataset.create_tractogram_files(sample_id)
```  

The command `create_tracrogram_files`:
* Creates a phantom regression dataset using FiberCup. [1-3] Parameters (such as number of nodes, depth etc.) can be altered in the `fibercup.py` file
* Generates the tractogram

### 2. Setup DWI parameters

Usage:
```
sample_id = 'subject_01'
dataset = SynthProcessor()
dataset.setup_dwi_params(sample_id)
```  

The command `setup_dwi_params`:
* Copies the DWI parameters (bval and bvec files) to a designated directory (sample_id/params/)
* Flips the corresponding eigenvectors for compatibility between MITK's Fiberfox and FSL's dtifit

### 3. Simulate DWI

Usage:
```
sample_id = 'subject_01'
dataset = SynthProcessor()
dataset.simulate_dwi(sample_id)
```  

The command `simulate_dwi`:
* Setups directories and files (from steps 1. and 2. above) for container (Docker or Singularity) use
* Runs the DWI simulation

### 4. Transfer the files from the container

Usage:
```
sample_id = 'subject_01'
dataset = SynthProcessor()
dataset.transfer_files_from_container(sample_id)
```  

The command `transfer_files_from_container`:
* Tranfer the results of the simulation from Step 3. from the container to the designated local directory

### 5. Fit DTI

Usage:
```
sample_id = 'subject_01'
dataset = SynthProcessor()
dataset.fit_dti(sample_id)
```  

The command `fit_dti`:
* Use FSL's `dtifit` command to fit diffusion tensors to the simulated data

### 6. Fit ODF

Usage:
```
sample_id = 'subject_01'
dataset = SynthProcessor()
dataset.fit_odf(sample_id)
```  

The command `fit_odf`:
* Apply a Constrained Spherical Deconvolution model to compute the fiber ODF of the provided simulated data

## References

1. Neher PF, Laun FB, Stieltjes B, Maier-Hein KH. Fiberfox: facilitating the creation of realistic white matter software phantoms. Magn Reson Med. 2014;72(5):1460-70.
2. Poupon C, Rieul B, Kezele I, Perrin M, Poupon F, Mangin JF. New diffusion phantoms dedicated to the study and validation of HARDI models. Magn Reson Med 2008;60:1276–1283.
3. Fillard P, Descoteaux M, Goh A, et al. Quantitative evaluation of 10 tractography algorithms on a realistic diffusion MR phantom. Neuroimage 2011;56:220–234.
