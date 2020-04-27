
#! /bin/bash

source activate dnn2

nvcc --version
echo Host name: `hostname`
nvidia-smi

CUDA_VISIBLE_DEVICES=`get_CUDA_VISIBLE_DEVICES` || exit
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES

python -u -c "from fwk.experiment import Experiment; Experiment.run('${1}')"