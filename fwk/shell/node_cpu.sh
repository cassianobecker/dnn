#! /bin/bash

source activate dnn-cluster

#module load mrtrix

echo Host name: `hostname`

cd ~/dnn

python -u -c "from fwk.experiment import Experiment; Experiment.run('${1}')"
