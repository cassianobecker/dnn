#! /bin/bash

source activate dnn-cluster

module load mrtrix

echo Host name: `hostname`

python -u -c "from fwk.experiment import Experiment; Experiment.run('${1}')"
