#! /bin/bash

source activate dnn-cluster

echo Host name: `hostname`

python -u -c "from fwk.experiment import Experiment; Experiment.run('${1}')"
