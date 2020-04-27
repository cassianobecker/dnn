#! /bin/bash

source activate dnn2

echo Host name: `hostname`

python -u -c "from fwk.experiment import Experiment; Experiment.run('${1}')"