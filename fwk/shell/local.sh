#!/bin/bash

source activate dnn-cluster

module load mrtrix

python -c "from fwk.experiment import Experiment; Experiment.run('${1}')"