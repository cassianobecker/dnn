#!/bin/bash

source activate dnn

#module load mrtrix

python -c "from fwk.experiment import Experiment; Experiment.run('${1}')"