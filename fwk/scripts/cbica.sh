#!/bin/bash

qsub -l gpu -l h_vmem=32G fwk/scripts/node.sh -o ${2} -e ${2} ${1}
