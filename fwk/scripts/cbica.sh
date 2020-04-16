#!/bin/bash

qsub -l gpu -l h_vmem=32G fwk/scripts/cbica_node.sh -o ${2} -e ${2} ${1}
