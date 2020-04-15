#!/bin/bash

qsub -l gpu -l h_vmem=32G fwk/scripts/cbica_node.sh ${1}
