#!/bin/bash

qsub -l gpu -l h_vmem=32G fwk/shell/node_gpu.sh ${1}