#!/bin/bash

qsub -l gpu -l h_vmem=32G fwk/scripts/node.sh ${1}