#!/bin/bash

qsub -l h_vmem=32G fwk/shell/node_cpu.sh ${1}