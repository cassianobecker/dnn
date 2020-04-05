#!/bin/bash

# copy FA template
in_file='/usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/templates/FMRIB58_FA_1mm.nii.gz'
cp ${in_file} ${out_file}

# reslice template
in_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/templates/FMRIB58_FA_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/templates/FMRIB58_FA_2mm.nii.gz'
flirt -interp nearestneighbour -in ${in_file} -ref ${in_file} -applyisoxfm 2 -out ${out_file}

# copy and binarise mask
in_file='/usr/local/fsl/data/standard/FMRIB58_FA-skeleton_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/templates/FMRIB58_FA-skeleton_1mm.nii.gz'
fslmaths ${in_file} -bin ${out_file}

# reslice mask
in_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/templates/FMRIB58_FA-skeleton_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/templates/FMRIB58_FA-skeleton_2mm.nii.gz'
flirt -interp nearestneighbour -in ${in_file} -ref ${in_file} -applyisoxfm 2 -out ${out_file}
