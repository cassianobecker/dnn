#!/bin/bash

# copy FA template
in_file='/usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA_1mm.nii.gz'
cp ${in_file} ${out_file}

# create inclusive FA mask
in_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA-mask_1mm.nii.gz'
fslmaths ${in_file} -thr 3000 -bin ${out_file}

# reslice inclusive FA mask
in_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA-mask_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA-mask_125mm.nii.gz'
flirt -interp nearestneighbour -in ${in_file} -ref ${in_file} -applyisoxfm 1.25 -out ${out_file}

# reslice template
in_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA_125mm.nii.gz'
flirt -interp nearestneighbour -in ${in_file} -ref ${in_file} -applyisoxfm 1.25 -out ${out_file}

# copy and binarise mask
in_file='/usr/local/fsl/data/standard/FMRIB58_FA-skeleton_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA-skeleton_1mm.nii.gz'
fslmaths ${in_file} -bin ${out_file}

# reslice mask
in_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA-skeleton_1mm.nii.gz'
out_file='/Users/lindenmp/Dropbox/Work/ResProjects/dnn/dataset/hcp/param/templates/FMRIB58_FA-skeleton_125mm.nii.gz'
flirt -interp nearestneighbour -in ${in_file} -ref ${in_file} -applyisoxfm 1.25 -out ${out_file}
