import os, glob
import nibabel as nib
import numpy as np
from IPython.display import clear_output


def update_progress(progress, my_str = ''):
    """
	Progress bar
    """
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = my_str + " Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


def return_dti_derivs(subjpath):
    """
    Reads in eigenvectors and eigenvalues from DTI fit and return v and l
    v = 3 * i * j * k * 3 matrix, where 
        dim[0] = eigenvectors 1,2 and 3
        dims[i,j,k] = voxels from image
        dim[4] = eigenvector values
    l = 3 * i * j * k matrix, where 
        dim[0] = eigenvalues 1,2 and 3
        dims[i,j,k] = voxels from image
    """
    
    # identify files
    v1_file = glob.glob(os.path.join(subjpath,'*V1*'))[0]
    v2_file = glob.glob(os.path.join(subjpath,'*V2*'))[0]
    v3_file = glob.glob(os.path.join(subjpath,'*V3*'))[0]
    l1_file = glob.glob(os.path.join(subjpath,'*L1*'))[0]
    l2_file = glob.glob(os.path.join(subjpath,'*L2*'))[0]
    l3_file = glob.glob(os.path.join(subjpath,'*L3*'))[0]

    # load niftis and stack eigenvecs
    v1_img = nib.load(v1_file); v1 = v1_img.get_fdata()
    v2_img = nib.load(v2_file); v2 = v2_img.get_fdata()
    v3_img = nib.load(v3_file); v3 = v3_img.get_fdata()
    v = np.stack((v1,v2,v3), axis = 0)
    
    # load niftis and stack eigenvals
    l2_img = nib.load(l2_file); l2 = l2_img.get_fdata()
    l1_img = nib.load(l1_file); l1 = l1_img.get_fdata()
    l3_img = nib.load(l3_file); l3 = l3_img.get_fdata()
    l = np.stack((l1,l2,l3), axis = 0)
    
    return v, l


def get_dti_object(v, l, report_progress = False):
    """
    Takes in eigenvectors and eigenvalues and returns 3*3*i*j*k DTI array for input to nn
    """
    dt = np.zeros([3, 3, v.shape[1], v.shape[2], v.shape[3]])

    for i in range(v.shape[1]):
        if report_progress: update_progress(i/v.shape[1])
        for j in range(v.shape[2]):
            for k in range(v.shape[3]):
                dt[:, :, i, j, k] = l[0,i,j,k]*np.outer(v[0,i,j,k], v[0,i,j,k]) +\
                					l[1,i,j,k]*np.outer(v[1,i,j,k], v[1,i,j,k]) +\
                					l[2,i,j,k]*np.outer(v[2,i,j,k], v[2,i,j,k])
    if report_progress: update_progress(1)    

    return dt


def build_dti_tensor_image(subjpath):
    """
    Reads in eigenvectors and eigenvalues from DTI fit and returns  3*3*i*j*k DTI array for input to nn
    """
    dti_tensor = 0
    for i in range(1,4):
        evecs_file = glob.glob(os.path.join(subjpath, '*V' + str(i) + '*'))[0]
        evals_file = glob.glob(os.path.join(subjpath, '*L' + str(i) + '*'))[0]
        evecs = nib.load(evecs_file).get_fdata()
        evals = nib.load(evals_file).get_fdata()
        dti_tensor = dti_tensor + np.einsum('abc,abci,abcj->ijabc', evals, evecs, evecs)
    return dti_tensor

# subjpath = '/Users/lindenmp/Dropbox/Work/ResProjects/Becker_DNN/dataset/hcp/reg_test/100307ToTemplate'
# v, l = return_dti_derivs(subjpath)
# dt = get_dti_object(v, l, report_progress = True)
# dt.shape
