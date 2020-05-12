from dataset.mnist.processor.movement import process_movement
from dataset.mnist.processor.synth import generate_image, generate_ground_truth_movement
from dataset.mnist.processor.fsleyes import open_fsl_eyes
from dataset.mnist.processor.dwi_fit import fit_dti, fit_odf


def test_dwi_fit():

    generate_image()
    generate_ground_truth_movement()
    # process_movement()
    fit_dti()
    # fit_odf()
    # open_fsl_eyes()


if __name__ == '__main__':
    test_dwi_fit()
