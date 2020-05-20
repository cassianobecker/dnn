import subprocess


def open_fsl_eyes():
    print(f"Opening fsleyes...")
    fsl_str = fsl_eyes_str()
    process_str = f'{fsl_str}'
    subprocess.Popen(process_str, shell=True)


def fsl_eyes_str():

    str_fsl_eyes = 'fsleyes --scene ortho --worldLoc 4.499949932098389 5.499949932098389 6.499949932098389 ' \
                   '--displaySpace /Users/cassiano/.dnn/datasets/synth/synth.nii.gz ' \
                   '--xcentre  0.00000  0.00000 ' \
                   '--ycentre  0.00000  0.00000 ' \
                   '--zcentre  0.00000  0.00000 ' \
                   '--xzoom 100.0 --yzoom 100.0 --zzoom 100.0 ' \
                   '--layout horizontal ' \
                   '--bgColour 0.0 0.0 0.0 --fgColour 1.0 1.0 1.0 ' \
                   '--cursorColour 0.0 1.0 0.0 --colourBarLocation top --colourBarLabelSide top-left ' \
                   '--colourBarSize 100.0 --labelSize 12 --performance 3 ' \
                   \
                   '--movieSync /Users/cassiano/.dnn/datasets/synth/synth.nii.gz --name "synth" ' \
                   '--overlayType volume --alpha 89.89219960986921 --brightness 50.0 --contrast 50.0 ' \
                   '--cmap greyscale --negativeCmap greyscale --displayRange -1.0354126021929868 1.0354126021929868 ' \
                   '--clippingRange -1.0354126021929868 1.0561208542368465 ' \
                   '--gamma 0.0 --cmapResolution 256 --interpolation none --numSteps 100 --blendFactor 0.1 --smoothing 0 ' \
                   '--resolution 100 --numInnerSteps 10 --clipMode intersection ' \
                   \
                   '--volume 0 /Users/cassiano/.dnn/datasets/synth/tensor_dti_truth.nii.gz --name "tensor_dti_truth" ' \
                   '--overlayType tensor --alpha 87.1396559959137 --brightness 50.0 --contrast 50.0 ' \
                   '--cmap greyscale --tensorResolution 10 --scale 100.0 ' \
                   '--xColour 1.0 0.0 0.0 --yColour 0.0 1.0 0.0 --zColour 0.0 0.0 1.0 ' \
                   '--suppressMode white --modulateRange 0.0 1.0 --clippingRange 0.0 1.0 ' \
                   \
                   '/Users/cassiano/.dnn/datasets/synth/tensor_dti.nii.gz --name "tensor_dti" ' \
                   '--overlayType tensor --alpha 87.10331879255905 --brightness 50.0 --contrast 50.0 ' \
                   '--cmap greyscale --tensorResolution 10 --scale 100.0 ' \
                   '--xColour 1.0 0.0 0.0 --yColour 0.0 1.0 0.0 --zColour 0.0 0.0 1.0 ' \
                   '--suppressMode white --modulateRange 0.0 1.0 --clippingRange 0.0 1.0'

    return str_fsl_eyes
