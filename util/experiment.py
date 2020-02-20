import configparser
import os
import psutil
from util.path import get_dir


def get_experiment_params(file, name):
    """
    Creates a ConfigParser object with parameter info for reading fMRI data.
    :return: params, a ConfigParser object
    """
    experiment_path = get_dir(file)

    params = configparser.ConfigParser()
    params_furl = os.path.join(experiment_path, 'conf', 'experiment.ini')
    params.read(params_furl)

    params.add_section('FILE')
    params.set('FILE', 'experiment_path', experiment_path)
    params.set('FILE', 'experiment_name', name)

    return params


def print_memory():
    divider = 1024. * 1024.
    percent = psutil.virtual_memory().percent
    available = psutil.virtual_memory().available / divider
    used = psutil.virtual_memory().used / divider

    mem_str = 'memory usage: {:.2f}%, used: {:.1f} MB, available: {:.1f} MB' \
        .format(percent, used, available)

    return mem_str
