import psutil
import importlib


def print_memory():
    divider = 1024. * 1024.
    percent = psutil.virtual_memory().percent
    available = psutil.virtual_memory().available / divider
    used = psutil.virtual_memory().used / divider

    mem_str = 'memory usage: {:.2f}%, used: {:.1f} MB, available: {:.1f} MB' \
        .format(percent, used, available)

    return mem_str


def class_for_name(full_class_name):

    module_name, _, class_name = full_class_name.rpartition('.')
    the_class = getattr(importlib.import_module(module_name), class_name)

    return the_class


