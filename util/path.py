import os


def get_root():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.split(root_dir)[0]


def absolute_path(relative_path):
    return os.path.join(get_root(), relative_path)


def append_path(module, relative_path):
    return os.path.join(get_dir(module), relative_path)


def get_dir(module):
    return os.path.dirname(os.path.abspath(module))


def is_project_in_cbica():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    return current_file_path.split('/')[1] == 'cbica'
