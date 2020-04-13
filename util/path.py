import os


def get_root():
    """
    gets the project's root directory
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.split(root_dir)[0]


def absolute_path(relative_path):
    return os.path.join(get_root(), relative_path)


def append_path(module, relative_path):
    return os.path.join(get_dir(module), relative_path)


def get_dir(module):
    """
    gets the module's current directory
    :param module: the client should call by setting module = __file__
    """
    return os.path.dirname(os.path.abspath(module))
