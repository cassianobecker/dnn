import os


def get_root():
    """
    gets the project's root directory
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.split(root_dir)[0]


def get_dir(module):
    """
    gets the module's current directory
    :param module: the client should call by setting module = __file__
    """
    return os.path.dirname(os.path.abspath(module))
