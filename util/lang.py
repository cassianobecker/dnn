import importlib


def class_for_name(full_class_name):

    module_name, _, class_name = full_class_name.rpartition('.')
    the_class = getattr(importlib.import_module(module_name), class_name)

    return the_class


def to_bool(value):
    """
       Converts 'something' to boolean. Raises exception for invalid formats
           Possible True  values: 1, True, "1", "TRue", "yes", "y", "t"
           Possible False values: 0, False, None, [], {}, "", "0", "faLse", "no", "n", "f", 0.0, ...
    """
    if str(value).lower() in ("yes", "y", "true",  "t", "1"): return True
    if str(value).lower() in ("no",  "n", "false", "f", "0", "0.0", "", "none", "[]", "{}"): return False
    raise Exception('Invalid value for boolean conversion: ' + str(value))


def is_method_implemented(instance, method_name):
    return method_name in instance.__class__.__dict__

