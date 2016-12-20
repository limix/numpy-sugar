import os


def get_include():
    import numpy_sugar as nsugar
    d = os.path.join(os.path.dirname(nsugar.__file__), 'include')
    return d


def get_lib():
    import numpy_sugar as nsugar
    d = os.path.join(os.path.dirname(nsugar.__file__), 'lib')
    return d
