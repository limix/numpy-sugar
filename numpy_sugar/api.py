import os


def get_include():
    import limix_math as nsugar
    d = os.path.join(os.path.dirname(nsugar.__file__), 'include')
    return d


def get_lib():
    import limix_math as nsugar
    d = os.path.join(os.path.dirname(nsugar.__file__), 'lib')
    return d
