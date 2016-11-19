import os


def get_include():
    import limix_math as lmath
    d = os.path.join(os.path.dirname(lmath.__file__), 'include')
    return d


def get_lib():
    import limix_math as lmath
    d = os.path.join(os.path.dirname(lmath.__file__), 'lib')
    return d
