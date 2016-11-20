from os.path import join


def _make():
    from cffi import FFI

    ffi = FFI()

    from ncephes import get_include
    from ncephes import get_lib

    sources = [join('numpy_sugar', 'special', 'special.c')]
    hdr = join('numpy_sugar', 'include', 'numpy_sugar', 'special.h')
    incls = [join('numpy_sugar', 'include'), get_include()]

    from distutils.ccompiler import new_compiler
    compiler = new_compiler()
    libraries = ['ncprob', 'nmisc']
    if 'msv' not in compiler.__class__.__name__.lower():
        libraries += ['m']

    ffi.set_source(
        'numpy_sugar.special._special_ffi',
        '#include "numpy_sugar/special.h"',
        include_dirs=incls,
        sources=sources,
        extra_compile_args=['-Ofast'],
        libraries=libraries,
        library_dirs=[get_lib()])

    with open(hdr, 'r') as f:
        data = f.read()
        data = data.replace("#ifndef SPECIAL_H\n", "")
        data = data.replace("#define SPECIAL_H\n", "")
        data = data.replace("#endif\n", "")
        ffi.cdef(data)

    return ffi


special = _make()
