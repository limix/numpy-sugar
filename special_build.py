from os.path import join


def _make():
    from cffi import FFI

    ffi = FFI()

    from ncephes import get_include

    sources = [join('numpy_sugar', 'special', 'special.c')]
    hdr = join('numpy_sugar', 'include', 'numpy_sugar', 'special.h')
    incls = [join('numpy_sugar', 'include')]

    from distutils.ccompiler import new_compiler
    compiler = new_compiler()
    libraries = ['hcephes']
    if 'msv' not in compiler.__class__.__name__.lower():
        libraries += ['m']

    ffi.set_source(
        'numpy_sugar.special._special_ffi',
        '#include "numpy_sugar/special.h"',
        include_dirs=incls,
        sources=sources,
        extra_compile_args=['-Ofast'],
        libraries=libraries)

    with open(hdr, 'r') as f:
        data = f.read()
        data = data.replace("#ifndef SPECIAL_H\n", "")
        data = data.replace("#define SPECIAL_H\n", "")
        data = data.replace("#endif\n", "")
        ffi.cdef(data)

    return ffi


special = _make()
