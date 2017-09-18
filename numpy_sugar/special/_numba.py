import pdb

from . import _special_ffi

try:
    from numba import cffi_support as _cffi_support
    from numba import (float64, int64)
    from numba import vectorize
    from numba import jit
    _cffi_support.register_module(_special_ffi)
    HAS_NUMBA = True
except ImportError:

    def vectorize(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    class Float64(object):
        def __call__(self, *_):
            pass

        def __getitem__(self, _):
            return Float64()

    class Int64(object):
        def __call__(self, *_):
            pass

        def __getitem__(self, _):
            return Int64()

    HAS_NUMBA = False

    float64 = Float64()
    int64 = Int64()
