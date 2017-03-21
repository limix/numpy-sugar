from __future__ import absolute_import as _

import _cffi_backend

from pkg_resources import DistributionNotFound as _DistributionNotFound
from pkg_resources import get_distribution as _get_distribution

from . import epsilon, linalg, random, special
from .api import get_include, get_lib
from ._array import cartesian, is_all_equal, is_all_finite, is_crescent

try:
    __version__ = _get_distribution('numpy_sugar').version
except _DistributionNotFound:
    __version__ = 'unknown'


def test():
    import os
    p = __import__('numpy_sugar').__path__[0]
    src_path = os.path.abspath(p)
    old_path = os.getcwd()
    os.chdir(src_path)

    try:
        return_code = __import__('pytest').main(['-q'])
    finally:
        os.chdir(old_path)

    if return_code == 0:
        print("Congratulations. All tests have passed!")

    return return_code
