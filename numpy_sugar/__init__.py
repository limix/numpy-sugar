from __future__ import absolute_import as _
from __future__ import unicode_literals as _

from pkg_resources import get_distribution as _get_distribution
from pkg_resources import DistributionNotFound as _DistributionNotFound

from . import linalg
from . import _epsilon as epsilon
from . import special
from .array import (is_all_equal, is_crescent, is_all_finite)
from . import random

from .api import get_include
from .api import get_lib

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
