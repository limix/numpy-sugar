from os.path import join
from pycparser import parse_file
from pycparser.c_ast import NodeVisitor
from pycparser.c_ast import FuncDecl


class FuncSign(object):
    def __init__(self, name, ret_type):
        self.name = name
        self.ret_type = ret_type
        self.param_names = []
        self.param_types = []

    def __str__(self):
        n = len(self.param_names)
        pnames = self.param_names
        ptypes = self.param_types

        sparams = ('%s %s' % (ptypes[i], pnames[i]) for i in range(n))
        sparams = str(tuple(sparams))
        sparams = sparams.replace("'", "")
        s = '%s %s%s' % (self.ret_type, self.name, sparams)
        s = s.replace(',)', ')')
        return s


class FuncDefVisitor(NodeVisitor):
    def __init__(self):
        self.functions = []

    def _parse_param_signature(self, p):
        name = getattr(p.type, 'declname', '')

        suffix = ''
        t = p.type.type
        while not hasattr(t, 'names'):
            t = t.type
            suffix += '*'
        typ = t.names[0]
        return (name, typ + suffix)

    def visit_Decl(self, node):
        if isinstance(node.type, FuncDecl):
            ret_type = node.type.type.type.names[0]
            fs = FuncSign(node.name, ret_type)
            for p in node.type.args.params:
                (name, typ) = self._parse_param_signature(p)
                fs.param_names.append(name)
                fs.param_types.append(typ)
            self.functions.append(fs)


def fetch_func_decl(filename):
    ast = parse_file(filename, use_cpp=True, cpp_path='cpp', cpp_args='')

    v = FuncDefVisitor()
    v.visit(ast)

    return v.functions


if __name__ == '__main__':

    fdecls = fetch_func_decl(
        join('numpy_sugar', 'include', 'numpy_sugar', 'special.h'))

    data = """
from . import _special_ffi
from numba import cffi_support as _cffi_support
from numba import void
from numba import float64
from numba import vectorize
from numba import jit
import numpy as np
_cffi_support.register_module(_special_ffi)

"""

    for f in fdecls:
        name = f.name.replace('nsugar_', '')
        data += "_%s = _special_ffi.lib.%s\n" % (name, f.name)

    data += "\n"

    for f in fdecls:
        if f.name.startswith('nsugar_beta'):
            data += """
@jit([float64(float64, float64, float64),
      float64[:](float64, float64, float64[:])])
def {fname}(a, b, x):
    if np.isscalar(x):
        return _{fname}(a, b, x)
    r = np.empty(x.size)
    for i in range(x.size):
        r[i] = _{fname}(a, b, x[i])
    return r
""".format(fname=f.name.replace('nsugar_', ''))
        elif f.name.startswith('nsugar_logbinom'):
            data += """
@vectorize([float64(float64, float64)], nopython=True)
def {fname}(a, b):
    return _{fname}(a, b)
""".format(fname=f.name.replace('nsugar_', ''))
        elif f.name.startswith('nsugar_normal'):
            data += """
@vectorize([float64(float64)], nopython=True)
def {fname}(x):
    return _{fname}(x)
""".format(fname=f.name.replace('nsugar_', ''))

    with open(join('numpy_sugar', 'special', 'special.py'), 'w') as f:
        f.write(data)

    with open(join('numpy_sugar', 'special', 'special.py'), 'a') as f:
        f.write("""
@vectorize([float64(float64, float64)], nopython=True)
def logaddexp(x, y):
    return _logaddexp(x, y)

@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def logaddexps(x, y, sx, sy):
    return _logaddexps(x, y, sx, sy)

def logaddexpss(x, y, sx, sy, r, sign):
    \"\"\"Logarithm of the sum of exponentiations of the inputs with sign.

    Suppose you are interested in computing::

        sx[i]*exp(x[i]) + sy[i]*exp(y[i])

    where ``sx[i]`` and ``sy[i]`` are either -1 or +1. Often a direct
    calculation of the above is numerically innacurate. Instead, let::

        sign[i]*exp(r[i]) = sx[i]*exp(x[i]) + sy[i]*exp(y[i])

    This function accurately computes ``r[i]`` and ``sign[i]``, where
    ``sign[i]`` is either -1 or +1.\"\"\"
    ptr = _special_ffi.ffi.cast("double*", sign.ctypes.data)
    for i in range(len(x)):
        r[i] = _logaddexpss(x[i], y[i], sx[i], sy[i], ptr + i)
""")
