from os.path import join
from sysconfig import get_config_var


def _make():
    from cffi import FFI

    ffi = FFI()

    sources = [join('numpy_sugar', 'special', 'special.c')]
    hdr = join('numpy_sugar', 'include', 'numpy_sugar', 'special.h')
    incls = [
        join('numpy_sugar', 'include'),
        join(get_config_var('prefix'), 'include')
    ]

    from distutils.ccompiler import new_compiler
    compiler = new_compiler()
    libraries = ['hcephes']
    if 'msv' not in compiler.__class__.__name__.lower():
        libraries += ['m']

    ffi.set_source(
        'numpy_sugar.special._special_ffi',
        '#include "numpy_sugar/special.h"',
        include_dirs=incls,
        library_dirs=[join(get_config_var('prefix'), 'lib')],
        sources=sources,
        extra_compile_args=['-Ofast'],
        libraries=libraries)

    ffi.cdef(r"""
double nsugar_chi2_sf(int k, double x);
double nsugar_lgamma(double x);
double nsugar_normal_pdf(double x);
double nsugar_normal_cdf(double x);
double nsugar_normal_icdf(double x);
double nsugar_normal_sf(double x);
double nsugar_normal_isf(double x);
double nsugar_normal_logpdf(double x);
double nsugar_normal_logcdf(double x);
double nsugar_normal_logsf(double x);

double nsugar_beta_isf(double a, double b, double x);

double nsugar_logaddexp(double x, double y);
double nsugar_logaddexps(double x, double y, double sx, double sy);
double nsugar_logaddexpss(double x, double y, double sx, double sy,
                         double* sign);

double nsugar_logbinom(double N, double K);
""")

    return ffi


special = _make()
