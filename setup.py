import os
from os.path import join
import sys
from setuptools import setup
from setuptools import find_packages

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

def get_capi_lib():
    from build_capi import CApiLib
    from ncephes import get_include

    sources = [join('numpy_sugar', 'special', 'special.c')]
    hdrs = [join('numpy_sugar', 'special', 'special.h')]
    incls = [join('numpy_sugar', 'include'), get_include()]

    return CApiLib(
        name='numpy_sugar.lib.numpy_sugar',
        sources=sources,
        include_dirs=incls,
        depends=hdrs)


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner'] if needs_pytest else []

    setup_requires = [
        'build-capi>=1.0', 'ncephes>=1.0.14', 'cffi>=1.7', 'numba>=0.28'
    ] + pytest_runner
    install_requires = [
        'ncephes>=1.0.14', 'scipy>=0.18.1', 'numpy>=1.11', 'numba>=0.28',
        'cffi>=1.7'
    ]
    tests_require = ['pytest']

    metadata = dict(
        name='numpy-sugar',
        version='1.0.10',
        maintainer="Danilo Horta",
        maintainer_email="horta@ebi.ac.uk",
        license="MIT",
        description="Missing NumPy functionalities.",
        long_description=long_description,
        url='http://github.com/glimix/numpy-sugar',
        packages=find_packages(),
        zip_safe=False,
        cffi_modules=['special_build.py:special'],
        install_requires=install_requires,
        tests_require=tests_require,
        setup_requires=setup_requires,
        include_package_data=True,
        capi_libs=[get_capi_lib],
        package_data={'': [join('numpy_sugar', 'lib', '*.*')]}
    )

    try:
        from distutils.command.bdist_conda import CondaDistribution
    except ImportError:
        pass
    else:
        metadata['distclass'] = CondaDistribution
        metadata['conda_buildnum'] = 0
        metadata['conda_features'] = ['mkl']

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)


if __name__ == '__main__':
    setup_package()
