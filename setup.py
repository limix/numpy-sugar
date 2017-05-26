import sys
from os import chdir, getcwd
from os.path import abspath, dirname, join

from setuptools import find_packages, setup

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (OSError, IOError, ImportError):
    long_description = open('README.md').read()


def get_capi_lib():
    from build_capi import CApiLib  # pylint: disable=E0401
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
    src_path = dirname(abspath(sys.argv[0]))
    old_path = getcwd()
    chdir(src_path)
    sys.path.insert(0, src_path)

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner'] if needs_pytest else []

    setup_requires = [
        'build-capi>=1.1.10',
        'ncephes>=1.0.22',
    ] + pytest_runner
    install_requires = ['ncephes>=1.0.26', 'numpy>=1.10', "scipy>=0.18.1",
                        'dask[complete]>=0.14']
    tests_require = ['pytest>=3']

    recommended = {
        "numba": ["numba>=0.30"],
    }

    metadata = dict(
        name='numpy-sugar',
        version='1.0.42',
        maintainer="Danilo Horta",
        maintainer_email="horta@ebi.ac.uk",
        license="MIT",
        description="Missing NumPy functionalities.",
        long_description=long_description,
        url='https://github.com/limix/numpy-sugar',
        packages=find_packages(),
        zip_safe=False,
        cffi_modules=['special_build.py:special'],
        install_requires=install_requires,
        tests_require=tests_require,
        setup_requires=setup_requires,
        extras_require=recommended,
        include_package_data=True,
        capi_libs=[get_capi_lib],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        package_data={'': [join('numpy_sugar', 'lib', '*.*')]})

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        chdir(old_path)


if __name__ == '__main__':
    setup_package()
