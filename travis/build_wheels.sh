#!/bin/bash
set -e -x

yum install -y atlas-devel libffi libffi-devel

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ $PYBIN == *"p26"* ]] || [[ $PYBIN == *"p33"* ]] || \
       [[ $PYBIN == *"p34"* ]]; then
        continue
    fi
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/numpy_sugar*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ $PYBIN == *"p26"* ]] || [[ $PYBIN == *"p33"* ]] \
        || [[ $PYBIN == *"p34"* ]]; then
        continue
    fi
    "${PYBIN}/pip" install numpy_sugar -f /io/wheelhouse
    "${PYBIN}/pip" install pytest numpy
    cd "$HOME"
    "${PYBIN}/python" -c "import sys; import numpy_sugar; sys.exit(numpy_sugar.test())"
done
