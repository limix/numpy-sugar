import numpy as np

build_err_msg = np.testing._private.utils.build_err_msg


def _build_err_msg(*args, **kwargs):
    if len(args) < 6:
        kwargs["precision"] = 16
    return build_err_msg(*args, **kwargs)


def pytest_runtest_call(item):
    np.testing._private.utils.build_err_msg = _build_err_msg


def pytest_runtest_teardown(item, nextitem):
    np.testing._private.utils.build_err_msg = build_err_msg
