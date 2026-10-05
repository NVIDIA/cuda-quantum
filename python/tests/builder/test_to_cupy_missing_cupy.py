# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Regression test for `cudaq.to_cupy` when CuPy is not installed. The helper
used to only `print` on the `import cupy` failure and then fall through,
dereferencing the never-imported module and raising a confusing
`NameError: ... 'cp' ...` that masked the real cause. It must instead raise a
clear error explaining that CuPy is required.

This test forces the `cupy` import to fail regardless of whether CuPy is
actually installed, so it is independent of the environment / GPU.
"""

import builtins
import os

import pytest

import cudaq


def test_to_cupy_without_cupy_raises_clear_error(monkeypatch):
    real_import = builtins.__import__

    def no_cupy_import(name, *args, **kwargs):
        if name == 'cupy' or name.startswith('cupy.'):
            raise ImportError("No module named 'cupy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_cupy_import)

    with pytest.raises(RuntimeError, match="CuPy"):
        # Argument is irrelevant: the import guard fires before the state or
        # target is ever touched.
        cudaq.to_cupy(object())


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
