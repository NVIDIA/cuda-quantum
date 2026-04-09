# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


def _preload_dynamics_libs():
    import ctypes, os
    lib_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
    for lib in ('libcudaq-logger.so', 'libnvqir-dynamics.so'):
        try:
            ctypes.CDLL(os.path.join(lib_dir, lib), ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_dynamics_libs()
del _preload_dynamics_libs

from .helpers import InitialState
from .schedule import Schedule
