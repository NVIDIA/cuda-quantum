# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import functools
import platform
import sys


@functools.cache
def mklq_targets_available():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False

    try:
        import cudaq
    except Exception:
        return False

    try:
        for target in ("mklq-cpu", "mklq-metal"):
            cudaq.reset_target()
            cudaq.set_target(target)
        return True
    except Exception:
        return False
    finally:
        try:
            cudaq.reset_target()
            cudaq.__clearKernelRegistries()
        except Exception:
            pass
