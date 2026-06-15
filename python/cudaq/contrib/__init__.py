# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .qiskit_convert import from_qiskit, from_qasm
from .propagator import propagator

_LAZY_ATTRS = {
    'amplitude_encode': '.encoding',
    'angular_encode': '.encoding',
}


def __getattr__(name):
    import importlib

    if name in _LAZY_ATTRS:
        mod = importlib.import_module(_LAZY_ATTRS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val

    raise AttributeError(f"module 'cudaq.contrib' has no attribute {name!r}")


def __dir__():
    names = list(globals().keys())
    names.extend(_LAZY_ATTRS.keys())
    return sorted(set(names))
