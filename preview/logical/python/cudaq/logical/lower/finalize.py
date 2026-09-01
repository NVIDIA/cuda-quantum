# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Small, typed finalizers used by CUDA-Q Logical target recipes."""


def public_finalizer(finalize, name: str):
    finalize.__logical_finalizer__ = f"cudaq.logical/{name}@0.3"
    return finalize


def emit_mlir():

    def finalize(module, ctx, **_):
        return str(module)

    return public_finalizer(finalize, "mlir-text")


def py_walker(fn, *, name=None):

    def finalize(module, ctx, **kwargs):
        return fn(module, **kwargs)

    if name is None:
        name = fn.__name__.strip("_").replace("_", "-")
    return public_finalizer(finalize, name)


__all__ = ["emit_mlir", "public_finalizer", "py_walker"]
