# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from .passes import Translation


def public_finalizer(finalize, name: str):
    """Attach one versioned public identity to a finalizer callable."""

    finalize.__logical_finalizer__ = f"cudaq.logical/{name}@0.3"
    return finalize


def emit_mlir():

    def finalize(module, ctx, **_):
        return str(module)

    return public_finalizer(finalize, "mlir-text")


def translate_text(translation: Translation, **translation_options):

    def finalize(module, ctx, **_):
        from cudaq.logical._native import native

        return native.translate(module, translation.value,
                                **translation_options)

    name = translation.value.replace("_", "-")
    return public_finalizer(finalize, f"translate-{name}")


def py_walker(fn, *, name=None):

    def finalize(module, ctx, **kwargs):
        return fn(module, **kwargs)

    if name is None:
        name = fn.__name__.strip("_").replace("_", "-")
    return public_finalizer(finalize, name)
