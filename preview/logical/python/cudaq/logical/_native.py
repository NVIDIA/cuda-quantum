# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Backward-compatibility shim exposing the CUDA-Q Logical native extensions under
single namespace objects.

The previous monolithic ``_qlxDialects`` extension exposed a single
``qlx`` (and ``fabric``) submodule.  After registering into CUDA-Q's
``cudaq.mlir`` bindings the native code lives in three extensions:

  - :mod:`cudaq.mlir._mlir_libs._qlx_ext`   -- two submodules ``qlx`` and
        ``fabric`` exposing the dialects' types, attributes, and helpers
        (dialect registration is handled automatically at import via the
        ``_site_initialize_1`` hook -- there is no explicit
        ``register_dialect`` / ``ensure_dialects`` entry point here)
  - :mod:`cudaq.mlir._mlir_libs._qlxRuntime` -- verify /
        lower / translate / run_pass

This module recombines them into the historical ``_native`` /
``_fabric_native`` shapes so existing consumers only have to change their
import line: ``from cudaq.mlir._mlir_libs._qlxDialects import qlx as _native``
becomes ``from cudaq.logical._native import native as _native``, and likewise
``fabric`` becomes ``from cudaq.logical._native import fabric_native``.
"""

from __future__ import annotations

from cudaq.mlir._mlir_libs import _qlx_ext  # noqa: F401
from cudaq.mlir._mlir_libs import _qlxRuntime  # noqa: F401


class _NativeFacade:
    """Combined facade over _qlx_ext.qlx and _qlxRuntime."""

    set_inherent_attr = staticmethod(_qlx_ext.qlx.set_inherent_attr)

    # Product CUDA-Q Logical attribute classes.
    PauliAttr = _qlx_ext.qlx.PauliAttr

    # Text helpers (from _qlxRuntime).
    clone_module = staticmethod(_qlxRuntime.clone_module)
    verify_clifford_t_module = staticmethod(
        _qlxRuntime.verify_clifford_t_module)
    lower_to_pbc_module = staticmethod(_qlxRuntime.lower_to_pbc_module)
    verify_pbc_module = staticmethod(_qlxRuntime.verify_pbc_module)
    clone_module_capsule = staticmethod(_qlxRuntime.clone_module_capsule)
    replace_module_contents_capsule = staticmethod(
        _qlxRuntime.replace_module_contents_capsule)
    run_pass = staticmethod(_qlxRuntime.run_pass)
    run_pass_capsule = staticmethod(_qlxRuntime.run_pass_capsule)
    translate = staticmethod(_qlxRuntime.translate)
    has_quake_import: bool = _qlxRuntime.has_quake_import

    @staticmethod
    def load_plugin(path: str) -> None:
        """Load a dialect/pass plugin without initializing the high-level surface."""
        from cudaq.mlir._mlir_libs import get_dialect_registry

        if path in _qlxRuntime.get_plugin_paths():
            return
        _qlxRuntime.load_plugin(path)
        _qlxRuntime.apply_dialect_plugin(path, get_dialect_registry())


# Singleton instances usable as namespace objects.
native = _NativeFacade()
fabric_native = _qlx_ext.fabric
