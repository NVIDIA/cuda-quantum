# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""cudaq.mlir.dialects.qlx -- CUDA-Q Logical IR dialect module.

Re-exports the TableGen-generated op and enum bindings, plus the
native extension's attribute classes.  This is the single
user-facing module for the QLX dialect; mirrors MLIR upstream's
`mlir.dialects.arith` layout where the dialect module lives directly
under `<package>.dialects.<name>` with no intermediate namespace.

Usage::

    from cudaq.mlir.dialects import qlx as qlx_dialect

    attr = qlx_dialect.PauliAttr.get("Z", ctx)
"""

from __future__ import annotations

# Generated op + enum bindings (declare_mlir_dialect_python_bindings).
from ._qlx_ops_gen import *  # noqa: F401,F403
from ._qlx_enum_gen import *  # noqa: F401,F403

# Native extension: attribute subclasses live in the `qlx` submodule
# of the combined _qlx_ext extension.
from .._mlir_libs._qlx_ext.qlx import (  # noqa: F401
    PauliAttr, set_inherent_attr,
)
