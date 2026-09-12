# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""qlx.dialects.cflow -- Cflow dialect Python module.

Re-exports the TableGen-generated Cflow op bindings (RepeatOp, IfOp,
WhileOp, WhileConditionOp, YieldOp). The dialect defines no custom types
or attributes, so unlike ``qlx``/``fabric``/``atoms`` there is no native
extension type/attr submodule to import here. The dialect itself is
auto-registered on every Context via the ``_site_initialize_0`` hook.
"""

from __future__ import annotations

# Generated op bindings (declare_mlir_dialect_python_bindings).
from ._cflow_ops_gen import *  # noqa: F401,F403
