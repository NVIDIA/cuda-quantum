# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""qlx.dialects.event -- Event dialect Python module.

Re-exports the TableGen-generated Event op bindings (TestOp, PollOp, IsOp,
SelectReadyOp, TryTakeOp, CancelOp, AwaitOp, FenceOp, SelectionOp, YieldOp)
plus the native extension's `HandleType` (`!event.handle<payload,
ownership[, stream]>`). The dialect itself is auto-registered on every
Context via the ``_site_initialize_0`` hook.
"""

from __future__ import annotations

# Generated op bindings (declare_mlir_dialect_python_bindings).
from ._event_ops_gen import *  # noqa: F401,F403

# Native extension: HandleType lives in the "event" submodule of the
# combined _qlx_ext extension, mirroring qlx.dialects.atoms/fabric.
from .._mlir_libs._qlx_ext.event import HandleType  # noqa: F401
