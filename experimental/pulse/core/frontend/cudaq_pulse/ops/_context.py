# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CompilationContext:
    """State carried through a single kernel compilation."""

    module: Any
    builder: Any
    symbol_table: Any
    function: Any
    extra: dict = field(default_factory=dict)


_tls = threading.local()


def get_active_context() -> Optional[CompilationContext]:
    return getattr(_tls, "active_context", None)


def set_active_context(ctx: Optional[CompilationContext]) -> None:
    _tls.active_context = ctx
