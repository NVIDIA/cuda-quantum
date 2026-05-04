# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Public API for the CUDA-Q span tracer.

The CUDA-Q process shares a single ``Tracer`` that routes begin / end
events to an installed backend. A backend placed here via ``set_backend``
captures events emitted from Python ``span(...)``, C++ ``ScopedTrace``
sites throughout the runtime, and the MLIR ``TracePassInstrumentation``
attached at every ``PassManager`` site.

Construct a backend, install it with ``set_backend``, emit spans, then
query the backend directly. ``ChromeBackend`` buffers events in memory
and writes `Chrome Trace Event Format
<https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU>`
JSON to a file on destruction when a path is given. ``SpdlogBackend``
routes events through the existing log. ``reset_backend()`` clears the
installed backend and disables capture.

``traced(name)`` is a decorator form of ``span``.
"""

from functools import wraps

from ..mlir._mlir_libs._quakeDialects.cudaq_runtime.trace import (
    span,
    TraceBackend,
    ChromeBackend,
    SpdlogBackend,
    set_backend,
    get_backend,
    reset_backend,
)


def traced(name=None):
    """Decorator that wraps a callable in a ``span``.

    Each invocation opens a span under the ``python`` category and emits
    a span event when the call returns. The wrapped function's name,
    documentation, and signature are preserved through ``functools.wraps``.

    If ``name`` is omitted, the span name is the fully qualified dotted
    path of the wrapped function (``fn.__module__`` joined with
    ``fn.__qualname__``).

        @trace.traced()                # inherits the function's path
        def compute(): ...

        @trace.traced("my_phase")      # explicit override
        def compute(): ...
    """

    def decorator(fn):
        effective = name or f"{fn.__module__}.{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            with span(effective):
                return fn(*args, **kwargs)

        return wrapper

    # Allow use as bare `@traced` (no parentheses).
    if callable(name):
        fn = name
        name = None
        return decorator(fn)
    return decorator


__all__ = [
    "span",
    "traced",
    "TraceBackend",
    "ChromeBackend",
    "SpdlogBackend",
    "set_backend",
    "get_backend",
    "reset_backend",
]
