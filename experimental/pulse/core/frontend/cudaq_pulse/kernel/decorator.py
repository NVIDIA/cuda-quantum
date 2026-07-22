# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import Any, Callable

_next_vid = 0


def _alloc_vid() -> int:
    global _next_vid
    vid = _next_vid
    _next_vid += 1
    return vid


class QuditRef:
    """A single qudit reference representing one quantum degree of freedom.

    Created via ``cudaq_pulse.qudit_ref()``.  Passed as an argument to
    ``@kernel`` functions to bind physical qubit resources.
    """

    __slots__ = ("_vid",)

    def __init__(self) -> None:
        self._vid = _alloc_vid()


class QvecRef:
    """A fixed-size vector of qudit references.

    Created via ``cudaq_pulse.qvec_ref(n)``.  Supports ``len()`` and
    integer indexing to access individual ``QuditRef`` elements.
    """

    def __init__(self, n: int):
        self._n = n
        self._qudits = [QuditRef() for _ in range(n)]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> QuditRef:
        if not 0 <= idx < self._n:
            raise IndexError(f"qudit index {idx} out of range [0, {self._n})")
        return self._qudits[idx]


def qudit_ref() -> QuditRef:
    """Allocate a single qudit reference.

    Use as an argument when calling or compiling a ``@kernel`` function::

        q = cudaq_pulse.qudit_ref()
        ck = cudaq_pulse.compile(my_kernel, [q], ...)

    Returns:
        A fresh ``QuditRef`` with a unique virtual ID.
    """
    return QuditRef()


def qvec_ref(n: int) -> QvecRef:
    """Allocate a vector of *n* qudit references.

    Example::

        qubits = cudaq_pulse.qvec_ref(4)
        ck = cudaq_pulse.compile(my_kernel, [qubits[0], qubits[1]], ...)

    Args:
        n: Number of qudits.

    Returns:
        A ``QvecRef`` containing *n* ``QuditRef`` elements.
    """
    return QvecRef(n)


def kernel(fn: Callable) -> Callable:
    """Decorator that marks a Python function as a pulse kernel.

    The decorated function is traced via bytecode capture when called,
    producing an intermediate representation that can be compiled to
    MLIR with ``cudaq_pulse.compile()``.

    Supported control flow inside the kernel:

    - ``for i in range(N)`` (compile-time bound)
    - ``if`` / ``else`` with compile-time conditions

    Example::

        @cudaq_pulse.kernel
        def my_kernel(q):
            d, t = get_drive_line(q)
            drive(d, gaussian(40, 0.3, 10.0), t)

    Args:
        fn: A plain Python function using ``cudaq_pulse`` ops.

    Returns:
        A wrapped callable that performs bytecode tracing on each call.
    """
    import sys
    _major, _minor = sys.version_info[:2]
    if _major != 3 or _minor < 9:
        raise RuntimeError(
            f"@cudaq_pulse.kernel requires Python >= 3.9, got {_major}.{_minor}"
        )

    _cache_key: list[Any] = [None]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (fn.__code__, fn.__module__)
        if _cache_key[0] != key or wrapper.__cudaq_pulse_emitter__ is None:
            from .bytecode_bridge import compile_kernel_bytecode
            wrapper.__cudaq_pulse_emitter__ = compile_kernel_bytecode(fn)
            _cache_key[0] = key
        return wrapper.__cudaq_pulse_emitter__(*args, **kwargs)

    def _trace_with_builder(builder, args):
        """Re-trace the kernel using a custom IR builder (e.g. MLIRIRBuilder)."""
        from .bytecode_bridge import _trace_kernel_with_builder
        _trace_kernel_with_builder(fn, builder, args)

    wrapper.__wrapped__ = fn
    wrapper.__cudaq_pulse_emitter__ = None
    wrapper._trace_with_builder = _trace_with_builder
    return wrapper
