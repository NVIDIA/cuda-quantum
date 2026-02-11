# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
"""
Type stubs for quantum types used only within CUDA-Q kernels. These types are
never executed at Python runtime; kernel code is parsed into MLIR and
JIT-executed. Kernel types raise a KernelTypeError if instantiated outside a
kernel.
"""

from __future__ import annotations

from typing import Iterator, overload

_KERNEL_ONLY_ERROR_MESSAGE = "Type '{}' can only be used within a CUDA-Q kernel."


class KernelTypeError(RuntimeError):
    """An error raised when a kernel type is used outside a kernel."""

    def __init__(self, cls: type):
        super().__init__(_KERNEL_ONLY_ERROR_MESSAGE.format(cls.__name__))


class KernelType:
    """A type that can exclusively be used within a CUDA-Q kernel."""

    # These types are never to be instantiated within the Python host: they
    # are always parsed and JIT-executed using the MLIR compiler.
    def __new__(cls, *args, **kwargs):
        raise KernelTypeError(cls)


class qubit(KernelType):
    """
    The qubit is the primary unit of information in a quantum computer.
    Qubits can be created individually or as part of larger registers.
    """

    def __init__(self) -> None:
        ...

    def __invert__(self) -> qubit:
        """Negate the control qubit."""
        ...

    def is_negated(self) -> bool:
        """Returns true if this is a negated control qubit."""
        ...

    def reset_negation(self) -> None:
        """Removes the negated state of a control qubit."""
        ...

    def id(self) -> int:
        """Return a unique integer identifier for this qubit."""
        ...


class qview(KernelType):
    """A non-owning view on a register of qubits."""

    def size(self) -> int:
        """Return the number of qubits in this view."""
        ...

    @overload
    def front(self) -> qubit:
        ...

    @overload
    def front(self, count: int) -> qview:
        ...

    def front(self, _count: int | None = None) -> qubit | qview:
        """Return first qubit(s) in this view."""
        ...

    @overload
    def back(self) -> qubit:
        ...

    @overload
    def back(self, count: int) -> qview:
        ...

    def back(self, _count: int | None = None) -> qubit | qview:
        """Return the last qubit(s) in this view."""
        ...

    def __iter__(self) -> Iterator[qubit]:
        ...

    def slice(self, _start: int, _count: int) -> qview:
        """Return the [start, start+count] qudits as a non-owning `qview`."""
        ...

    def __getitem__(self, idx: int) -> qubit:
        """Return the qubit at the given index."""
        ...


class qvector(KernelType):
    """
    An owning, dynamically sized container for qubits. The semantics of the
    `qvector` follows that of a `std::vector` or list for qubits.
    """

    def __init__(self, size: int) -> None:
        ...

    def size(self) -> int:
        """Return the number of qubits in this `qvector`."""
        ...

    @overload
    def front(self) -> qubit:
        ...

    @overload
    def front(self, count: int) -> qview:
        ...

    def front(self, count: int | None = None) -> qubit | qview:
        """Return first qubit(s) in this `qvector`."""
        ...

    @overload
    def back(self) -> qubit:
        ...

    @overload
    def back(self, count: int) -> qview:
        ...

    def back(self, count: int | None = None) -> qubit | qview:
        """Return the last qubit(s) in this `qvector`."""
        ...

    def __iter__(self) -> Iterator[qubit]:
        ...

    def slice(self, start: int, count: int) -> qview:
        """Return the [start, start+count] qudits as a non-owning `qview`."""
        ...

    def __getitem__(self, idx: int) -> qubit:
        """Return the qubit at the given index."""
        ...
