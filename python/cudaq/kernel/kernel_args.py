# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
"""
Typed argument wrappers for CUDA-Q kernel arguments.

Each wrapper represents a Python value that can be passed to a kernel and
corresponds to a supported MLIR type. These types will be marshaled into
OpaqueArguments by the C++ type caster in `python/utils/OpaqueArgumentsCaster.h`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(slots=True)
class Float64Arg:
    """A float value destined for an MLIR Float64Type parameter."""
    value: float


@dataclass(slots=True)
class Float32Arg:
    """A float value destined for an MLIR Float32Type parameter."""
    value: float


@dataclass(slots=True)
class IntArg:
    """An integer value destined for an MLIR IntegerType (width != 1)."""
    value: int


@dataclass(slots=True)
class BoolArg:
    """A boolean value destined for an MLIR IntegerType(1)."""
    value: bool


@dataclass(slots=True)
class ComplexF64Arg:
    """A complex value destined for an MLIR ComplexType(f64)."""
    value: complex


@dataclass(slots=True)
class ComplexF32Arg:
    """A complex value destined for an MLIR ComplexType(f32)."""
    value: complex


@dataclass(slots=True)
class PauliWordArg:
    """A cudaq.pauli_word destined for an MLIR cc::CharspanType."""
    value: Any


@dataclass(slots=True)
class StateArg:
    """A cudaq.State destined for an MLIR cc::PointerType(StateType)."""
    value: Any


@dataclass(slots=True)
class StructArg:
    """A structured value destined for an MLIR cc::StructType.

    Carries the opaque MLIR type and module handles so the C++ type caster
    can compute the target layout.
    """
    value: Any
    mlir_type: Any  # MlirType (opaque handle)
    module: Any  # MlirModule (opaque handle)


@dataclass(slots=True)
class VecArg:
    """A list value destined for an MLIR cc::StdvecType.

    Carries the opaque MLIR type so the C++ type caster can determine the
    element type for nested vector handling.
    """
    value: list
    mlir_type: Any  # MlirType (opaque handle)


@dataclass(slots=True)
class DecoratorCapture:
    """A captured kernel decorator (Python-defined kernel passed as argument).

    Attributes:
        decorator: The PyKernelDecorator instance.
        resolved: The resolved captured arguments (list of ArgWrapper values).
    """
    decorator: Any
    resolved: list

    def __str__(self) -> str:
        return self.decorator.name + " -> " + str(self.resolved)

    def __repr__(self) -> str:
        return "name: " + self.decorator.name + ", resolved: " + str(
            self.resolved)


@dataclass(slots=True)
class LinkedKernelCapture:
    """A captured C++ kernel linked into the current module.

    Attributes:
        linkedKernel: The mangled name of the linked C++ kernel.
        qkeModule: The parsed MLIR module containing the kernel's Quake code.
    """
    linkedKernel: str
    qkeModule: Any  # MlirModule

    def __str__(self) -> str:
        return self.linkedKernel

    def __repr__(self) -> str:
        return "name: " + self.linkedKernel


KernelArg = Union[Float64Arg, Float32Arg, IntArg, BoolArg, ComplexF64Arg,
                  ComplexF32Arg, PauliWordArg, StateArg, StructArg, VecArg,
                  DecoratorCapture, LinkedKernelCapture]


def wrap_for_mlir_type(arg: Any,
                       arg_type: Any,
                       module: Any = None) -> KernelArg:
    """Wrap a Python value in the appropriate typed wrapper based on the
    expected MLIR type.

    Args:
        arg: The Python value to wrap.
        arg_type: The expected MLIR type (from the kernel signature).
        module: The MLIR module (needed for StructArg).
    """
    from cudaq.mlir.ir import ComplexType, F32Type, F64Type, IntegerType
    from cudaq.mlir.dialects import cc

    if F64Type.isinstance(arg_type):
        return Float64Arg(float(arg))
    if F32Type.isinstance(arg_type):
        return Float32Arg(float(arg))
    if IntegerType.isinstance(arg_type):
        if IntegerType(arg_type).width == 1:
            return BoolArg(bool(arg))
        return IntArg(int(arg))
    if ComplexType.isinstance(arg_type):
        element_type = ComplexType(arg_type).element_type
        if F64Type.isinstance(element_type):
            return ComplexF64Arg(complex(arg))
        return ComplexF32Arg(complex(arg))
    if cc.CharspanType.isinstance(arg_type):
        return PauliWordArg(arg)
    if cc.PointerType.isinstance(arg_type):
        element_type = cc.PointerType.getElementType(arg_type)
        if cc.StateType.isinstance(element_type):
            return StateArg(arg)
    if cc.StructType.isinstance(arg_type):
        return StructArg(arg, arg_type, module)
    if cc.StdvecType.isinstance(arg_type):
        return VecArg(arg if isinstance(arg, list) else list(arg), arg_type)
    if cc.CallableType.isinstance(arg_type):
        raise RuntimeError(
            f"Unexpected callable type for non-decorator argument: {arg}")
    raise RuntimeError(f"Unsupported MLIR type for argument: {arg_type}")
