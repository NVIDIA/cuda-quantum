# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
from dataclasses import dataclass
import inspect
from typing import Callable, Literal, Optional, Type, overload

import cudaq.mlir.ir as mlir
from cudaq.mlir.dialects import cc

from .analysis import FunctionDefVisitor
from .utils import (getMLIRContext, mlirTypeFromAnnotation, nvqppPrefix,
                    recover_func_op)


class KernelSignatureError(RuntimeError):
    """
    An error raised when a kernel signature is invalid.
    """
    pass


@dataclass
class CapturedVariable:
    """A captured variable, given by a name and a type."""
    name: str
    type: mlir.Type


@dataclass
class CapturedLinkedKernel:
    """A captured C++ linked kernel."""
    kernel_name: str
    type: mlir.Type


@dataclass(eq=True)
class KernelSignature:
    """
    A `KernelSignature` represents the signature of a CUDA-Q kernel.
    """
    # The types of the input arguments to the kernel.
    arg_types: list[mlir.Type]
    # The type of the return value from the kernel, if any.
    return_type: Optional[mlir.Type]
    # The name and types of the captured arguments to the kernel.
    #
    # Will be `None` if the kernel has not been compiled.
    captured_args: Optional[list[CapturedVariable |
                                 CapturedLinkedKernel]] = None

    @staticmethod
    def parse_from_mlir(mlir_module: mlir.Module,
                        kernel_name: str,
                        compiled: bool = True) -> "KernelSignature":
        """
        Parse the signature of a CUDA-Q kernel from an MLIR module.
        
        If `compiled` is `True`, the kernel has been compiled and the captured
        arguments list is assumed to be empty.
        """
        funcOp = recover_func_op(mlir_module, nvqppPrefix + kernel_name)
        fnTy = mlir.FunctionType(
            mlir.TypeAttr(funcOp.attributes['function_type']).value)
        if len(fnTy.results) > 1:
            raise KernelSignatureError(
                "Multiple return values in MLIR kernel are not supported")
        captured_args = [] if compiled else None
        return KernelSignature(
            arg_types=fnTy.inputs,
            return_type=fnTy.results[0] if fnTy.results else None,
            captured_args=captured_args,
        )

    @staticmethod
    def parse_from_ast(ast_module: ast.Module,
                       kernel_name: str) -> "KernelSignature":
        """Parse the signature of a CUDA-Q kernel from a Python AST."""
        visitor = FunctionDefVisitor(kernel_name)
        visitor.visit(ast_module)

        if visitor.return_annotation is None and visitor.has_return_statement:
            raise KernelSignatureError(
                f"CUDA-Q kernel {kernel_name} has return statement but no return type annotation"
            )

        arg_types = [
            _mlir_type_from_annotation(annotation)
            for _name, annotation in visitor.arg_annotations
        ]
        return_type = _mlir_type_from_annotation(visitor.return_annotation,
                                                 acceptNoneType=False)
        return KernelSignature(arg_types=arg_types, return_type=return_type)

    def add_variable_capture(self, name: str, capture_type: mlir.Type):
        """
        Add a captured variable to the signature.
        """
        self.captured_args.append(CapturedVariable(name, capture_type))

    def add_linked_kernel_capture(self, kernel_name: str,
                                  callable_type: mlir.Type):
        """
        Add a captured linked kernel to the signature.
        """
        if kernel_name in self.captured_linked_kernels():
            # don't add the same linked kernel twice
            return
        self.captured_args.append(
            CapturedLinkedKernel(kernel_name, callable_type))

    def captured_linked_kernels(self) -> list[str]:
        """The names of the captured linked kernels."""
        if self.captured_args is None:
            raise KernelSignatureError("Kernel has not been compiled")
        return [
            arg.kernel_name
            for arg in self.captured_args
            if isinstance(arg, CapturedLinkedKernel)
        ]

    def captured_variable_names(self) -> list[str]:
        """The names of all captured variables."""
        if self.captured_args is None:
            raise KernelSignatureError("Kernel has not been compiled")
        return [
            arg.name
            for arg in self.captured_args
            if isinstance(arg, CapturedVariable)
        ]

    def captured_variables(self) -> list[CapturedVariable]:
        """The captured variables."""
        if self.captured_args is None:
            raise KernelSignatureError("Kernel has not been compiled")
        return [
            arg for arg in self.captured_args
            if isinstance(arg, CapturedVariable)
        ]

    def get_callable_type(self) -> cc.CallableType:
        """
        The signature of the kernel as a `cc.CallableType`.

        This is the kernel signature without the lifted captured arguments.
        """
        inputs = self.arg_types
        outputs = [self.return_type] if self.return_type is not None else []
        return cc.CallableType.get(getMLIRContext(), inputs, outputs)

    def get_lifted_type(self) -> mlir.FunctionType:
        """
        The signature of the kernel, including the list of captured arguments as
        inputs.
        """
        inputs = self.arg_types + self.captured_types()
        outputs = [self.return_type] if self.return_type is not None else []
        return mlir.FunctionType.get(inputs, outputs, getMLIRContext())

    def get_all_types(self) -> list[mlir.Type]:
        """
        The signature of the kernel as a concatenated list of all types.
        """
        ret_list = [self.return_type] if self.return_type is not None else []
        return self.arg_types + self.captured_types() + ret_list

    def captured_types(self) -> list[mlir.Type]:
        if self.captured_args is None:
            raise KernelSignatureError("Kernel has not been compiled")
        return [arg.type for arg in self.captured_args]


@overload
def _mlir_type_from_annotation(
        annotation, acceptNoneType: Literal[True] = ...) -> mlir.Type:
    ...


@overload
def _mlir_type_from_annotation(
        annotation, acceptNoneType: Literal[False]) -> Optional[mlir.Type]:
    ...


def _mlir_type_from_annotation(annotation,
                               acceptNoneType: bool = True
                              ) -> Optional[mlir.Type]:
    is_none_type = isinstance(annotation,
                              ast.Constant) and annotation.value is None
    if annotation is None or (not acceptNoneType and is_none_type):
        return None
    return mlirTypeFromAnnotation(annotation, getMLIRContext(), raiseError=True)
