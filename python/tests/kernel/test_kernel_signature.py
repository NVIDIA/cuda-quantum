# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, pytest
import ast, inspect, textwrap
from dataclasses import dataclass

import numpy as np

import cudaq
from cudaq.kernel.kernel_signature import KernelSignature
from cudaq.kernel.utils import getMLIRContext, mlirTypeFromPyType
from cudaq.mlir.dialects import cc, quake

_ctx = getMLIRContext()


class MlirTypes:
    """Expected MLIR types for assertions, grouped by category."""

    # -- scalars (native Python) --
    INT64 = mlirTypeFromPyType(int, _ctx)
    FLOAT64 = mlirTypeFromPyType(float, _ctx)
    BOOL = mlirTypeFromPyType(bool, _ctx)
    COMPLEX = mlirTypeFromPyType(complex, _ctx)

    # -- scalars (distinct bit-widths) --
    INT32 = mlirTypeFromPyType(np.int32, _ctx)
    FLOAT32 = mlirTypeFromPyType(np.float32, _ctx)
    COMPLEX64 = mlirTypeFromPyType(np.complex64, _ctx)

    # -- flat containers --
    VEC_INT = mlirTypeFromPyType(list[int], _ctx)
    VEC_FLOAT = mlirTypeFromPyType(list[float], _ctx)
    VEC_BOOL = mlirTypeFromPyType(list[bool], _ctx)
    VEC_COMPLEX = mlirTypeFromPyType(list[complex], _ctx)

    # -- nested containers --
    VEC_VEC_FLOAT = mlirTypeFromPyType(list[list[float]], _ctx)
    VEC_VEC_INT = mlirTypeFromPyType(list[list[int]], _ctx)

    # -- tuples (aggregate types) --
    TUPLE_INT_FLOAT = mlirTypeFromPyType(tuple[int, float], _ctx)
    TUPLE_BOOL_INT_FLOAT = mlirTypeFromPyType(tuple[bool, int, float], _ctx)


def _get_ast_from_callable(fn) -> ast.Module:
    """Build an AST module from a callable's source code."""
    src = textwrap.dedent(inspect.getsource(fn))
    return ast.parse(src)


def _parse_and_assert_equal(fn):
    """
    Parse the given callable via both AST and MLIR routes,
    assert that the resulting signatures are identical,
    and return the signature for further assertions.
    """
    # Parse via AST
    sig_a = KernelSignature.parse_from_ast(_get_ast_from_callable(fn),
                                           fn.__name__)
    # replace None with empty list for comparision with MLIR
    sig_a.captured_args = []

    # Parse via MLIR
    decorated = cudaq.kernel(fn)
    sig_m = KernelSignature.parse_from_mlir(decorated.qkeModule,
                                            decorated.uniqName)

    assert sig_a == sig_m

    return sig_m


def _make_kernel_ast(arg_types=None, return_type=None) -> ast.Module:
    """
    Build an ``ast.Module`` for a stub kernel with the given type annotations.

    ``arg_types`` is a list of annotation strings (e.g. ``["int", "float"]``).
    ``return_type`` is an annotation string (e.g. ``"bool"``) or ``None``.
    """
    params = ", ".join(f"a{i}: {t}" for i, t in enumerate(arg_types or []))
    ret = f" -> {return_type}" if return_type else ""
    src = textwrap.dedent(f"""\
        def kernel({params}){ret}:
            pass
    """)
    return ast.parse(src)


# -- test cases -------------------------------------------------------------


def test_no_args_no_return():

    def my_kernel():
        q = cudaq.qubit()
        h(q)

    sig = _parse_and_assert_equal(my_kernel)
    assert len(sig.arg_types) == 0
    assert sig.return_type is None


def test_single_int_arg():

    def my_kernel(n: int):
        q = cudaq.qvector(n)
        h(q)

    sig = _parse_and_assert_equal(my_kernel)
    assert sig.arg_types == [MlirTypes.INT64]
    assert sig.return_type is None


def test_single_float_arg():

    def my_kernel(theta: float):
        q = cudaq.qubit()
        ry(theta, q)

    sig = _parse_and_assert_equal(my_kernel)
    assert sig.arg_types == [MlirTypes.FLOAT64]
    assert sig.return_type is None


def test_multiple_mixed_args():

    def my_kernel(n: int, theta: float):
        q = cudaq.qvector(n)
        ry(theta, q[0])

    sig = _parse_and_assert_equal(my_kernel)
    assert sig.arg_types == [MlirTypes.INT64, MlirTypes.FLOAT64]
    assert sig.return_type is None


def test_bool_return():

    def my_kernel() -> bool:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    sig = _parse_and_assert_equal(my_kernel)
    assert sig.arg_types == []
    assert sig.return_type == MlirTypes.BOOL


def test_int_arg_with_bool_return():

    def my_kernel(n: int) -> bool:
        q = cudaq.qvector(n)
        h(q[0])
        return mz(q[0])

    sig = _parse_and_assert_equal(my_kernel)
    assert sig.arg_types == [MlirTypes.INT64]
    assert sig.return_type == MlirTypes.BOOL


def test_list_int_arg():

    def my_kernel(angles: list[float]) -> list[int]:
        return [1]

    sig = _parse_and_assert_equal(my_kernel)
    assert sig.arg_types == [MlirTypes.VEC_FLOAT]
    assert sig.return_type == MlirTypes.VEC_INT


def test_none_return_annotation():
    """
    `-> None` should be treated the same as no return annotation.
    """

    def my_kernel_with_none(x: int) -> None:
        q = cudaq.qvector(x)
        h(q)

    def my_kernel_no_annotation(x: int):
        q = cudaq.qvector(x)
        h(q)

    sig_with_none = _parse_and_assert_equal(my_kernel_with_none)
    sig_no_annotation = _parse_and_assert_equal(my_kernel_no_annotation)

    # Both should have no return type
    assert sig_with_none.return_type is None
    assert sig_no_annotation.return_type is None

    # Both should have the same signature
    assert sig_with_none == sig_no_annotation


def test_return_statement_without_annotation_raises_error():
    """
    If a return statement exists but no return type is annotated,
    an error should be raised during compilation.
    """

    def my_kernel(x: int):
        q = cudaq.qvector(x)
        h(q[0])
        return mz(q[0])  # return statement without return type annotation

    # This should raise an error during pre_compile
    with pytest.raises(RuntimeError, match="return type annotation"):
        cudaq.kernel(my_kernel)


def test_unannotated_arg_raises_error():
    """
    If function arguments lack type annotations, an error should be raised
    during compilation.
    """

    def my_kernel(x: int, y):  # missing type annotation
        pass

    # This should raise an error during pre_compile
    with pytest.raises(RuntimeError,
                       match="must have argument type annotations"):
        cudaq.kernel(my_kernel)


# testing a larger variety of argument types
_SCALAR_ARG_CASES = [
    ("complex", MlirTypes.COMPLEX),
    ("np.complex64", MlirTypes.COMPLEX64),
    ("np.complex128", MlirTypes.COMPLEX),
    ("np.int32", MlirTypes.INT32),
    ("np.int64", MlirTypes.INT64),
    ("np.float32", MlirTypes.FLOAT32),
    ("np.float64", MlirTypes.FLOAT64),
]


@pytest.mark.parametrize("annotation, expected", _SCALAR_ARG_CASES)
def test_scalar_arg_types(annotation, expected):
    sig = KernelSignature.parse_from_ast(
        _make_kernel_ast(arg_types=[annotation]), "kernel")
    assert sig.arg_types == [expected]
    assert sig.return_type is None


# testing container types
_CONTAINER_ARG_CASES = [
    ("list[bool]", MlirTypes.VEC_BOOL),
    ("list[complex]", MlirTypes.VEC_COMPLEX),
    ("list[list[float]]", MlirTypes.VEC_VEC_FLOAT),
    ("list[list[int]]", MlirTypes.VEC_VEC_INT),
    ("np.ndarray", MlirTypes.VEC_FLOAT),  # ndarray defaults to vec<f64>
]


@pytest.mark.parametrize("annotation, expected", _CONTAINER_ARG_CASES)
def test_container_arg_types(annotation, expected):
    sig = KernelSignature.parse_from_ast(
        _make_kernel_ast(arg_types=[annotation]), "kernel")
    assert sig.arg_types == [expected]
    assert sig.return_type is None


# testing return types
_RETURN_TYPE_CASES = [
    # scalars
    ("int", MlirTypes.INT64),
    ("float", MlirTypes.FLOAT64),
    ("bool", MlirTypes.BOOL),
    ("complex", MlirTypes.COMPLEX),
    ("np.int32", MlirTypes.INT32),
    ("np.float32", MlirTypes.FLOAT32),
    ("np.complex64", MlirTypes.COMPLEX64),
    # containers
    ("list[float]", MlirTypes.VEC_FLOAT),
    ("list[int]", MlirTypes.VEC_INT),
    ("list[complex]", MlirTypes.VEC_COMPLEX),
    # tuples (aggregate return types)
    ("tuple[int, float]", MlirTypes.TUPLE_INT_FLOAT),
    ("tuple[bool, int, float]", MlirTypes.TUPLE_BOOL_INT_FLOAT),
]


@pytest.mark.parametrize("annotation, expected", _RETURN_TYPE_CASES)
def test_return_types(annotation, expected):
    sig = KernelSignature.parse_from_ast(
        _make_kernel_ast(return_type=annotation), "kernel")
    assert sig.arg_types == []
    assert sig.return_type == expected


# testing dataclass types


def test_dataclass_arg():

    @dataclass(slots=True)
    class MyPair:
        x: int
        y: float

    @cudaq.kernel
    def my_kernel(p: MyPair):
        pass

    arg_types = my_kernel.arg_types()
    assert len(arg_types) == 1
    assert cc.StructType.isinstance(arg_types[0])


def test_dataclass_return():

    @dataclass(slots=True)
    class MyResult:
        flag: bool
        count: int

    @cudaq.kernel
    def my_kernel() -> MyResult:
        return MyResult(True, 1)

    return_type = my_kernel.return_type
    assert cc.StructType.isinstance(return_type)


def test_dataclass_arg_and_return():

    @dataclass(slots=True)
    class Pair:
        a: int
        b: float

    @cudaq.kernel
    def my_kernel(p: Pair) -> Pair:
        return p

    arg_types = my_kernel.arg_types()
    return_type = my_kernel.return_type

    assert len(arg_types) == 1
    assert cc.StructType.isinstance(arg_types[0])
    assert cc.StructType.isinstance(return_type)
    assert arg_types[0] == return_type


# mixed-argument tests


def test_multiple_complex_args():
    """Multiple arguments of different complex/container types."""
    sig = KernelSignature.parse_from_ast(
        _make_kernel_ast(arg_types=['int', 'list[float]', 'complex']), "kernel")
    assert sig.arg_types == [
        MlirTypes.INT64, MlirTypes.VEC_FLOAT, MlirTypes.COMPLEX
    ]
    assert sig.return_type is None


def test_nested_list_arg_with_scalar_return():
    sig = KernelSignature.parse_from_ast(
        _make_kernel_ast(arg_types=['list[list[float]]'], return_type='int'),
        "kernel")
    assert sig.arg_types == [MlirTypes.VEC_VEC_FLOAT]
    assert sig.return_type == MlirTypes.INT64


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
