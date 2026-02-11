# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, pytest
import ast, inspect, textwrap

import cudaq
from cudaq.kernel.kernel_signature import KernelSignature
from cudaq.kernel.utils import getMLIRContext, mlirTypeFromPyType


class MlirTypes:
    INT = mlirTypeFromPyType(int, getMLIRContext())
    FLOAT = mlirTypeFromPyType(float, getMLIRContext())
    BOOL = mlirTypeFromPyType(bool, getMLIRContext())
    VEC_FLOAT = mlirTypeFromPyType(list[float], getMLIRContext())
    VEC_INT = mlirTypeFromPyType(list[int], getMLIRContext())


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


# -- test cases -----------------------------------------------------------


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
    assert sig.arg_types == [MlirTypes.INT]
    assert sig.return_type is None


def test_single_float_arg():

    def my_kernel(theta: float):
        q = cudaq.qubit()
        ry(theta, q)

    sig = _parse_and_assert_equal(my_kernel)
    assert sig.arg_types == [MlirTypes.FLOAT]
    assert sig.return_type is None


def test_multiple_mixed_args():

    def my_kernel(n: int, theta: float):
        q = cudaq.qvector(n)
        ry(theta, q[0])

    sig = _parse_and_assert_equal(my_kernel)
    assert sig.arg_types == [MlirTypes.INT, MlirTypes.FLOAT]
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
    assert sig.arg_types == [MlirTypes.INT]
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

    # Both should have the same argument types
    assert len(sig_with_none.arg_types) == len(sig_no_annotation.arg_types) == 1


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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
