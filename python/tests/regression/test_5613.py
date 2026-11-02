# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
import json
from typing import Callable
import pytest


def test_issue_5613_quake_json_default_and_commitment():
    """Test JSON serialization and commitment for kernel and callable arguments."""
    try:
        from cudaq.logical.compiler.quake import _json_default, _commitment
    except ImportError:
        pytest.skip("cudaq.logical is not available in the current environment")

    class FakeKernel:
        def __init__(self, name):
            self.name = name

    class FakeDecoratorCapture:
        def __init__(self, decorator):
            self.decorator = decorator

    kernel_obj = FakeKernel("my_kernel")
    capture_obj = FakeDecoratorCapture(kernel_obj)

    assert _json_default(kernel_obj) == "@cudaq.kernel:my_kernel"
    assert _json_default(capture_obj) == "@cudaq.kernel:my_kernel"

    def ordinary_callable():
        pass

    assert _json_default(ordinary_callable) == "callable:ordinary_callable"

    arguments = (kernel_obj, capture_obj, ordinary_callable, 42)
    commitment = _commitment(arguments)
    assert isinstance(commitment, str)
    assert commitment.startswith("sha256:")

    serialized = json.dumps(arguments, default=_json_default, sort_keys=True, separators=(",", ":"))
    assert '@cudaq.kernel:my_kernel' in serialized
    assert 'callable:ordinary_callable' in serialized


def test_callable_ast_annotation_inspection():
    """Test AST parsing of Callable and typing.Callable annotations."""
    code_subscript_callable = "def kernel(fn: Callable[[cudaq.qview], None]): pass"
    code_subscript_typing = "def kernel(fn: typing.Callable[[cudaq.qview], None]): pass"
    code_bare_callable = "def kernel(fn: Callable): pass"
    code_bare_typing = "def kernel(fn: typing.Callable): pass"

    ann_subscript_callable = ast.parse(code_subscript_callable).body[0].args.args[0].annotation
    ann_subscript_typing = ast.parse(code_subscript_typing).body[0].args.args[0].annotation
    ann_bare_callable = ast.parse(code_bare_callable).body[0].args.args[0].annotation
    ann_bare_typing = ast.parse(code_bare_typing).body[0].args.args[0].annotation

    assert isinstance(ann_subscript_callable, ast.Subscript)
    assert ann_subscript_callable.value.id == "Callable"

    assert isinstance(ann_subscript_typing, ast.Subscript)
    assert ann_subscript_typing.value.attr == "Callable"

    assert isinstance(ann_bare_callable, ast.Name)
    assert ann_bare_callable.id == "Callable"

    assert isinstance(ann_bare_typing, ast.Attribute)
    assert ann_bare_typing.attr == "Callable"


def test_issue_5613_callable_kernel_argument_import_cudaq():
    """Regression test for issue #5613: Callable kernel arguments in import_cudaq."""
    cudaq = pytest.importorskip("cudaq")
    try:
        import cudaq.logical as cql
    except ImportError:
        pytest.skip("cudaq.logical is not available in the current environment")

    @cudaq.kernel
    def layer(q: cudaq.qview):
        h(q[0])

    @cudaq.kernel
    def circuit(operation: Callable[[cudaq.qview], None]):
        q = cudaq.qvector(1)
        operation(q)
        mz(q)

    source_p0 = cql.compiler.import_cudaq(circuit, layer)
    assert source_p0 is not None
    p0_text = source_p0.to_mlir()
    assert "#qlx.action<h>" in p0_text
    assert "qlx.measure <Z>" in p0_text


def test_pykerneldecorator_decorator_and_resolved_properties():
    """Test .decorator and .resolved property compatibility on PyKernelDecorator."""
    cudaq = pytest.importorskip("cudaq")
    if not hasattr(cudaq, "kernel"):
        pytest.skip("cudaq.kernel is not available in the current environment")

    @cudaq.kernel
    def dummy_kernel(q: cudaq.qubit):
        h(q)

    assert hasattr(dummy_kernel, "decorator")
    assert dummy_kernel.decorator is dummy_kernel
    assert hasattr(dummy_kernel, "resolved")
    assert isinstance(dummy_kernel.resolved, list)
