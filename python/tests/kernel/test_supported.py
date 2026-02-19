# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.kernel.ast_bridge import KernelSignature
import os, sys, pytest
import ast, inspect
import cudaq


def test_unprocessed_ast():

    node_visitors = inspect.getmembers(
        sys.modules['ast'],
        lambda v: inspect.isclass(v) and issubclass(v, ast.NodeVisitor))
    dummy_bridge = cudaq.PyASTBridge(
        KernelSignature(arg_types=[], return_type=None))

    unsupported_nodes = set()
    for _, cls in node_visitors:
        fcts = inspect.getmembers(
            cls,
            lambda v: inspect.isroutine(v) and v.__name__.startswith("visit_"))
        for fct_name, _ in fcts:
            node_name = fct_name[6:]
            try:
                if node_name == 'Ellipsis':
                    # ast.Ellipsis is deprecated in Python 3.14
                    # https://docs.python.org/3/whatsnew/3.14.html#removed
                    cls = ast.Constant
                    node = ast.Constant(value=...)
                else:
                    cls = getattr(ast, node_name)
                    node = cls.__new__(cls)
            except:
                print(f"skipping test for {fct_name}")
                continue
            if not hasattr(dummy_bridge, f"visit_{type(node).__name__}"):
                unsupported_nodes.add(node)

    print("Not currently implemented:")
    for node in unsupported_nodes:
        node_name = type(node).__name__
        print(f"visit_{node_name}")
        with pytest.raises(RuntimeError) as e:
            dummy_bridge.visit(node)
        assert f"CUDA-Q does not currently support {node_name} expressions" in str(
            e.value)

    # We will never override all nodes, and this is to check
    # that the test indeed tests something.
    # The number 5 here is more or less arbitrary; it just
    # so happens that I see at least that many nodes that make
    # no sense for CUDA-Q to override.
    assert len(unsupported_nodes) > 5


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
