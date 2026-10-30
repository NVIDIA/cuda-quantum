# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Native declaration validation and compiled logical-operand identity."""

import json

import pytest

import cudaq.logical as ql
import cudaq.mlir.ir as ir
from cudaq.logical.compiler import CompilationContext


def _declaration(family):
    context = CompilationContext()
    qubit = ir.Type.parse("!qlx.logical_qubit", context=context.context)
    bit = ir.IntegerType.get_signless(1, context=context.context)
    symbol = context.declare_action(family=family,
                                    requested_symbol="named_inputs",
                                    kind="composite",
                                    inputs=(qubit, bit),
                                    results=(qubit,))
    return context, context.find_symbol(symbol)


@pytest.mark.parametrize("family", ["action", "instrument"])
@pytest.mark.parametrize("names, diagnostic", [
    ("[]", "cover every signature input"),
    ('["first"]', "cover every signature input"),
    ('["first", "second", "third"]', "cover every signature input"),
    ('["", "second"]', "entries must be nonempty"),
    ('["first", "first"]', "entries must be unique"),
    ('[0 : i64, "second"]', "operand_names"),
    ('"first"', "operand_names"),
])
def test_native_declaration_rejects_invalid_operand_names(
        family, names, diagnostic):
    context, declaration = _declaration(family)
    text = str(context.module).replace(
        f'operand_names = {declaration.attributes["operand_names"]}',
        f"operand_names = {names}")
    with pytest.raises(ir.MLIRError, match=diagnostic):
        ir.Module.parse(text, context=context.context)


@pytest.mark.parametrize("family", ["action", "instrument"])
@pytest.mark.parametrize("names", [None, '["quantum", "classical"]'])
def test_native_declaration_accepts_legacy_or_complete_operand_names(
        family, names):
    context, declaration = _declaration(family)
    if names is None:
        del declaration.attributes["operand_names"]
    else:
        declaration.attributes["operand_names"] = ir.Attribute.parse(
            names, context=context.context)
    assert context.module.operation.verify()


def test_standard_objective_uses_compiled_operand_names():
    from cudaq.mlir._mlir_libs import _qlxRuntime

    bare = ql.patch[ql.codes.BareQubit]

    @ql.gadget(implements=ql.logical.cx,
               logical_ports={
                   "control": "left.q0",
                   "target": "right.q0"
               })
    def cnot(left: bare, right: bare) -> tuple[bare, bare]:
        return ql.cx(left.data, right.data)

    build = ql.compile(ql.gadgets.GadgetContract(cnot))
    module = build._fresh_module()
    declaration = next(op for op in module.body.operations
                       if op.operation.name == "qlx.action")
    assert str(
        declaration.attributes["operand_names"]) == '["control", "target"]'
    assert 'ports = ["control", "target"]' in str(
        declaration.attributes["clifford_action"])
    declaration.attributes["operand_names"] = ir.Attribute.parse(
        '["lhs", "rhs"]', context=module.context)
    spec = next(op for op in module.body.operations
                if op.operation.name == "fabric.gadget_spec")
    spec.attributes["logical_ports"] = ir.Attribute.parse(
        '{lhs = "left.q0", rhs = "right.q0"}', context=module.context)
    result = json.loads(
        _qlxRuntime.verify_gadget_json(module, build.root.symbol,
                                       json.dumps({"source": "gadget_spec"})))
    assert result["logical_action"]["status"] == "verified"
