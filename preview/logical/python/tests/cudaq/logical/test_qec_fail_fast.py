# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Missing QEC capabilities reject before analyzing or converting large roots."""

import importlib

import cudaq.logical as cql
import cudaq.mlir.ir as mlir_ir
import pytest

qec_lower = importlib.import_module("cudaq.logical.compiler.qec_lower")


def _device(*roots):
    builder = cql.devices.DeviceBuilder("FailFastDevice", source_module=None)
    compute = builder.logical.add_compute(capacity=1)
    builder.qec.bind(compute,
                     architecture=cql.devices.QECArchitecture(
                         "fail_fast",
                         cql.codes.Steane.default_encoding,
                         link_roots=roots))
    return builder.build()


def _unreachable(*args, **kwargs):
    pytest.fail("unsupported objective reached whole-program analysis/emission")


@pytest.mark.parametrize("gate", ("h", "s"))
@pytest.mark.parametrize("nested", (False, True))
def test_missing_objective_rejects_before_owner_analysis(
        monkeypatch, gate, nested):
    action = getattr(cql, gate)

    @cql.program
    def program() -> bool:
        q = cql.prepare_zero()
        if nested:
            q, = cql.ops.repeat(1000,
                                carries=(q,),
                                body=lambda i, value: (action(value),))
        else:
            q = action(q)
        # A supported suffix must never be inspected after the missing gate.
        for _ in range(20):
            q = cql.x(q)
        return cql.measure_z(q)

    device = _device()
    p1 = cql.compiler.place(cql.compile(program), device=device)
    monkeypatch.setattr(qec_lower._P1ToP2, "_index_value_bindings",
                        _unreachable)
    monkeypatch.setattr(qec_lower._P1ToP2, "_select_sites", _unreachable)
    objective_name = qec_lower._P1ToP2._objective_name

    def bounded_objective_name(self, attribute):
        name = objective_name(self, attribute)
        if name == "x":
            pytest.fail("preflight walked beyond the unsupported gate")
        return name

    monkeypatch.setattr(qec_lower._P1ToP2, "_objective_name",
                        bounded_objective_name)
    with pytest.raises(ValueError,
                       match=f"no feasible P2 implementation for {gate}"):
        cql.compile(p1, pipeline=cql.compiler.pipelines.qec(), device=device)


@pytest.mark.parametrize("family", ("action", "any"))
@pytest.mark.parametrize("named", (False, True))
def test_generated_lowering_remains_supported(family, named):
    code = cql.codes.Steane

    @cql.gadget(implements=cql.logical.h)
    def encoded_h(block: cql.patch[code]) -> cql.patch[code]:
        return cql.h(block.data)

    calls = []

    @cql.compiler.qec_lowering(objective_family=family,
                               objective=cql.logical.h if named else None,
                               codes=(code,))
    def lower(site, context):
        calls.append(site.objective)
        return encoded_h

    @cql.program
    def program() -> bool:
        return cql.measure_z(cql.h(cql.prepare_zero()))

    device = _device(lower, cql.gadgets.prepare_zero(code),
                     cql.gadgets.measure_z(code))
    p0 = cql.compile(program)
    _target_placer(device)._preflight(p0)
    p1 = cql.compiler.place(p0, device=device)
    p2 = cql.compile(p1, pipeline=cql.compiler.pipelines.qec(), device=device)
    assert p2.module.operation.verify()
    assert calls == ["h"]


def test_encoding_derived_fallback_remains_supported():

    @cql.program
    def program() -> bool:
        return cql.measure_z(cql.z(cql.x(cql.prepare_zero())))

    builder = cql.devices.DeviceBuilder("FallbackDevice", source_module=None)
    compute = builder.logical.add_compute(capacity=1)
    builder.qec.bind(compute, encoding=cql.codes.Steane)
    device = builder.build()
    p1 = cql.compiler.place(cql.compile(program), device=device)
    p2 = cql.compile(p1, pipeline=cql.compiler.pipelines.qec(), device=device)
    assert p2.module.operation.verify()


def test_incompatible_provider_rejects_during_selection(monkeypatch):
    code = cql.codes.Steane

    @cql.gadget(implements=cql.logical.h)
    def wrong_arity(
            a: cql.patch[code],
            b: cql.patch[code]) -> tuple[cql.patch[code], cql.patch[code]]:
        return cql.h(a.data), cql.h(b.data)

    @cql.program
    def program() -> bool:
        return cql.measure_z(cql.h(cql.prepare_zero()))

    device = _device(wrong_arity)
    p1 = cql.compiler.place(cql.compile(program), device=device)
    monkeypatch.setattr(qec_lower._P1ToP2, "_emit_protocol", _unreachable)
    with pytest.raises(ValueError, match="no feasible P2 implementation for h"):
        cql.compile(p1, pipeline=cql.compiler.pipelines.qec(), device=device)


def _target_placer(device):
    target = cql.targets.Target.from_device("fail_fast", device)
    backend = target.runtime_endpoint
    while not isinstance(backend, cql.targets.LogicalMachineBackend):
        backend = backend.next_backend
    return backend


@pytest.mark.parametrize("nested", (False, True))
def test_qec_target_rejects_before_placement_but_allows_logical_requests(
        monkeypatch, nested):

    @cql.program
    def program() -> bool:
        q = cql.prepare_zero()
        if nested:
            q, = cql.ops.repeat(1000,
                                carries=(q,),
                                body=lambda i, value: (cql.h(value),))
        else:
            q = cql.h(q)
        return cql.measure_z(q)

    device = _device()
    p0 = cql.compile(program)
    backend = _target_placer(device)
    # A direct request for P1 does not require a realizable P2 implementation.
    assert backend.compile(p0).profile == "p1"
    monkeypatch.setattr(backend, "compile", _unreachable)
    assert "LOGICAL" in backend._estimate(p0, tier="logical")
    with pytest.raises(ValueError, match="no feasible P2 implementation for h"):
        backend._estimate(p0, tier="static")
    monkeypatch.setattr(cql.targets, "_stage_estimate", _unreachable)
    with pytest.raises(ValueError, match="no feasible P2 implementation for h"):
        backend._estimate(p0)


def test_target_preflight_accepts_encoding_derived_paulis():

    @cql.program
    def program() -> bool:
        return cql.measure_z(cql.x(cql.prepare_zero()))

    device = _device(cql.gadgets.measure_z(cql.codes.Steane))
    p0 = cql.compile(program)
    assert _target_placer(device)._traced_compile(p0).profile == "p1"


def test_preflight_uses_verified_build_not_mutated_inspection_clone():

    @cql.program
    def program() -> bool:
        return cql.measure_z(cql.x(cql.prepare_zero()))

    p0 = cql.compile(program)
    inspected = p0.module
    root = next(
        op for op in inspected.body.operations if op.name == "qlx.program")
    action = next(op for op in root.regions[0].blocks[0].operations
                  if op.name == "qlx.apply")
    action.attributes["action"] = mlir_ir.Attribute.parse(
        "#qlx.action<h>", context=inspected.context)
    # The immutable build still contains X, which has a standard realization.
    _target_placer(_device())._preflight(p0)


@pytest.mark.parametrize("depth", (1, 2))
def test_preflight_rejects_missing_objective_inside_helper(depth):

    @cql.program
    def inner(q: cql.types.logical_qubit) -> cql.types.logical_qubit:
        return cql.h(q)

    @cql.program
    def outer(q: cql.types.logical_qubit) -> cql.types.logical_qubit:
        return inner(q)

    helper = inner if depth == 1 else outer

    @cql.program
    def program() -> bool:
        q = helper(helper(cql.prepare_zero()))
        return cql.measure_z(q)

    p0 = cql.compile(program)
    assert "qlx.call" in p0.to_mlir()
    backend = _target_placer(_device())
    with pytest.raises(ValueError, match="no feasible P2 implementation for h"):
        backend._preflight(p0)
