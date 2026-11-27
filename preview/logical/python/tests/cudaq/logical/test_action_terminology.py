# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Canonical gadget terminology preserves legacy APIs and serialized artifacts."""

import pytest

import cudaq.logical as ql
from cudaq.logical.programs.definition import ProgramDefinition


def test_action_and_objective_decorators_produce_the_same_saved_definition():

    def identity(q: ql.types.logical_qubit) -> ql.types.logical_qubit:
        return q

    canonical = ql.action(identity)
    legacy = ql.objective(identity)
    assert canonical.kind == legacy.kind == "action"
    canonical_build = ql.compile(canonical)
    legacy_build = ql.compile(legacy)
    assert canonical_build.serialize() == legacy_build.serialize()
    replayed = ql.compiler.Build.replay(legacy_build.serialize())
    assert replayed.root == canonical_build.root
    # Saved IR keeps its pre-migration spellings and remains verifiable.
    assert "qlx.objective_body" in str(replayed.module)
    assert replayed.module.operation.verify()


def test_unitary_kind_normalizes_the_legacy_definition_arguments():

    def identity(q: ql.types.logical_qubit) -> ql.types.logical_qubit:
        return q

    canonical = ql.action(identity, kind="unitary")
    legacy = ProgramDefinition(identity,
                               kind="objective",
                               objective_kind="action")
    assert legacy.kind == "action"
    assert canonical.action_kind == legacy.action_kind == "unitary"
    assert canonical.objective_kind == legacy.objective_kind == "action"
    assert ql.compile(canonical).serialize() == ql.compile(legacy).serialize()
    with pytest.raises(AttributeError, match="immutable"):
        canonical.action_kind = "instrument"
    with pytest.raises(TypeError, match="not both"):
        ProgramDefinition(identity,
                          kind="action",
                          action_kind="unitary",
                          objective_kind="action")


def test_action_includes_readout_but_an_explicit_unitary_kind_rejects_it():

    def readout(q: ql.types.logical_qubit) -> bool:
        return ql.measure_z(q)

    instrument = ql.action(readout, kind="instrument")
    assert ql.compile(instrument).root.kind == "instrument"
    with pytest.raises(TypeError, match="kind='unitary'"):
        ql.compile(ql.action(readout, kind="unitary"))


def test_operand_binding_aliases_retain_identity_and_ownership():
    owner = ql.logical.cx
    canonical = ql.types.ActionOperandRef(action=owner, name="control", index=0)
    legacy = ql.types.ObjectiveOperandRef(objective=owner,
                                          name="control",
                                          index=0)
    assert canonical == legacy == owner.operands.control
    assert len({canonical, legacy, owner.operands.control}) == 1
    assert canonical.action is legacy.objective is owner
    assert isinstance(owner.operands, ql.programs.ActionOperands)
    with pytest.raises(TypeError, match="not both"):
        ql.types.ActionOperandRef(action=owner,
                                  objective=ql.logical.cz,
                                  name="control",
                                  index=0)


def test_qec_lowering_aliases_preserve_manifest_identity():

    def provider(site, context):
        return None

    canonical = ql.qec.QECLowering(provider,
                                   action_family="action",
                                   action=ql.logical.cx,
                                   plugin="terminology-test",
                                   version="1")
    legacy = ql.qec.QECLowering(provider,
                                objective_family="action",
                                objective=ql.logical.cx,
                                plugin="terminology-test",
                                version="1")
    assert canonical.action is legacy.objective is ql.logical.cx
    assert canonical.action_family == legacy.objective_family == "action"
    assert canonical.manifest_sha256 == legacy.manifest_sha256
    with pytest.raises(TypeError, match="not both"):
        ql.qec.QECLowering(provider,
                           action_family="action",
                           objective_family="instrument",
                           plugin="terminology-test",
                           version="1")


def test_parameter_map_and_error_names_keep_their_compatibility_views():
    mapping = ql.gadgets.ParameterMap((("physical_angle", "logical_angle"),))
    assert mapping.realization_to_action == mapping.realization_to_objective
    assert mapping.action_to_realization == mapping.objective_to_realization
    assert ql.errors.ActionMismatch is ql.errors.ObjectiveMismatch
    assert ql.errors.MissingAction is ql.errors.MissingObjective


def test_canonical_dialect_aliases_keep_the_registered_operation_names():
    from cudaq.mlir.dialects import fabric, qlx

    assert qlx.ActionBodyOp.OPERATION_NAME == "qlx.objective_body"
    assert fabric.ActionOp.OPERATION_NAME == "fabric.objective"
    assert qlx.ActionBodyOp is qlx.ObjectiveBodyOp
    assert fabric.ActionOp is fabric.ObjectiveOp
