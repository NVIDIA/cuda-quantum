# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Surface memory against an independently authored Stim circuit."""

import builtins
from dataclasses import replace

import pytest

import cudaq.logical as cql
from cudaq.logical.architecture.geometry import rotated_surface_lattice
from cudaq.logical.architectures import surface


@pytest.fixture(scope="module")
def stim():
    return pytest.importorskip("stim", minversion="1.16")


@pytest.mark.parametrize("distance, rounds", [(3, 2), (3, 3), (3, 5), (5, 5),
                                              (7, 3)])
@pytest.mark.parametrize("probability", [0.0, 0.002])
def test_surface_memory_matches_reference(make_memory, stim,
                                          surface_memory_reference, distance,
                                          rounds, probability):
    memory = make_memory(distance, rounds)
    circuit = stim.Circuit(memory.to_stim(data_depolarization=probability))
    assert circuit == surface_memory_reference(stim, distance, rounds,
                                               probability)
    assert circuit.num_qubits == distance**2
    assert circuit.num_detectors == (distance**2 -
                                     1) // 2 + (rounds - 1) * (distance**2 - 1)
    assert circuit.num_observables == 1


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("preparation, deterministic_family", [
    ("prepare_zero", "z_check"),
    ("prepare_plus", "x_check"),
])
def test_builtin_surface_preparations_project_and_validate(
        distance, preparation, deterministic_family):
    definitions = surface.definitions(distance)
    code = definitions.code
    gadget = getattr(definitions, preparation)
    projection = cql.compiler.project_clifford(cql.compile(gadget),
                                               gadget=gadget.name)
    assert projection.instructions[0].kind == "R"
    assert len(projection.instructions[0].carriers) == code.n
    assert sum(
        op.kind == "MPP" for op in projection.instructions) == (len(code.hx) +
                                                                len(code.hz))
    assert any(op.kind == "H" for op in projection.instructions) == (
        preparation == "prepare_plus")

    records = {
        (family, i): gadget.record(f"{family}{i}.outcome")
        for family, checks in (("x_check", code.hx), ("z_check", code.hz))
        for i in range(len(checks))
    }
    graph = cql.gadgets.GadgetContractBuilder(gadget.interface)
    for (family, _), record in records.items():
        if family == deterministic_family:
            graph.detector(record)
    graph.bind(gadget.interface.outputs.only().syndrome,
               tuple(records.values()))
    # Derivation proves the ideal detector parities, output syndrome bindings,
    # and completeness, including the random check outcomes after preparation.
    model = cql.analysis.derive_edem(cql.compile(graph.freeze()),
                                     faults=cql.analysis.FaultSet())
    assert len(model.inputs) == 0
    assert len(model.outputs) == 1


def test_surface_memory_emits_text_without_stim(run_memory_example, monkeypatch,
                                                capsys):
    original_import = builtins.__import__

    def without_stim(name, *args, **kwargs):
        if name == "stim" or name.startswith("stim."):
            raise ModuleNotFoundError("Stim is unavailable", name="stim")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_stim)
    result = run_memory_example("04_stim_emission.py")
    implementation = result["implementation"]
    assert result["prep"].gadget is implementation.prepare_zero
    assert result["sec"].gadget is implementation.memory_round
    assert result["readout"].gadget is implementation.measure_z
    assert "MPP " in result["text"]
    assert capsys.readouterr().out == result["text"]


def test_surface_memory_samples_detectors(run_memory_example, stim):
    result = run_memory_example("04_stim_emission.py", shots=100, seed=7)
    assert result["detectors"].shape == (100, 20)
    assert result["observables"].shape == (100, 1)


def test_mpp_noise_must_be_explicit(make_memory):
    memory = make_memory()
    with pytest.raises(ValueError,
                       match="MPP requires an explicit fault model"):
        memory.to_stim(
            cql.compiler.StandardDepolarizingNoise(
                before_measure_flip_probability=.01))
    placements = memory.depolarizations(.02)
    assert len(memory.profiles) - 1 not in placements
    for i, profile in enumerate(memory.profiles[:-1]):
        projection = cql.compiler.project_clifford(cql.compile(profile),
                                                   gadget=profile.gadget.name)
        entry = next(op for op in projection.instructions if op.kind == "MPP")
        assert placements[i][0].before == entry.site_id
        assert len(placements[i][0].carriers) == 9


def test_stim_export_validates_explicit_noise_sites_and_closed_chain(
        make_memory):
    memory = make_memory()
    # Native MPPs are not Clifford gates: no gate depolarization is invented.
    assert "DEPOLARIZE" not in cql.compiler.emit_stim(
        memory.profiles,
        noise=cql.compiler.StandardDepolarizingNoise(
            after_clifford_depolarization=.01))
    with pytest.raises(ValueError, match="closed single-block chain"):
        cql.compiler.emit_stim(memory.profiles[1:])
    with pytest.raises(ValueError, match="projected instruction"):
        cql.compiler.emit_stim(memory.profiles,
                               depolarizations={
                                   0: (cql.compiler.Depolarize1Placement(
                                       "missing", (0,), .01),)
                               })
    for index in (-1, len(memory.profiles), True):
        with pytest.raises(ValueError, match="occurrence index"):
            cql.compiler.emit_stim(memory.profiles, depolarizations={index: ()})
    site, _ = memory.extraction_entries[0]
    with pytest.raises(ValueError, match="carrier is out of range"):
        cql.compiler.emit_stim(memory.profiles,
                               depolarizations={
                                   0: (cql.compiler.Depolarize1Placement(
                                       site, (cql.compiler.project_clifford(
                                           cql.compile(memory.profiles[0]),
                                           gadget=memory.profiles[0].gadget.name
                                       ).interface.carrier_count,), .01),)
                               })
    with pytest.raises(TypeError, match="Depolarize1Placement"):
        cql.compiler.emit_stim(memory.profiles,
                               depolarizations={0: ("missing",)})


def test_stim_export_rejects_missing_virtual_bindings_and_incompatible_presentations(
        make_memory):
    memory = make_memory()
    prep, sec, *_, readout = memory.profiles
    unbound = replace(sec,
                      boundary=None,
                      output_syndromes=(),
                      boundary_complete=False)
    with pytest.raises(ValueError, match="every output virtual syndrome"):
        cql.compiler.emit_stim((prep, unbound, readout))
    with pytest.raises(ValueError, match="same code presentation"):
        cql.compiler.emit_stim((prep, make_memory(5, 3).profiles[1], readout))


def test_stim_export_preserves_signed_measurements_and_affine_constants():
    prepare = cql.gadgets.prepare_zero(cql.codes.BareQubit)

    @cql.gadget(implements=cql.logical.measure_z)
    def negative_readout(block: cql.patch[cql.codes.BareQubit]) -> bool:
        block, outcome = cql.ops.measure_pauli(block.data,
                                               paulis="-Z",
                                               record="negative")
        cql.discard(block)
        return cql.ops.parity(outcome)

    record = negative_readout.record("negative.outcome")
    graph = cql.gadgets.GadgetContractBuilder(negative_readout.interface)
    graph.detector(cql.gadgets.ProfileParity(records=(record,), constant=True))
    profiles = (cql.gadgets.GadgetContract(prepare), graph.freeze())
    text = cql.compiler.emit_stim(profiles)
    # The synthetic constant bit carries the affine offset without changing
    # physical gates. The signed MPP produces one in the ideal realization.
    assert "MPP !Z0\n" in text
    assert "MPAD 1\n" in text
    assert "DETECTOR rec[-2] rec[-1]\n" in text
    assert "OBSERVABLE_INCLUDE(0)" in text
    with pytest.raises(ValueError,
                       match="MPP requires an explicit fault model"):
        cql.compiler.emit_stim(profiles,
                               noise=cql.compiler.StandardDepolarizingNoise(
                                   before_measure_flip_probability=.01))
    selected = replace(profiles[-1],
                       success=(cql.gadgets.SuccessPredicate(record),))
    with pytest.raises(ValueError, match="postselection"):
        cql.compiler.emit_stim((profiles[0], selected))


@pytest.mark.parametrize(
    "probability",
    [-1, 1.1, float("nan"), float("inf"), True])
def test_standard_depolarizing_noise_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match=r"finite in \[0, 1\]"):
        cql.compiler.StandardDepolarizingNoise(
            after_clifford_depolarization=probability)
    with pytest.raises(ValueError, match=r"finite in \[0, 1\]"):
        cql.compiler.Depolarize1Placement("site", (0,), probability)


@pytest.mark.parametrize("carriers", [(), (0, 0), (-1,), (True,), ("0",)])
def test_explicit_depolarization_rejects_invalid_carriers(carriers):
    with pytest.raises(ValueError, match="distinct nonnegative integers"):
        cql.compiler.Depolarize1Placement("site", carriers, .01)


@pytest.mark.parametrize("distance, error", [
    (True, TypeError),
    (3.0, TypeError),
    ("3", TypeError),
    (None, TypeError),
    (1, ValueError),
    (2, ValueError),
    (4, ValueError),
])
def test_surface_definitions_reject_invalid_distance(distance, error):
    with pytest.raises(error, match="odd Python int at least 3"):
        surface.definitions(distance)
