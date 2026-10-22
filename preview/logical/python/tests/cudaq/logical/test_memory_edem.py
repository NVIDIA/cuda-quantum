# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compare the circuit and packed per-gadget EDEM paths on surface memory."""

from pathlib import Path
import runpy

import pytest

import cudaq.logical as cql
import cudaq_edem as edem

stim = pytest.importorskip("stim", minversion="1.16")
EXAMPLE = Path(
    __file__).resolve().parents[4] / "examples/standalone/07_memory_edem.py"


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(EXAMPLE))


def _laws(models):
    return tuple(
        (distribution.size,
         tuple((group.kind, tuple(group.fault_ids), tuple(group.probabilities))
               for group in distribution.groups))
        for model in models
        for distribution in model.faults)


@pytest.mark.parametrize("distance, rounds", [(3, 2), (3, 3), (3, 5), (5, 5)])
def test_memory_edem_matches_circuit_and_generated_dem(example, distance,
                                                       rounds):
    noise = cql.compiler.StandardDepolarizingNoise(.001, .003, .004)
    (memory, values, composed, projected, circuit, circuit_dem,
     edem_dem) = example["compare"](distance,
                                    rounds,
                                    noise,
                                    data_depolarization=.002)
    reference = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=.002,
        **{name: getattr(noise, name) for name in noise.__dataclass_fields__})
    expected = example["dem_errors"](reference.detector_error_model())
    assert example["dem_errors"](edem_dem) == pytest.approx(expected,
                                                            abs=1e-12,
                                                            rel=0)
    assert circuit.num_detectors == rounds * (distance**2 - 1)
    assert circuit.num_observables == 1
    assert len(values) == len(memory.profiles) == rounds
    assert len({id(model) for model in values}) == min(rounds, 3)
    if rounds > 3:
        assert values[1] is values[2]
    assert composed.inputs == composed.outputs == []
    assert isinstance(projected, edem.DEM)
    assert _laws([composed]) == _laws([projected]) == _laws(values)
    assert projected.to_bytes() == composed.to_bytes()
    loaded = edem.DEM.from_bytes(projected.to_bytes())
    reduced, _ = edem.coalesce_dem_columns(edem.factorize_dem(loaded))
    assert example["dem_errors"](example["to_stim_dem"](reduced)) == \
        pytest.approx(example["dem_errors"](circuit_dem), abs=1e-12, rel=0)


@pytest.mark.parametrize("parameter", [
    "data_depolarization",
    *cql.compiler.StandardDepolarizingNoise.__dataclass_fields__
])
def test_each_noise_component_agrees_after_composition(example, parameter):
    parameters = {} if parameter == "data_depolarization" else {parameter: .01}
    noise = cql.compiler.StandardDepolarizingNoise(**parameters)
    example["compare"](
        3,
        3,
        noise,
        data_depolarization=.01 if parameter == "data_depolarization" else 0)


def test_extra_data_faults_match_extraction_entries(example):
    memory = example["memory_recipe"](3, 3)
    placements = memory.depolarizations(.02)
    for index, profile in enumerate(memory.profiles):
        build, faults = example["memory_faults"](
            profile, cql.compiler.StandardDepolarizingNoise(),
            placements[index])
        projection = cql.compiler.project_clifford(build,
                                                   gadget=profile.gadget.name)
        entry = next(i for i, op in enumerate(projection.instructions)
                     if op.site_id == placements[index][0].before)
        after = projection.instructions[entry - 1].site_id if entry else ""
        assert len(faults.faults) == 9 * 3
        assert len(faults.exclusive_groups) == 9
        for q, group in zip(placements[index][0].carriers,
                            faults.exclusive_groups):
            alternatives = [faults.faults[i] for i in group]
            assert [f.insertions[0].factors for f in alternatives
                   ] == [((q, p),) for p in "xyz"]
            assert all(f.probability == .02 / 3 for f in alternatives)
            assert all(f.insertions[0].after == after for f in alternatives)


def test_noiseless_memory_dem_and_open_boundary_rejection(example):
    memory, values, composed, projected, _, _, dem = example["compare"](
        3, 3, cql.compiler.StandardDepolarizingNoise(), data_depolarization=0)
    assert composed.vector_sizes == (0, 25)
    assert example["dem_errors"](dem) == {}
    assert projected.vector_sizes == composed.vector_sizes
    partial = edem.compose_chain(values[1:])
    assert partial.inputs and not partial.outputs
    with pytest.raises(ValueError):
        edem.project_DEM(partial)
    with pytest.raises(ValueError):
        edem.DEM.from_bytes(values[0].to_bytes())
    with pytest.raises(TypeError, match="project_DEM"):
        example["to_stim_dem"](composed)


def _uniform_channel(responses, probability):
    count = len(responses)
    rows = max(responses).bit_length()
    entries = [(row, col)
               for col, response in enumerate(responses)
               for row in range(rows)
               if response >> row & 1]
    return edem.EDEM(faults=[
        edem.FaultDistribution(count, [
            edem.FaultGroup(edem.GroupKind.EXCLUSIVE, list(range(count)),
                            [probability / count] * count)
        ])
    ],
                     detectors=[rows],
                     blocks=[
                         edem.Block(edem.Endpoint(edem.PortKind.DETECTOR),
                                    edem.Endpoint(edem.PortKind.FAULT),
                                    edem.SparseMatrix(rows, count, entries))
                     ])


@pytest.mark.parametrize("count", [3, 15])
def test_depolarizing_dem_preserves_the_full_joint_distribution(example, count):
    probability = .3
    model = _uniform_channel(list(range(1, count + 1)), probability)
    reduced, _ = edem.coalesce_dem_columns(
        edem.factorize_dem(edem.project_DEM(model)))
    dem = example["to_stim_dem"](reduced)
    distribution = [1.] + [0.] * count
    for instruction in dem:
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        mask = sum(1 << target.val for target in instruction.targets_copy())
        distribution = [(1 - p) * distribution[i] + p * distribution[i ^ mask]
                        for i in range(count + 1)]
    assert distribution == pytest.approx([1 - probability] +
                                         [probability / count] * count,
                                         abs=1e-14,
                                         rel=0)


def test_dem_export_rejects_unsupported_exclusive_laws(example):
    for responses, probability in (([1, 2, 3], .9), ([1, 2, 4], .1)):
        model = edem.project_DEM(_uniform_channel(responses, probability))
        with pytest.raises(ValueError, match="unsupported fault distribution"):
            edem.factorize_dem(model)
        with pytest.raises(ValueError, match="factorize_dem"):
            example["to_stim_dem"](model)


def test_stim_formatter_preserves_duplicate_mechanisms_without_math(example):
    model = edem.project_DEM(
        edem.EDEM(faults=[edem.FaultDistribution.independent([.1, .2, .3])],
                  detectors=[1],
                  blocks=[
                      edem.Block(edem.Endpoint(edem.PortKind.DETECTOR),
                                 edem.Endpoint(edem.PortKind.FAULT),
                                 edem.SparseMatrix(1, 3, [(0, 0), (0, 1)]))
                  ]))
    formatted = example["to_stim_dem"](model)
    terms = [op for op in formatted if op.type == "error"]
    assert [op.args_copy()[0] for op in terms] == [.1, .2, .3]
    assert [len(op.targets_copy()) for op in terms] == [1, 1, 0]
    reduced, mapping = edem.coalesce_dem_columns(model)
    formatted = example["to_stim_dem"](reduced)
    assert mapping == [0, 0, None]
    terms = [op for op in formatted if op.type == "error"]
    assert len(terms) == 1
    assert terms[0].args_copy()[0] == pytest.approx(.26)


def test_memory_edem_example_writes_exchange_artifacts(example, tmp_path,
                                                       capsys):
    example["main"](["--output", str(tmp_path)])
    assert "219 matching DEM error terms" in capsys.readouterr().out
    circuit = stim.Circuit((tmp_path / "memory.stim").read_text())
    assert circuit.num_detectors == 24
    generated = stim.DetectorErrorModel((tmp_path / "composed.dem").read_text())
    direct = stim.DetectorErrorModel((tmp_path / "circuit.dem").read_text())
    assert example["dem_errors"](generated) == pytest.approx(
        example["dem_errors"](direct), abs=1e-12, rel=0)
    packed = edem.EDEM.from_bytes((tmp_path / "composed.edemb").read_bytes())
    assert packed.inputs == packed.outputs == []
    projected = edem.DEM.from_bytes((tmp_path / "projected.edemb").read_bytes())
    assert projected.to_bytes() == packed.to_bytes()
    assert _laws([projected]) == _laws([packed])
    assert len(list(tmp_path.glob("surface_d3_*.edemb"))) == 3


def test_native_projection_retains_successes_but_stim_adapter_rejects_them(
        example):
    model = edem.EDEM(successes=[1])
    projected = edem.project_DEM(model)
    assert projected.successes == [1]
    with pytest.raises(ValueError, match="postselection"):
        example["to_stim_dem"](projected)
