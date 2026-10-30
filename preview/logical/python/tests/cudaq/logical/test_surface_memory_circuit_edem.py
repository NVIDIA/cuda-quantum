# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Circuit-level memory against Stim's independent rotated-memory generator."""

from collections import Counter
from functools import lru_cache
from itertools import product
from types import SimpleNamespace

import pytest

import cudaq.logical as cql
from cudaq.logical.architectures import surface
import cudaq_edem as edem

stim = pytest.importorskip("stim", minversion="1.16")


@pytest.fixture(scope="module")
def make_circuit_memory():

    @lru_cache(maxsize=None)
    def make(distance=3, rounds=3):
        recipe = surface.interleaved_nz_definitions(distance)
        prep, sec, readout = map(
            surface.gadget_contract,
            (recipe.prepare_zero, recipe.syndrome_extraction,
             recipe.measure_z_with_syndrome))
        contracts = (prep, *((sec,) * (rounds - 2)), readout)
        projections = {
            contract.gadget:
                cql.compiler.project_clifford(cql.compile(contract),
                                              gadget=contract.gadget.name)
            for contract in (prep, sec, readout)
        }
        coordinates = "\n".join(f"QUBIT_COORDS({x}, {y}) {q}"
                                for q, (x, y) in enumerate(recipe.coordinates))

        def to_stim(noise=None, data_depolarization=0):
            return coordinates + "\n" + cql.compiler.emit_stim(
                contracts,
                noise=noise,
                depolarizations=depolarizations(data_depolarization))

        def depolarizations(probability):
            return {
                index:
                    surface.data_depol(contract, projections[contract.gadget],
                                       probability)
                for index, contract in enumerate(contracts)
            }

        return SimpleNamespace(recipe=recipe,
                               contracts=contracts,
                               depolarizations=depolarizations,
                               to_stim=to_stim)

    return make


def _reference(distance, rounds, noise, data_depolarization=0):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=data_depolarization,
        after_clifford_depolarization=noise.after_clifford_depolarization,
        before_measure_flip_probability=noise.before_measure_flip_probability,
        after_reset_flip_probability=noise.after_reset_flip_probability)


def _surface_circuit_contents(circuit):
    """Compare physical operations and parity rows modulo representation.

    Coordinates identify qubits; repeats and batched gates are expanded, and
    MR becomes M followed by R. Per-qubit histories preserve operation order
    on every shared carrier while allowing disjoint operations to commute.
    Measurement indices and noise probabilities remain exact. Detector rows
    retain their order and observables retain their indices, with record
    references resolved to absolute measurement indices.

    Ticks and annotation coordinates are excluded. This helper supports the
    surface-memory gate subset, not arbitrary Stim circuit equivalence.
    """
    coordinates = {
        q: tuple(point)
        for q, point in circuit.get_final_qubit_coordinates().items()
    }
    assert len(set(coordinates.values())) == len(coordinates)
    histories = {point: [] for point in coordinates.values()}
    detectors, observables = [], []
    measurements = 0
    for op in circuit.flattened():
        name = op.name
        targets = op.targets_copy()
        args = tuple(op.gate_args_copy())
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            assert all(q.is_measurement_record_target for q in targets)
            row = tuple(sorted(measurements + q.value for q in targets))
            assert all(0 <= index < measurements for index in row)
            if name == "DETECTOR":
                detectors.append(row)
            else:
                observables.append((args, row))
            continue
        assert name in ("R", "H", "CX", "M", "MR", "DEPOLARIZE1", "DEPOLARIZE2",
                        "X_ERROR", "Z_ERROR")
        assert all(q.is_qubit_target for q in targets)
        if name in ("R", "H", "CX", "M", "MR"):
            assert not args
        width = 2 if name in ("CX", "DEPOLARIZE2") else 1
        for start in range(0, len(targets), width):
            group = tuple((coordinates[q.value], q.is_inverted_result_target)
                          for q in targets[start:start + width])
            for kind in (("M", "R") if name == "MR" else (name,)):
                record = measurements if kind == "M" else None
                token = kind, args, group, record
                for point, _ in group:
                    histories[point].append(token)
                if kind == "M":
                    measurements += 1
    return histories, detectors, observables


def _fault_laws(stim, circuit):
    """Exact categorical laws over detector/observable responses at each site.

    Place a tick at every channel and use Stim's backward sensitivities. This
    retains all three/fifteen exclusive Pauli alternatives, including faults
    with zero response, instead of comparing an approximated merged DEM.
    """
    probe = stim.Circuit()
    channels = []

    def channel(kind, probability, targets):
        tick = probe.num_ticks
        probe.append("TICK")
        width = 2 if kind == "DEPOLARIZE2" else 1
        for start in range(0, len(targets), width):
            channels.append(
                (kind, probability,
                 tuple(q.value for q in targets[start:start + width]), tick))

    for op in circuit.flattened():
        if op.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"):
            channel(op.name, op.gate_args_copy()[0], op.targets_copy())
        elif op.name in ("M", "MX", "MR", "MRX"):
            args = op.gate_args_copy()
            if args and args[0]:
                channel("Z_ERROR" if op.name.endswith("X") else "X_ERROR",
                        args[0], op.targets_copy())
            probe.append(op.name, op.targets_copy())
        elif op.name not in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            probe.append(op)

    regions = probe.detecting_regions()
    laws = Counter()
    for kind, probability, qubits, tick in channels:
        alternatives = ((1,),) if kind == "X_ERROR" else ((
            3,),) if kind == "Z_ERROR" else tuple(
                word for word in product(range(4), repeat=len(qubits))
                if any(word))
        responses = []
        for word in alternatives:
            response = 0
            for target, by_tick in regions.items():
                region = by_tick.get(tick)
                if region is None:
                    continue
                flips = sum(
                    bool(fault and region[q] and fault != region[q])
                    for q, fault in zip(qubits, word)) % 2
                index = target.val + (probe.num_detectors if
                                      target.is_logical_observable_id() else 0)
                response ^= flips << index
            responses.append(response)
        laws[kind, probability, tuple(sorted(responses))] += 1
    return laws


def _dem_terms(model):
    terms = {}
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        support = tuple(sorted(str(t) for t in instruction.targets_copy()))
        assert support not in terms
        terms[support] = instruction.args_copy()[0]
    return terms


@pytest.mark.parametrize("distance,rounds", [(3, 2), (3, 3), (5, 5)])
def test_noiseless_circuit_and_readout_detectors(make_circuit_memory, distance,
                                                 rounds):
    memory = make_circuit_memory(distance, rounds)
    noise = cql.compiler.StandardDepolarizingNoise()
    circuit = stim.Circuit(memory.to_stim(noise))
    reference = _reference(distance, rounds, noise)
    assert _surface_circuit_contents(circuit) == _surface_circuit_contents(
        reference)
    assert circuit.num_qubits == 2 * distance**2 - 1
    assert circuit.num_measurements == rounds * (distance**2 - 1) + distance**2
    assert circuit.num_detectors == rounds * (distance**2 - 1)
    assert circuit.num_observables == 1
    assert len(memory.contracts[-1].detectors) == 3 * (distance**2 - 1) // 2
    detectors, observables = circuit.compile_detector_sampler(seed=4).sample(
        32, separate_observables=True)
    assert not detectors.any() and not observables.any()


@pytest.mark.parametrize("active", [
    None, "data_depolarization",
    *cql.compiler.StandardDepolarizingNoise.__dataclass_fields__
])
def test_physical_operations_and_exact_channels_match_reference(
        make_circuit_memory, active):
    parameters = {
        name: .001 if active is None or active == name else 0
        for name in cql.compiler.StandardDepolarizingNoise.__dataclass_fields__
    }
    noise = cql.compiler.StandardDepolarizingNoise(**parameters)
    data = .001 if active in (None, "data_depolarization") else 0
    circuit = stim.Circuit(make_circuit_memory().to_stim(noise, data))
    reference = _reference(3, 3, noise, data)
    assert _surface_circuit_contents(circuit) == _surface_circuit_contents(
        reference)
    assert _fault_laws(stim, circuit) == _fault_laws(stim, reference)


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_graphlike_fault_distance(make_circuit_memory, distance):
    noise = cql.compiler.StandardDepolarizingNoise(.001, .003, .004)
    circuit = stim.Circuit(
        make_circuit_memory(distance, distance).to_stim(noise, .002))
    reference = _reference(distance, distance, noise, .002)
    assert _surface_circuit_contents(circuit) == _surface_circuit_contents(
        reference)
    if distance <= 5:
        assert _fault_laws(stim, circuit) == _fault_laws(stim, reference)
    assert len(circuit.shortest_graphlike_error()) == distance
    assert len(reference.shortest_graphlike_error()) == distance


@pytest.mark.parametrize("distance,rounds", [(3, 2), (3, 3), (3, 5), (5, 5)])
def test_composed_edem_matches_circuit_and_reference(make_circuit_memory,
                                                     distance, rounds):
    memory = make_circuit_memory(distance, rounds)
    noise = cql.compiler.StandardDepolarizingNoise(.001, .003, .004)
    placements = memory.depolarizations(.002)
    composed = cql.analysis.derive_edem_chain(memory.contracts,
                                              noise=noise,
                                              depolarizations=placements)
    projected = edem.project_DEM(composed)
    assert projected.to_bytes() == composed.to_bytes()
    assert sum(projected.detectors) == rounds * (distance**2 - 1)
    assert any(
        group.kind == edem.GroupKind.EXCLUSIVE and len(group.fault_ids) == 15
        for fault in projected.faults
        for group in fault.groups)
    loaded = edem.DEM.from_bytes(projected.to_bytes())
    reduced, mapping = edem.coalesce_dem_columns(edem.factorize_dem(loaded))
    assert len(mapping) == projected.vector_sizes[0]
    actual = _dem_terms(cql.analysis.to_stim_dem(reduced))
    for circuit in (stim.Circuit(memory.to_stim(noise, .002)),
                    _reference(distance, rounds, noise, .002)):
        assert actual == pytest.approx(_dem_terms(
            circuit.detector_error_model()),
                                       rel=0,
                                       abs=1e-12)


def test_example_exports_compares_and_samples(run_memory_example, capsys):
    result = run_memory_example("04_surface_memory_reference.py",
                                compare_with_stim=True,
                                shots=32,
                                seed=7)
    assert {path.name for path in result["output"].iterdir()} == {
        "memory.stim", "reference.stim", "memory.edemb", "prep.edemb",
        "sec.edemb", "mz.edemb", "circuit.dem", "composed.dem"
    }
    assert result["detectors"].shape == (32, 24)
    assert result["observables"].shape == (32, 1)
    assert result["projected"].vector_sizes[0] == 1307
    assert "match the circuit and Stim reference" in capsys.readouterr().out
    assert edem.DEM.from_bytes(
        (result["output"] / "memory.edemb"
        ).read_bytes()).to_bytes() == result["projected"].to_bytes()


def test_example_uses_independent_standard_noise_parameters(run_memory_example):
    result = run_memory_example("04_surface_memory_reference.py",
                                after_clifford_depolarization=.002,
                                before_measure_flip_probability=.003,
                                after_reset_flip_probability=.004,
                                data_depolarization=.005)
    circuit = stim.Circuit(result["text"])
    reference = _reference(3, 3, result["noise"], .005)
    assert _surface_circuit_contents(circuit) == _surface_circuit_contents(
        reference)
    assert _fault_laws(stim, circuit) == _fault_laws(stim, reference)


@pytest.mark.parametrize("sampling_mode", ["circuit", "dem"])
def test_example_samples_selected_representation_and_keeps_original_laws(
        run_memory_example, sampling_mode, capsys):
    result = run_memory_example("04_surface_memory_reference.py",
                                compare_with_stim=False,
                                shots=32,
                                sampling_mode=sampling_mode)
    assert result["detectors"].shape == (32, 24)
    assert result["observables"].shape == (32, 1)
    assert f"from the {sampling_mode} model" in capsys.readouterr().out
    saved = edem.DEM.from_bytes(
        (result["output"] / "memory.edemb").read_bytes())
    assert saved.to_bytes() == result["projected"].to_bytes()
    assert any(group.kind == edem.GroupKind.EXCLUSIVE
               for fault in saved.faults
               for group in fault.groups)
    if sampling_mode == "dem":
        assert all(group.kind == edem.GroupKind.INDEPENDENT
                   for fault in result["independent"].faults
                   for group in fault.groups)
    assert {path.name for path in result["output"].iterdir()} == {
        "memory.stim", "memory.edemb", "prep.edemb", "sec.edemb", "mz.edemb"
    }


def test_example_rejects_unknown_sampling_mode(run_memory_example):
    with pytest.raises(ValueError, match="sampling_mode"):
        run_memory_example("04_surface_memory_reference.py",
                           sampling_mode="unknown")
