# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Surface-memory gadget profiles against an independent Stim generator."""

import builtins
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
import runpy

import pytest

import cudaq.logical as cql
from cudaq.logical.architecture.geometry import rotated_surface_lattice

EXAMPLE = Path(
    __file__).resolve().parents[4] / "examples/standalone/06_surface_memory.py"


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(EXAMPLE))


@pytest.fixture(scope="module")
def make_memory(example):
    return example["surface_memory"]


@pytest.fixture(scope="module")
def stim():
    return pytest.importorskip("stim", minversion="1.16")


def _reference(stim, distance, rounds, noise, data_depolarization=0.0):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=data_depolarization,
        **{name: getattr(noise, name) for name in noise.__dataclass_fields__})


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


@pytest.mark.parametrize(
    "before, after",
    [("CX 0 1", "CX 1 0"), ("R 0 1\n", ""),
     ("DEPOLARIZE1(0.1)", "DEPOLARIZE1(0.11)"),
     ("DEPOLARIZE2(0.2)", "DEPOLARIZE1(0.2)"),
     ("H 0\nDEPOLARIZE1(0.1) 0", "DEPOLARIZE1(0.1) 0\nH 0"),
     ("M 0 1", "M !0 1"), ("DETECTOR rec[-2] rec[-1]", "DETECTOR rec[-2]"),
     ("OBSERVABLE_INCLUDE(0) rec[-1]", "OBSERVABLE_INCLUDE(0) rec[-2]")])
def test_surface_circuit_comparison_detects_physical_and_parity_changes(
        stim, before, after):
    text = """QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
R 0 1
H 0
DEPOLARIZE1(0.1) 0
CX 0 1
DEPOLARIZE2(0.2) 0 1
M 0 1
DETECTOR rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
    changed = text.replace(before, after, 1)
    assert changed != text
    assert (_surface_circuit_contents(stim.Circuit(text))
            != _surface_circuit_contents(stim.Circuit(changed)))


def test_surface_memory_emits_text_without_stim(example, make_memory,
                                                monkeypatch, tmp_path, capsys):
    original_import = builtins.__import__

    def without_stim(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == "stim" or name.startswith("stim.")):
            raise ImportError("Stim is unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_stim)
    memory = make_memory(3, 3)
    noise = cql.compiler.StandardDepolarizingNoise(.001, .003, .004)
    text = memory.to_stim(noise, data_depolarization=.002)
    output = tmp_path / "surface-memory.stim"
    output.write_text(text)
    assert output.read_text() == text
    lines = text.splitlines()
    assert sum(line.startswith("QUBIT_COORDS") for line in lines) == 17
    assert sum(line.startswith("DETECTOR") for line in lines) == 24
    assert sum(line.startswith("OBSERVABLE_INCLUDE(0)") for line in lines) == 1
    assert "DEPOLARIZE1(0.002)" in text
    assert "DEPOLARIZE2(0.001)" in text
    assert "X_ERROR(0.003)" in text
    assert "X_ERROR(0.004)" in text
    example["main"]([])
    captured = capsys.readouterr()
    assert captured.out.startswith("QUBIT_COORDS(")
    assert captured.err == ""
    with pytest.raises(SystemExit) as error:
        example["main"](["--shots", "1"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "sampling requires Stim" in captured.err


def test_surface_memory_example_samples_emitted_text(example, stim, capsys):
    example["main"](["--shots", "1000", "--seed", "7"])
    captured = capsys.readouterr()
    circuit = stim.Circuit(captured.out)
    assert circuit.num_measurements == 33
    assert circuit.num_detectors == 24
    assert circuit.num_observables == 1
    assert "Sampled 1000 shots (detectors=24, observables=1)" in captured.err
    assert "Shots with detection events:" in captured.err
    assert "Shots with uncorrected observable flips:" in captured.err


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_surface_memory_interleaves_hook_safe_layers(make_memory, distance):
    memory = make_memory(distance, 3)
    sec = memory.profiles[1]
    projection = cql.compiler.project_clifford(cql.compile(sec),
                                               gadget=sec.gadget.name)
    port, = projection.interface.inputs
    lattice = rotated_surface_lattice(distance)
    xs, zs = port.carriers("sx"), port.carriers("sz")
    coordinates = dict(zip(port.data_carriers, lattice.coordinates))
    for ancillas, sublattice in ((xs, 1), (zs, 0)):
        coordinates.update(
            zip(ancillas, (face.coordinate
                           for face in lattice.faces
                           if face.sublattice == sublattice)))
    # These are the four N/Z layers, expressed in physical corner coordinates.
    orders = ({
        (1, 1): 0,
        (-1, 1): 1,
        (1, -1): 2,
        (-1, -1): 3
    }, {
        (1, 1): 0,
        (1, -1): 1,
        (-1, 1): 2,
        (-1, -1): 3
    })
    used = [set() for _ in range(4)]
    histories = {ancilla: [] for ancilla in (*xs, *zs)}
    last_layer = 0
    for op in projection.instructions:
        if op.kind != "CX":
            continue
        for control, target in zip(op.carriers[::2], op.carriers[1::2]):
            x_check = control in xs
            assert x_check or target in zs
            ancilla, data = (control, target) if x_check else (target, control)
            a, d = coordinates[ancilla], coordinates[data]
            layer = orders[0 if x_check else 1][d[0] - a[0], d[1] - a[1]]
            assert layer >= last_layer
            assert not used[layer].intersection((control, target))
            used[layer].update((control, target))
            histories[ancilla].append(d)
            last_layer = layer
    assert all(used)
    code = cql.codes.Surface[distance]
    # The same-Pauli logical strings run perpendicular to each hook pair.
    assert len({lattice.coordinates[q][0] for q in code.lx[0]}) == 1
    assert len({lattice.coordinates[q][1] for q in code.lz[0]}) == 1
    for ancilla, neighbors in histories.items():
        assert len(neighbors) in (2, 4)
        if len(neighbors) == 4:
            a, b = neighbors[-2:]
            if ancilla in xs:
                assert a[1] == b[1] and a[0] != b[0]
            else:
                assert a[0] == b[0] and a[1] != b[1]


@pytest.mark.parametrize("distance, rounds", [(3, 2), (3, 3), (5, 5)])
def test_surface_memory_profiles_and_noiseless_boundaries(
        make_memory, stim, distance, rounds):
    memory = make_memory(distance, rounds)
    before = tuple(cql.compile(p).content_sha256 for p in memory.profiles)
    circuit = stim.Circuit(memory.to_stim())
    reference = _reference(stim, distance, rounds,
                           cql.compiler.StandardDepolarizingNoise())
    assert _surface_circuit_contents(circuit) == _surface_circuit_contents(
        reference)
    assert circuit.num_measurements == reference.num_measurements == rounds * (
        distance**2 - 1) + distance**2
    assert circuit.num_detectors == reference.num_detectors == rounds * (
        distance**2 - 1)
    assert circuit.num_observables == reference.num_observables == 1
    assert len(memory.coordinates) == 2 * distance**2 - 1
    assert len(memory.profiles[0].gadget.interface.inputs) == 0
    assert len(memory.profiles[-1].gadget.interface.outputs) == 0
    assert len(memory.profiles[-1].detectors) == 3 * (distance**2 - 1) // 2
    # DEM extraction rejects nondeterministic detector or observable parities.
    model = circuit.detector_error_model()
    assert not any(op.type == "error" for op in model.flattened())
    detectors, observables = circuit.compile_detector_sampler(seed=4).sample(
        64, separate_observables=True)
    assert detectors.shape == (64, circuit.num_detectors)
    assert observables.shape == (64, 1)
    assert not detectors.any() and not observables.any()
    assert before == tuple(
        cql.compile(p).content_sha256 for p in memory.profiles)


@pytest.mark.parametrize("active", [
    None, "data_depolarization",
    *cql.compiler.StandardDepolarizingNoise.__dataclass_fields__
])
def test_surface_memory_exact_fault_laws_match_stim(make_memory, stim, active):
    parameters = {
        name: .001 if active is None or active == name else 0
        for name in cql.compiler.StandardDepolarizingNoise.__dataclass_fields__
    }
    noise = cql.compiler.StandardDepolarizingNoise(**parameters)
    data_probability = .001 if active in (None, "data_depolarization") else 0
    memory = make_memory(3, 3)
    circuit = stim.Circuit(
        memory.to_stim(noise, data_depolarization=data_probability))
    reference = _reference(stim, 3, 3, noise, data_probability)
    assert _surface_circuit_contents(circuit) == _surface_circuit_contents(
        reference)
    assert _fault_laws(stim, circuit) == _fault_laws(stim, reference)
    assert len(circuit.shortest_graphlike_error()) == 3


def test_distance_five_memory_matches_exact_fault_laws(make_memory, stim):
    noise = cql.compiler.StandardDepolarizingNoise(.001, .003, .004)
    circuit = stim.Circuit(
        make_memory(5, 5).to_stim(noise, data_depolarization=.002))
    assert _surface_circuit_contents(circuit) == _surface_circuit_contents(
        _reference(stim, 5, 5, noise, .002))
    assert _fault_laws(stim, circuit) == _fault_laws(
        stim, _reference(stim, 5, 5, noise, .002))
    assert len(circuit.shortest_graphlike_error()) == 5


def test_distance_seven_memory_preserves_circuit_distance(make_memory, stim):
    noise = cql.compiler.StandardDepolarizingNoise(.001, .001, .001)
    circuit = stim.Circuit(
        make_memory(7, 7).to_stim(noise, data_depolarization=.001))
    assert _surface_circuit_contents(circuit) == _surface_circuit_contents(
        _reference(stim, 7, 7, noise, .001))
    assert len(circuit.shortest_graphlike_error()) == 7


def test_stim_export_validates_explicit_noise_sites_and_closed_chain(
        make_memory):
    memory = make_memory()
    # Operation-attached noise needs no extra layer placements.
    assert "DEPOLARIZE2(0.01)" in cql.compiler.emit_stim(
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
                                       site, (len(memory.coordinates),), .01),)
                               })
    with pytest.raises(TypeError, match="Depolarize1Placement"):
        cql.compiler.emit_stim(memory.profiles,
                               depolarizations={0: ("missing",)})


def test_extra_data_layers_match_extraction_entries(make_memory, stim):
    memory = make_memory(3, 3)
    placements = memory.depolarizations(.02)
    channels = [
        op for op in stim.Circuit(memory.to_stim(data_depolarization=.02))
        if op.name == "DEPOLARIZE1"
    ]
    assert sum(len(op.targets_copy()) for op in channels) == 3 * 9
    for index, profile in enumerate(memory.profiles):
        projection = cql.compiler.project_clifford(cql.compile(profile),
                                                   gadget=profile.gadget.name)
        entry = next(i for i, op in enumerate(projection.instructions)
                     if op.site_id == placements[index][0].before)
        assert all(op.kind == "R" for op in projection.instructions[:entry])
        assert projection.instructions[entry].kind == "H"
        port, = projection.interface.inputs or projection.interface.outputs
        assert projection.instructions[entry].carriers == port.carriers("sx")
        assert placements[index][0].carriers == port.data_carriers


def test_stim_export_rejects_missing_virtual_bindings_and_incompatible_presentations(
        make_memory):
    memory = make_memory()
    prep, sec, readout = memory.profiles
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
    graph = cql.gadgets.ProfileGraph(negative_readout.interface)
    graph.detector(cql.gadgets.ProfileParity(records=(record,), constant=True))
    profiles = (cql.gadgets.GadgetProfile(prepare), graph.freeze())
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
