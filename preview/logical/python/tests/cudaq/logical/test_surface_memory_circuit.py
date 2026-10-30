# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""The interleaved ancilla recipe, its contracts and explicit noise entries."""

import builtins
from dataclasses import replace

import pytest

import cudaq.logical as cql
from cudaq.logical.architectures import surface
from cudaq.logical.architecture.geometry import rotated_surface_lattice
import cudaq_edem as edem


def test_recipe_is_cached_and_distinct_from_ideal_mpp():
    circuit = surface.interleaved_nz_definitions(3)
    ideal = surface.definitions(3)
    assert circuit is surface.interleaved_nz_definitions(3)
    assert ideal is surface.definitions(3)
    assert circuit.code is ideal.code
    for gadget, has_mpp in ((circuit.syndrome_extraction, False),
                            (ideal.memory_round, True)):
        contract = surface.gadget_contract(gadget)
        assert contract is surface.gadget_contract(gadget)
        projection = cql.compiler.project_clifford(cql.compile(contract),
                                                   gadget=contract.gadget.name)
        assert any(
            op.kind == "MPP" for op in projection.instructions) == has_mpp
    assert len(ideal.coordinates) == 9
    assert len(circuit.coordinates) == 17


@pytest.mark.parametrize("example, rounds", [
    ("06_stim_emission.py", 0),
    ("06_stim_emission.py", True),
    ("06_stim_emission.py", 2.5),
    ("07_memory_edem.py", 0),
    ("08_surface_memory_circuit.py", 0),
    ("08_surface_memory_circuit.py", 1),
    ("08_surface_memory_circuit.py", "3"),
])
def test_examples_reject_invalid_round_counts(run_memory_example, example,
                                              rounds):
    with pytest.raises(ValueError, match="rounds must be an integer"):
        run_memory_example(example, rounds=rounds)


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_interleaved_layers_cover_checks_and_orient_hooks(distance):
    recipe = surface.interleaved_nz_definitions(distance)
    contract = surface.gadget_contract(recipe.syndrome_extraction)
    projection = cql.compiler.project_clifford(cql.compile(contract),
                                               gadget=contract.gadget.name)
    port, = projection.interface.inputs
    lattice = rotated_surface_lattice(distance)
    xs, zs = port.carriers("sx"), port.carriers("sz")
    coordinates = dict(zip(port.data_carriers, lattice.coordinates))
    for ancillas, sublattice in ((xs, 1), (zs, 0)):
        coordinates.update(
            zip(ancillas, (face.coordinate
                           for face in lattice.faces
                           if face.sublattice == sublattice)))
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
    # Check incidence coverage against the code algebra, not the circuit builder.
    for ancillas, checks in ((xs, recipe.code.hx), (zs, recipe.code.hz)):
        for ancilla, support in zip(ancillas, checks):
            assert sorted(histories[ancilla]) == sorted(
                lattice.coordinates[q] for q in support)
    assert len({lattice.coordinates[q][0] for q in recipe.code.lx[0]}) == 1
    assert len({lattice.coordinates[q][1] for q in recipe.code.lz[0]}) == 1
    for ancilla, neighbors in histories.items():
        if len(neighbors) == 4:
            a, b = neighbors[-2:]
            if ancilla in xs:
                assert a[1] == b[1] and a[0] != b[0]
            else:
                assert a[0] == b[0] and a[1] != b[1]


@pytest.mark.parametrize("distance", [3, 5])
def test_complete_contracts_and_physical_endcaps(distance):
    recipe = surface.interleaved_nz_definitions(distance)
    for gadget, expected in ((recipe.prepare_zero,
                              (0, 1)), (recipe.syndrome_extraction, (1, 1)),
                             (recipe.measure_z_with_syndrome, (1, 0))):
        contract = surface.gadget_contract(gadget)
        assert contract.boundary_complete
        receipt = cql.gadgets.verify_gadget(contract)
        assert receipt.verified
        model = cql.analysis.derive_edem(cql.compile(contract),
                                         faults=cql.analysis.FaultSet())
        assert (len(model.inputs), len(model.outputs)) == expected
    assert len(recipe.coordinates) == 2 * distance**2 - 1


def test_explicit_data_noise_precedes_extraction_including_readout():
    recipe = surface.interleaved_nz_definitions(3)
    prep, sec, readout = map(surface.gadget_contract,
                             (recipe.prepare_zero, recipe.syndrome_extraction,
                              recipe.measure_z_with_syndrome))
    contracts = (prep, sec, sec, readout)
    for contract in contracts:
        projection = cql.compiler.project_clifford(cql.compile(contract),
                                                   gadget=contract.gadget.name)
        entry, = surface.data_depol(contract, projection, .002)
        position = next(i for i, op in enumerate(projection.instructions)
                        if op.site_id == entry.before)
        assert all(op.kind == "R" for op in projection.instructions[:position])
        op = projection.instructions[position]
        port, = projection.interface.inputs or projection.interface.outputs
        assert op.kind == "H" and op.carriers == port.carriers("sx")
        assert entry.carriers == port.data_carriers


@pytest.mark.parametrize("name", [
    "prepare_zero", "prepare_one", "prepare_plus", "prepare_minus",
    "memory_round"
])
def test_ideal_mpp_noise_site_follows_initialization(name):
    contract = surface.gadget_contract(getattr(surface.definitions(3), name))
    projection = cql.compiler.project_clifford(cql.compile(contract),
                                               gadget=contract.gadget.name)
    entry, = surface.data_depol(contract, projection, .002)
    position = next(i for i, op in enumerate(projection.instructions)
                    if op.site_id == entry.before)
    assert projection.instructions[position].kind == "MPP"
    assert all(
        op.kind in ("R", "H") for op in projection.instructions[:position])
    assert entry.carriers == projection.interface.output_data_carriers


def test_data_noise_site_validates_realization_and_does_not_compile(
        monkeypatch):
    recipe = surface.interleaved_nz_definitions(3)
    contract = surface.gadget_contract(recipe.syndrome_extraction)
    projection = cql.compiler.project_clifford(cql.compile(contract),
                                               gadget=contract.gadget.name)
    foreign = surface.gadget_contract(surface.definitions(3).memory_round)
    other = cql.compiler.project_clifford(cql.compile(foreign),
                                          gadget=foreign.gadget.name)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("data_depol must use the supplied projection")

    monkeypatch.setattr(cql.compiler, "compile", unexpected_compile)
    assert surface.data_depol(contract, projection, 0)[0].probability == 0
    with pytest.raises(ValueError, match="does not match"):
        surface.data_depol(contract, other, .002)
    instructions = list(projection.instructions)
    index = next(i for i, op in enumerate(instructions) if op.kind == "CX")
    instructions[index] = replace(instructions[index], kind="CZ")
    with pytest.raises(ValueError, match="does not match"):
        surface.data_depol(
            contract, replace(projection, instructions=tuple(instructions)),
            .002)
    with pytest.raises(ValueError, match="no data-noise site"):
        surface.data_depol(contract, projection, .002, site="before_readout")
    with pytest.raises(TypeError, match="gadget contract"):
        surface.data_depol(contract.gadget, projection, .002)
    with pytest.raises(TypeError, match="CliffordProjection"):
        surface.data_depol(contract, object(), .002)


def test_direct_readout_has_no_syndrome_noise_site():
    contract = surface.gadget_contract(surface.definitions(3).measure_z)
    projection = cql.compiler.project_clifford(cql.compile(contract),
                                               gadget=contract.gadget.name)
    with pytest.raises(ValueError, match="no data-noise site"):
        surface.data_depol(contract, projection, .002)


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan"), True])
def test_data_noise_site_validates_probability(probability):
    contract = surface.gadget_contract(surface.definitions(3).memory_round)
    projection = cql.compiler.project_clifford(cql.compile(contract),
                                               gadget=contract.gadget.name)
    with pytest.raises(ValueError, match="probability"):
        surface.data_depol(contract, projection, probability)


def test_example_exports_original_models_without_stim(run_memory_example,
                                                      monkeypatch):
    original_import = builtins.__import__

    def without_stim(name, *args, **kwargs):
        if name == "stim" or name.startswith("stim."):
            raise ModuleNotFoundError("Stim is unavailable", name="stim")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_stim)
    result = run_memory_example("08_surface_memory_circuit.py",
                                compare_with_stim=False,
                                rounds=5)
    assert {path.name for path in result["output"].iterdir()} == {
        "memory.stim", "memory.edemb", "prep.edemb", "sec.edemb", "mz.edemb"
    }
    occurrences = result["occurrences"]
    assert len(occurrences) == 5 and len({id(m) for m in occurrences}) == 3
    assert occurrences[1] is occurrences[2] is occurrences[3]
    assert result["projected"].is_dem
    for name, contract in (("prep", result["prep"]), ("sec", result["sec"]),
                           ("mz", result["readout"])):
        model = edem.EDEM.from_bytes(
            (result["output"] / f"{name}.edemb").read_bytes())
        assert model.to_bytes() == result["models"][contract.gadget].to_bytes()
    assert any(
        group.kind == edem.GroupKind.EXCLUSIVE and len(group.fault_ids) == 15
        for fault in result["projected"].faults
        for group in fault.groups)


def test_example_requires_both_endcap_extractions(run_memory_example):
    with pytest.raises(ValueError, match="at least two"):
        run_memory_example("08_surface_memory_circuit.py", rounds=1)
