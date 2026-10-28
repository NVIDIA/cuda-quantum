# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Recipe defaults, custom occurrence faults, and simulator-free export."""

import builtins
from dataclasses import replace

import pytest

import cudaq.logical as cql
from cudaq.logical.architectures import surface
from cudaq.logical.analysis import chain
import cudaq_edem as edem


@pytest.mark.parametrize("name", [
    "prepare_zero", "prepare_one", "prepare_plus", "prepare_minus",
    "memory_round", "measure_z", "measure_x", "h", "fold_s", "transversal_cx",
    "cz"
])
@pytest.mark.parametrize("distance", [3, 5])
def test_recipe_gadget_profiles_are_stable_and_semantically_valid(
        name, distance):
    gadget = getattr(surface.definitions(distance), name)
    profile = surface.gadget_profile(gadget)
    assert profile is surface.gadget_profile(gadget)
    assert profile.gadget is gadget
    cql.analysis.derive_edem(cql.compile(profile))


def test_generated_surface_measurement_exposes_its_existing_profile():
    recipe = surface.definitions(3)
    measurement = recipe.wsc_measurement(cql.types.Z(0) @ cql.types.Z(1),
                                         rounds=1)
    assert surface.gadget_profile(
        measurement.realization) is measurement.analysis
    assert cql.compile(measurement.analysis).module.operation.verify()


def test_custom_profile_overrides_default():
    gadget = surface.definitions(3).memory_round
    profile = surface.gadget_profile(gadget)
    custom = replace(profile,
                     boundary=None,
                     detectors=profile.detectors + profile.detectors[:1])
    # Redundant detectors are valid and must not be replaced by the default.
    model = cql.analysis.derive_edem_chain(
        (surface.gadget_profile(surface.definitions(3).prepare_zero), custom,
         surface.gadget_profile(surface.definitions(3).measure_z)))
    assert sum(model.detectors) == 4 + 9
    assert surface.gadget_profile(gadget) is profile


def test_chain_reuses_equal_faults_but_distinguishes_occurrence_noise(
        monkeypatch):
    recipe = surface.definitions(3)
    profiles = tuple(
        surface.gadget_profile(g)
        for g in (recipe.prepare_zero, recipe.memory_round, recipe.memory_round,
                  recipe.memory_round, recipe.measure_z))
    original = chain.derive_edem
    derived = []

    def record(build, *, faults):
        derived.append((build, faults))
        return original(build, faults=faults)

    def faults(index, profile, projection):
        result = cql.analysis.FaultSet()
        if index in (1, 2, 3):
            result = result.pauli_error(
                "", projection.interface.inputs[0].data_carriers[0], "x",
                .02 if index == 3 else .01)
        return result

    monkeypatch.setattr(chain, "derive_edem", record)
    model = chain.derive_edem_chain(profiles, fault_factory=faults)
    assert len(derived) == 4  # preparation, two noise settings, readout
    assert [f.groups[0].probabilities for f in model.faults] == [[.01], [.01],
                                                                 [.02]]
    assert model.inputs == model.outputs == []


def test_chain_rejects_ambiguous_noise_configuration():
    recipe = surface.definitions(3)
    profiles = tuple(
        surface.gadget_profile(g)
        for g in (recipe.prepare_zero, recipe.measure_z))
    with pytest.raises(ValueError, match="choose fault_factory"):
        chain.derive_edem_chain(
            profiles,
            noise=cql.compiler.StandardDepolarizingNoise(),
            fault_factory=lambda *args: cql.analysis.FaultSet())
    with pytest.raises(ValueError, match="occurrence index"):
        chain.derive_edem_chain(profiles, depolarizations={2: ()})


def test_experiment_exports_biased_noise_without_simulators(
        run_memory_example, monkeypatch):
    original = builtins.__import__

    def without_samplers(name, *args, **kwargs):
        if name in ("stim", "cudaq_qec") or name.startswith(
            ("stim.", "cudaq_qec.")):
            raise ModuleNotFoundError(name, name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_samplers)
    result = run_memory_example("07_memory_edem.py",
                                rounds=4,
                                noisy_rounds=(1, 3),
                                noisy_data_qubits=(0, 1),
                                data_pauli_probabilities={
                                    "x": .001,
                                    "y": .002,
                                    "z": .01
                                })
    loaded = edem.DEM.from_bytes(result["artifact"].read_bytes())
    assert loaded.to_bytes() == result["projected"].to_bytes()
    assert sum(loaded.detectors) == 28
    assert sum(f.size for f in loaded.faults) == 12
    assert all(g.kind == edem.GroupKind.EXCLUSIVE and
               g.probabilities == [.001, .002, .01]
               for f in loaded.faults
               for g in f.groups)
    with pytest.raises(ValueError, match="unsupported fault distribution"):
        edem.factorize_dem(loaded)


def test_surface_recipe_rejects_foreign_gadget_with_matching_code_and_name():
    foreign = cql.gadgets.css_memory_round(
        cql.codes.Surface[3], name=surface.definitions(3).memory_round.name)
    with pytest.raises(ValueError, match="does not belong"):
        surface.gadget_profile(foreign)
