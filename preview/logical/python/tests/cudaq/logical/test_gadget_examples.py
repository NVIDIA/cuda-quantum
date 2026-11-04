# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Execute the introductory gadget tutorials and check their exported results."""

import builtins

import pytest

import cudaq.logical as cql
import cudaq_edem as edem


def test_gadget_tutorial_verifies_contract_and_rejects_wrong_action(
        run_memory_example, capsys):
    result = run_memory_example("00_verify_gadget.py")
    receipt = result["receipt"]
    assert receipt.contract == result["contract"].name
    assert receipt.detector_contract.ok
    assert not receipt.detector_contract.failures
    assert receipt.logical_action.status == "verified"
    assert not receipt.logical_action.failures
    assert receipt.verified

    wrong = result["wrong_receipt"]
    assert wrong.contract == result["wrong_contract"].name
    assert wrong.detector_contract.ok
    assert not wrong.detector_contract.failures
    assert wrong.logical_action.status == "failed"
    assert [failure.kind for failure in wrong.logical_action.failures
           ] == ["sign_mismatch"]
    assert not wrong.verified
    output = capsys.readouterr().out
    assert "logical action: verified" in output
    assert "logical action: failed" in output


def test_surface_memory_tutorial_exports_without_stim(run_memory_example,
                                                      monkeypatch):
    original_import = builtins.__import__

    def without_stim(name, *args, **kwargs):
        if name == "stim" or name.startswith("stim."):
            raise ModuleNotFoundError("Stim is unavailable", name="stim")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_stim)
    result = run_memory_example("01_surface_memory.py", rounds=2)
    assert result["contracts"] == (result["prep"], result["readout"])
    assert len(result["verification"]) == 3
    for receipt in result["verification"].values():
        assert receipt.detector_contract.ok
        assert receipt.logical_action.status == "verified"
        assert receipt.verified
    assert result["noise"] == cql.compiler.StandardDepolarizingNoise(
        .001, .001, .001)
    assert "MPP " not in result["text"]
    assert result["text"].count("DETECTOR ") == 16
    assert result["text"].count("OBSERVABLE_INCLUDE(0)") == 1
    assert result["projected"].is_dem
    assert sum(result["projected"].detectors) == 16
    assert sum(result["projected"].observables) == 1
    assert any(
        group.kind == edem.GroupKind.EXCLUSIVE and len(group.fault_ids) == 15
        for fault in result["projected"].faults
        for group in fault.groups)
    output = result["output"]
    assert {path.name for path in output.iterdir()
           } == {"memory.stim", "memory.edemb"}
    assert (output / "memory.stim").read_text() == result["text"]
    assert edem.DEM.from_bytes(
        (output / "memory.edemb"
        ).read_bytes()).to_bytes() == result["projected"].to_bytes()


def test_surface_memory_tutorial_samples_the_physical_circuit(
        run_memory_example, capsys):
    stim = pytest.importorskip("stim", minversion="1.16")
    result = run_memory_example("01_surface_memory.py", shots=32, seed=7)
    assert result["contracts"] == (result["prep"], result["sec"],
                                   result["readout"])
    circuit = stim.Circuit(result["text"])
    assert circuit.num_qubits == 17
    assert circuit.num_measurements == 33
    assert circuit.num_detectors == 24
    assert circuit.num_observables == 1
    assert result["detectors"].shape == (32, 24)
    assert result["observables"].shape == (32, 1)
    detectors, observables = circuit.compile_detector_sampler(seed=7).sample(
        32, separate_observables=True)
    assert (result["detectors"] == detectors).all()
    assert (result["observables"] == observables).all()
    assert "Sampled 32 shots" in capsys.readouterr().out

    reduced, _ = edem.coalesce_dem_columns(
        edem.factorize_dem(result["projected"]))

    def error_terms(model):
        return {
            tuple(sorted(str(target) for target in op.targets_copy())):
                op.args_copy()
                [0] for op in model.flattened() if op.type == "error"
        }

    assert error_terms(cql.analysis.to_stim_dem(reduced)) == pytest.approx(
        error_terms(circuit.detector_error_model()), rel=0, abs=1e-12)
