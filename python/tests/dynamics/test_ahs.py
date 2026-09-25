# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import dataclasses

import numpy as np
import pytest

from cudaq import cudaq_runtime


def _program(sites, omega, phase, detuning, times):
    ahs = cudaq_runtime.ahs
    program = ahs.Program()
    program.setup.ahs_register.sites = sites
    program.setup.ahs_register.filling = [1] * len(sites)
    drive = ahs.DrivingField()
    for name, values in (("amplitude", omega), ("phase", phase), ("detuning",
                                                                  detuning)):
        field = ahs.PhysicalField()
        field.time_series = ahs.TimeSeries(list(zip(values, times)))
        setattr(drive, name, field)
    program.hamiltonian.drivingFields = [drive]
    return program


def test_emulation_pulser_sampling_parity(monkeypatch):
    """Compare GPU emulation samples with Pulser on an asymmetric register."""
    num_atoms = 3
    import cudaq

    if not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics") and
            cudaq.num_available_gpus() > 0):
        pytest.skip(
            "AHS emulation requires the PASQAL target, dynamics and a GPU")
    pulser = pytest.importorskip("pulser")
    simulation = pytest.importorskip("pulser_simulation")
    sites_um = [(0., 0.), (6., 0.), (0., 8.)][:num_atoms]
    device = dataclasses.replace(pulser.devices.MockDevice, rydberg_level=60)
    sequence = pulser.Sequence(
        pulser.Register.from_coordinates(sites_um, center=False, prefix="q"),
        device)
    sequence.declare_channel("drive", "rydberg_global")
    sequence.add(pulser.Pulse.ConstantPulse(800, 4., 1.5, 0.7), "drive")
    reference = simulation.QutipEmulator.from_sequence(sequence,
                                                       sampling_rate=1.,
                                                       with_modulation=False)
    state = reference.run(rtol=1e-10,
                          atol=1e-12).get_state(0.4).full().ravel()[::-1]
    program = _program([(x * 1e-6, y * 1e-6) for x, y in sites_um], [4e6] * 2,
                       [0.7] * 2, [1.5e6] * 2, [0., 4e-7])
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    cudaq.set_target("pasqal", emulate=True)
    cudaq.set_random_seed(31)
    result = cudaq_runtime.launch_analog_kernel(
        "__analog_hamiltonian_kernel__pulser_parity", program.to_json(), 20000)
    assert result.get_total_shots() == 20000
    for basis, probability in enumerate(np.abs(state)**2):
        assert result.probability(format(
            basis, f"0{num_atoms}b")) == pytest.approx(probability, abs=0.02)


def test_emulation_seeding_matches_digital(monkeypatch):
    """Seeded launches repeat exactly, sync or queued, as digital emulation."""
    import cudaq

    if not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics") and
            cudaq.num_available_gpus()):
        pytest.skip("AHS emulation requires PASQAL, dynamics and a GPU")
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    cudaq.set_target("pasqal", emulate=True)
    # A pi/2 pulse on two distant atoms: each basis state has probability 1/4.
    payload = _program([(0., 0.), (1e-4, 0.)], [np.pi / 2 * 1e7] * 2, [0.] * 2,
                       [0.] * 2, [0., 1e-7]).to_json()
    name = "__analog_hamiltonian_kernel__seeding"
    counts = lambda result: {k: result.count(k) for k in result}

    cudaq.set_random_seed(11)
    futures = [
        cudaq_runtime.launch_analog_kernel_async(name, payload, 4000)
        for _ in range(4)
    ]
    results = [counts(cudaq_runtime.launch_analog_kernel(name, payload, 4000))]
    results += [counts(f.get()) for f in futures]
    assert all(result == results[0] for result in results)
    for bits in ("00", "01", "10", "11"):
        assert results[0][bits] / 4000 == pytest.approx(0.25, abs=0.04)

    cudaq.set_random_seed(0)
    unseeded = [
        counts(cudaq_runtime.launch_analog_kernel(name, payload, 4000))
        for _ in range(2)
    ]
    assert unseeded[0] != unseeded[1]


def test_invalid_emulation_rydberg_c6_fails_at_set_target():
    """Reject a malformed C6 override when selecting the target."""
    import cudaq

    if not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics")):
        pytest.skip("AHS emulation requires PASQAL and dynamics")
    with pytest.raises(ValueError,
                       match="Invalid `emulation_rydberg_c6` value"):
        cudaq.set_target("pasqal",
                         emulate=True,
                         emulation_rydberg_c6="5e-24xyz")
    cudaq.reset_target()


def test_vacancy_e2e(monkeypatch):
    """Evolve occupied atoms around a vacancy through the real QPU launch path."""
    import cudaq
    from cudaq.dynamics import Schedule
    from cudaq.operators import RydbergHamiltonian, ScalarOperator
    from scipy.linalg import expm

    if not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics") and
            cudaq.num_available_gpus()):
        pytest.skip("AHS emulation requires PASQAL, dynamics and a GPU")
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    c6 = 5.42e-24
    cudaq.set_target("pasqal", emulate=True, emulation_rydberg_c6=str(c6))
    cudaq.set_random_seed(47)
    h = RydbergHamiltonian([(0., 0.), (1e-6, 0.), (6e-6, 0.)],
                           ScalarOperator.const(4e6),
                           ScalarOperator.const(0.7),
                           ScalarOperator.const(1e6),
                           atom_filling=[1, 0, 1])
    result = cudaq.evolve(h,
                          schedule=Schedule([0., 4e-7], ["t"]),
                          shots_count=20000)
    x = np.array([[0., 1.], [1., 0.]])
    y = np.array([[0., -1j], [1j, 0.]])
    n, identity = np.diag([0., 1.]), np.eye(2)
    single = 2e6 * (np.cos(0.7) * x + np.sin(0.7) * y) - 1e6 * n
    matrix = (np.kron(single, identity) + np.kron(identity, single) + c6 /
              (6e-6)**6 * np.kron(n, n))
    state = expm(-4e-7j * matrix) @ [1., 0., 0., 0.]
    assert result.get_total_shots() == 20000
    for basis, probability in enumerate(np.abs(state)**2):
        left, right = format(basis, "02b")
        assert result.probability(left + "0" + right) == pytest.approx(
            probability, abs=0.02)


def test_cpp_emulation_e2e(tmp_path, monkeypatch):
    """Compile and run C++ AHS evolution with the actual nvq++ emulation flag."""
    import cudaq
    import shutil
    import subprocess

    compiler = shutil.which("nvq++")
    if not (compiler and cudaq.has_target("pasqal") and
            cudaq.has_target("dynamics") and cudaq.num_available_gpus()):
        pytest.skip(
            "C++ AHS emulation requires nvq++, PASQAL, dynamics and a GPU")
    source = tmp_path / "ahs.cpp"
    executable = tmp_path / "ahs"
    source.write_text(r'''
#include <cudaq.h>
#include <cudaq/algorithms/integrator.h>
#include <cudaq/algorithms/evolve.h>
#include <cassert>
#include <numbers>

int main() {
  cudaq::rydberg_hamiltonian pulse(
      {{0., 0.}, {6e-6, 0.}}, cudaq::scalar_operator(std::numbers::pi / 2e-7),
      cudaq::scalar_operator(0.7), cudaq::scalar_operator(0.), {0, 1});
  cudaq::schedule times(std::vector<double>{0., 2e-7}, {"t"});
  auto sync = cudaq::evolve(pulse, times, 31);
  assert(sync.sampling_result->count("01") == 31);
  auto async = cudaq::evolve_async(pulse, times, 37).get();
  assert(async.sampling_result->count("01") == 37);
  cudaq::rydberg_hamiltonian idle({{0., 0.}, {6e-6, 0.}},
      cudaq::scalar_operator(0.), cudaq::scalar_operator(0.),
      cudaq::scalar_operator(0.), {0, 1});
  auto ground = cudaq::evolve(idle, times, 19);
  assert(ground.sampling_result->count("00") == 19);
  // A pi/2 pulse: seeded asynchronous runs repeat exactly.
  cudaq::schedule half(std::vector<double>{0., 1e-7}, {"t"});
  cudaq::set_random_seed(5);
  auto first = cudaq::evolve_async(pulse, half, 500).get();
  cudaq::set_random_seed(5);
  auto second = cudaq::evolve_async(pulse, half, 500).get();
  const auto excited = first.sampling_result->count("01");
  assert(excited == second.sampling_result->count("01"));
  assert(excited > 150 && excited < 350);
}
''')
    monkeypatch.setenv("DISABLE_REMOTE_SEND", "1")
    compiled = subprocess.run([
        compiler, "--target", "pasqal", "--emulate",
        str(source), "-o",
        str(executable)
    ],
                              capture_output=True,
                              text=True,
                              timeout=120)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(executable)],
                              capture_output=True,
                              text=True,
                              timeout=60)
    assert executed.returncode == 0, executed.stdout + executed.stderr
