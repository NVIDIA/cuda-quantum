# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np
import pytest

ALLOCATION_ERROR = (
    "qubit allocations are not supported inside an atomic quantum region; "
    "allocate in the caller and pass the qubits as arguments")
MEASUREMENT_ERROR = (
    "measurement operations are not supported inside an atomic quantum region; "
    "measure outside the region")


@cudaq.kernel(atomic_quantum_region=True)
def atomic_h(q: cudaq.qubit):
    h(q)


@cudaq.kernel(atomic_quantum_region=True)
def atomic_entangling_workload(q: cudaq.qview):
    h(q[0])
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])


@cudaq.kernel
def ordinary_entangling_workload(q: cudaq.qview):
    h(q[0])
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])


@cudaq.kernel
def ordinary_entangling_round_trip():
    q = cudaq.qvector(3)
    ordinary_entangling_workload(q)
    cudaq.adjoint(ordinary_entangling_workload, q)


@cudaq.kernel
def atomic_entangling_round_trip_with_spectator():
    q = cudaq.qvector(4)
    y(q[3])
    atomic_entangling_workload(q)
    cudaq.adjoint(atomic_entangling_workload, q)
    y(q[3])


@cudaq.kernel
def atomic_round_trip():
    q = cudaq.qubit()
    atomic_h(q)
    cudaq.adjoint(atomic_h, q)


@cudaq.kernel(atomic_quantum_region=True)
def single_atomic_h(q: cudaq.qubit):
    h(q)


@cudaq.kernel
def single_atomic_h_entry():
    q = cudaq.qubit()
    single_atomic_h(q)


@cudaq.kernel
def measured_atomic_round_trip() -> int:
    q = cudaq.qubit()
    atomic_h(q)
    cudaq.adjoint(atomic_h, q)
    result = 0
    if mz(q):
        result = 1
    return result


@pytest.fixture(autouse=True)
def reset_cudaq_state():
    yield
    cudaq.reset_target()
    cudaq.__clearKernelRegistries()


def test_atomic_quantum_region_decorator_contract():

    @cudaq.kernel
    def ordinary():
        pass

    @cudaq.kernel(atomic_quantum_region=False)
    def explicitly_ordinary():
        pass

    assert atomic_h.atomic_quantum_region is True
    assert ordinary.atomic_quantum_region is False
    assert explicitly_ordinary.atomic_quantum_region is False

    # The marker determines the cached module. Keep it immutable after kernel
    # construction so Python state cannot diverge from compiled IR.
    str(ordinary)
    with pytest.raises(AttributeError):
        ordinary.atomic_quantum_region = True
    assert "atomic_quantum_region" not in str(ordinary)

    def direct_function(q: cudaq.qubit):
        h(q)

    direct = cudaq.kernel(direct_function,
                          atomic_quantum_region=True,
                          defer_compilation=False)
    assert direct.atomic_quantum_region is True
    assert direct.is_compiled()
    assert "atomic_quantum_region" in str(direct)

    @cudaq.kernel(atomic_quantum_region=True, defer_compilation=False)
    def eager(q: cudaq.qubit):
        h(q)

    assert eager.is_compiled()
    assert "atomic_quantum_region" in str(eager)

    with pytest.raises(TypeError, match="atomic_quantum_region must be a bool"):

        @cudaq.kernel(atomic_quantum_region="yes")
        def invalid():
            pass

    with pytest.raises(TypeError, match="atomic_quantum_region must be a bool"):
        cudaq.kernel(direct_function, atomic_quantum_region=1)

    with pytest.raises(TypeError, match="atomic_quantum_regoin"):
        cudaq.kernel(direct_function, atomic_quantum_regoin=True)


def test_atomic_quantum_region_state_apis():
    drawing = cudaq.draw(atomic_round_trip)
    assert drawing.count("┤ h ├") == 2

    state = np.asarray(cudaq.get_state(single_atomic_h_entry))
    np.testing.assert_allclose(state,
                               np.array([1.0, 1.0]) / np.sqrt(2),
                               atol=1e-12)

    unitary = cudaq.get_unitary(single_atomic_h_entry)
    np.testing.assert_allclose(unitary,
                               np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2),
                               atol=1e-12)


def test_atomic_quantum_region_entangling_workload():
    ordinary_drawing = cudaq.draw(ordinary_entangling_round_trip)
    assert "┤ h ├" not in ordinary_drawing
    assert "┤ x ├" not in ordinary_drawing

    atomic_drawing = cudaq.draw(atomic_entangling_round_trip_with_spectator)
    expected = ("     ╭───╮                    ╭───╮\n"
                "q0 : ┤ h ├──●──────────────●──┤ h ├\n"
                "     ╰───╯╭─┴─╮          ╭─┴─╮╰───╯\n"
                "q1 : ─────┤ x ├──●────●──┤ x ├─────\n"
                "          ╰───╯╭─┴─╮╭─┴─╮╰───╯     \n"
                "q2 : ──────────┤ x ├┤ x ├──────────\n"
                "     ╭───╮╭───╮╰───╯╰───╯          \n"
                "q3 : ┤ y ├┤ y ├────────────────────\n"
                "     ╰───╯╰───╯                    \n")
    assert atomic_drawing == expected


def test_atomic_quantum_region_builder_contract():

    def make_h_helper(atomic):
        helper, target = cudaq.make_kernel(cudaq.qubit)
        if atomic:
            helper.atomic_quantum_region()
        helper.h(target)
        return helper

    atomic = make_h_helper(True)
    assert "atomic_quantum_region" in str(atomic)

    marked_after_compile = make_h_helper(False)
    marked_after_compile.compile()
    assert "atomic_quantum_region" not in str(marked_after_compile.qkeModule)
    marked_after_compile.atomic_quantum_region()
    assert not hasattr(marked_after_compile, "qkeModule")
    marked_after_compile.compile()
    assert "atomic_quantum_region" in str(marked_after_compile.qkeModule)


def test_atomic_quantum_region_builder_preserves_entangling_workload():

    def make_workload(atomic):
        workload, q0, q1, q2 = cudaq.make_kernel(cudaq.qubit, cudaq.qubit,
                                                 cudaq.qubit)
        if atomic:
            workload.atomic_quantum_region()
        workload.h(q0)
        workload.cx(q0, q1)
        workload.cx(q1, q2)
        return workload

    def make_round_trip(workload):
        round_trip = cudaq.make_kernel()
        qubits = round_trip.qalloc(3)
        round_trip.apply_call(workload, qubits[0], qubits[1], qubits[2])
        round_trip.adjoint(workload, qubits[0], qubits[1], qubits[2])
        return round_trip

    ordinary = make_round_trip(make_workload(False))
    atomic = make_round_trip(make_workload(True))

    assert cudaq.draw(ordinary) == ""
    atomic_drawing = cudaq.draw(atomic)
    assert atomic_drawing.count("┤ h ├") == 2
    assert atomic_drawing.count("┤ x ├") == 4

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x",
                                cudaq.KrausChannel([np.kron(pauli_x, pauli_x)]),
                                num_controls=1)

    cudaq.set_target("density-matrix-cpu")
    shots = 8
    ordinary_counts = cudaq.sample(ordinary,
                                   shots_count=shots,
                                   noise_model=noise)
    atomic_counts = cudaq.sample(atomic, shots_count=shots, noise_model=noise)
    assert ordinary_counts.count("000") == shots
    assert atomic_counts.count("011") == shots
    cudaq.reset_target()


def test_atomic_quantum_region_decorator_rejects_local_operations():

    @cudaq.kernel(atomic_quantum_region=True)
    def measured(q: cudaq.qubit):
        mz(q)

    with pytest.raises(RuntimeError) as e:
        measured.compile()
    assert MEASUREMENT_ERROR in str(e.value)

    @cudaq.kernel(atomic_quantum_region=True)
    def allocated():
        q = cudaq.qubit()
        x(q)

    with pytest.raises(RuntimeError) as e:
        allocated.compile()
    assert ALLOCATION_ERROR in str(e.value)


def test_atomic_quantum_region_decorator_rejects_entry_point():

    @cudaq.kernel(atomic_quantum_region=True)
    def invalid_entry_point():
        q = cudaq.qubit()
        mz(q)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(invalid_entry_point)
    diagnostics = str(e.value)
    assert MEASUREMENT_ERROR in diagnostics
    assert ALLOCATION_ERROR in diagnostics


def test_atomic_quantum_region_decorator_rejects_transitive_measurement(capfd):

    @cudaq.kernel
    def measure_helper(q: cudaq.qubit):
        mz(q)

    @cudaq.kernel(atomic_quantum_region=True)
    def marked(q: cudaq.qubit):
        measure_helper(q)

    @cudaq.kernel
    def caller():
        q = cudaq.qubit()
        marked(q)

    with pytest.raises(RuntimeError,
                       match="Could not successfully apply kernel "
                       "specialization"):
        cudaq.sample(caller)
    assert MEASUREMENT_ERROR in capfd.readouterr().err


def test_atomic_quantum_region_positive_controls():

    # Empty atomic regions are legal.
    @cudaq.kernel(atomic_quantum_region=True)
    def empty():
        pass

    @cudaq.kernel(atomic_quantum_region=True)
    def unitary(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def caller():
        q = cudaq.qubit()
        unitary(q)
        mz(q)

    empty.compile()
    counts = cudaq.sample(caller, shots_count=1)
    assert counts.count("1") == 1


@pytest.mark.parametrize("mark_first", [False, True])
@pytest.mark.parametrize("operation", ["allocation", "measurement"])
def test_atomic_quantum_region_builder_rejects_operations(
        mark_first, operation):
    if operation == "measurement":
        kernel, q = cudaq.make_kernel(cudaq.qubit)
    else:
        kernel = cudaq.make_kernel()

    if mark_first:
        kernel.atomic_quantum_region()

    if operation == "measurement":
        kernel.mz(q)
        expected_error = MEASUREMENT_ERROR
    else:
        kernel.qalloc()
        expected_error = ALLOCATION_ERROR

    if not mark_first:
        kernel.atomic_quantum_region()

    with pytest.raises(RuntimeError) as e:
        kernel.compile()
    assert expected_error in str(e.value)


def test_atomic_quantum_region_builder_reports_each_violation():
    # Builder operations carry no source location, so the verifier cannot
    # collapse them by line the way it collapses the inlined copies of one
    # decorated kernel. Each recorded operation is reported on its own.
    allocating = cudaq.make_kernel()
    allocating.qalloc()
    allocating.qalloc()
    allocating.atomic_quantum_region()

    with pytest.raises(RuntimeError) as e:
        allocating.compile()
    assert str(e.value).count(ALLOCATION_ERROR) == 2

    measuring, q0, q1 = cudaq.make_kernel(cudaq.qubit, cudaq.qubit)
    measuring.mz(q0)
    measuring.mz(q1)
    measuring.atomic_quantum_region()

    with pytest.raises(RuntimeError) as e:
        measuring.compile()
    assert str(e.value).count(MEASUREMENT_ERROR) == 2


def test_atomic_quantum_region_sync_execution_apis():
    shots = 8
    counts = cudaq.sample(atomic_round_trip, shots_count=shots)
    assert counts.count("0") == shots

    result = cudaq.observe(atomic_round_trip, cudaq.spin.z(0))
    assert result.expectation() == pytest.approx(1.0)

    values = cudaq.run(measured_atomic_round_trip, shots_count=shots)
    assert values == [0] * shots


def test_atomic_quantum_region_async_execution_apis():
    shots = 8
    counts = cudaq.sample_async(atomic_round_trip, shots_count=shots).get()
    assert counts.count("0") == shots

    result = cudaq.observe_async(atomic_round_trip, cudaq.spin.z(0)).get()
    assert result.expectation() == pytest.approx(1.0)

    values = cudaq.run_async(measured_atomic_round_trip,
                             shots_count=shots).get()
    assert values == [0] * shots

    state = np.asarray(cudaq.get_state_async(atomic_round_trip).get())
    np.testing.assert_allclose(state, np.array([1.0, 0.0]), atol=1e-12)


def test_atomic_quantum_region_executes_preserved_gates_with_noise():
    cudaq.set_target("density-matrix-cpu")
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("h", cudaq.BitFlipChannel(1.0))
    shots = 8

    counts = cudaq.sample(atomic_round_trip,
                          shots_count=shots,
                          noise_model=noise)
    assert counts.count("1") == shots

    result = cudaq.observe(atomic_round_trip,
                           cudaq.spin.z(0),
                           noise_model=noise)
    assert result.expectation() == pytest.approx(-1.0)

    values = cudaq.run(measured_atomic_round_trip,
                       shots_count=shots,
                       noise_model=noise)
    assert values == [1] * shots
    cudaq.reset_target()
