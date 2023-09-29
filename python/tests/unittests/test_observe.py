# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq
from cudaq import spin


def assert_close(want, got, tolerance=1.e-4) -> bool:
    return abs(want - got) < tolerance


def test_observe_result():
    """
    Test the `cudaq.ObserveResult` class to ensure its member
    functions are working as expected.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    hamiltonian = spin.z(0) + spin.x(1) + spin.y(0)
    shots_count = 1000

    # Shots provided.
    observe_result = cudaq.observe(kernel, hamiltonian, shots_count=shots_count)
    # Return the entire `cudaq.SampleResult` data from observe_result.
    sample_result = observe_result.counts()
    # Get the list of all register names in the `SampleResult`.
    register_names = sample_result.register_names
    if '__global__' in register_names:
        register_names.remove('__global__')

    # Loop through each of term of the hamiltonian.
    # Note: we don't have an `__iter__` defined on cudaq.SpinOperator,
    # so this must be a bounded loop.
    # Extract the register name from the spin term and check
    # that our `SampleResult` is as expected.
    for index, sub_term in enumerate(hamiltonian):
        print(sub_term)
        # Extract the register name from the spin term.
        name = str(sub_term).split(" ")[1].rstrip()
        # Does the register exist in the measurement results?
        assert name in register_names
        # Check `cudaq.ObserveResult::counts(sub_term)`
        # against `cudaq.SampleResult::get_register_counts(sub_term_str)`
        sub_term_counts = observe_result.counts(sub_term=sub_term)
        sub_register_counts = sample_result.get_register_counts(name)
        # Check that each has `shots_count` number of total observations
        assert sum(sub_term_counts.values()) == shots_count
        assert sum(sub_register_counts.values()) == shots_count
        # Check they have the same number of elements
        assert len(sub_register_counts) == len(sub_term_counts)
        # Check `cudaq.ObserveResult::expectation_z(sub_term)`
        # against each of the the expectation values returned
        # from `cudaq.SampleResult`.
        expectation_z = observe_result.expectation_z(sub_term=sub_term)
        assert assert_close(sub_register_counts.expectation_z(), expectation_z,
                            1e-1)
        assert assert_close(sub_term_counts.expectation_z(), expectation_z,
                            1e-1)
    observe_result.dump()


@pytest.mark.parametrize("want_state, want_expectation",
                         [["0", 1.0], ["1", -1.0]])
@pytest.mark.parametrize("shots_count", [-1, 10])
def test_observe_no_params(want_state, want_expectation, shots_count):
    """
    Test `cudaq.observe()` when no parameters are provided for 
    two instances: 
    1. We leave the qubit in the 0 state and call `observe()`
    2. We rotate the qubit to the 1 state and call `observe()`

    Tests both with and without shots.
    """
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    if want_state == "0":
        # Keep qubit in the 0-state.
        # <kernel |H| kernel> = 1.0
        pass
    else:
        # Place the qubit in the 1-state.
        # <kernel |H| kernel> = -1.0
        kernel.x(qubit)

    # Measuring in the Z-basis.
    hamiltonian = spin.z(0)

    # Call `cudaq.observe()` at the specified number of shots.
    observe_result = cudaq.observe(kernel=kernel,
                                   spin_operator=hamiltonian,
                                   shots_count=shots_count)
    got_expectation = observe_result.expectation_z()
    assert want_expectation == got_expectation

    # If shots mode was enabled, check those results.
    if shots_count != -1:
        sample_result = observe_result.counts()
        register_names = sample_result.register_names
        if '__global__' in register_names:
            register_names.remove('__global__')
        # Check that each register is in the proper state.
        for index, sub_term in enumerate(hamiltonian):
            # Extract the register name from the spin term.
            got_name = str(sub_term).split(" ")[1].rstrip()
            # Pull the counts for that hamiltonian sub term from the
            # `ObserveResult::counts` overload.
            sub_term_counts = observe_result.counts(sub_term=sub_term)
            # Pull the counts for that hamiltonian sub term from the
            # `SampleResult` dictionary by its name.
            sub_register_counts = sample_result.get_register_counts(got_name)
            # Sub-term should have the same expectation as the entire
            # system.
            assert sub_term_counts.expectation_z() == want_expectation
            assert sub_register_counts.expectation_z() == want_expectation
            # Should have `shots_count` results for each.
            assert sum(sub_term_counts.values()) == shots_count
            assert sum(sub_register_counts.values()) == shots_count
            # Check the state.
            assert want_state in sub_term_counts
            assert want_state in sub_register_counts

    with pytest.raises(RuntimeError) as error:
        # Can't accept args.
        cudaq.observe(kernel, hamiltonian, 0.0)


@pytest.mark.parametrize("angle, want_state, want_expectation",
                         [[np.pi, "1", -2.0], [0.0, "0", 2.0]])
@pytest.mark.parametrize("shots_count", [-1, 10])
def test_observe_single_param(angle, want_state, want_expectation, shots_count):
    """
    Test `cudaq.observe()` on a parameterized circuit that takes
    one argument. Checks with shots mode turned both on and off.

    First round we test a kernel with rx gates by np.pi. This should
    result in the 1-state for both qubits and `<Z> = -2.0`.

    Second round we test a kernel with rx gates by 0.0. This should
    result in the 0-state for both qubits and `<Z> = 2.0`.
    """
    qubit_count = 2
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(qubit_count)

    # Rotate both qubits by the provided `theta`.
    kernel.rx(theta, qreg[0])
    kernel.rx(theta, qreg[1])

    # Measure both qubits in the Z-basis.
    hamiltonian = spin.z(0) + spin.z(1)

    # Call `cudaq.observe()` at the specified number of shots.
    observe_result = cudaq.observe(kernel,
                                   hamiltonian,
                                   angle,
                                   shots_count=shots_count)
    got_expectation = observe_result.expectation_z()
    assert want_expectation == got_expectation

    # If shots mode was enabled, check those results.
    if shots_count != -1:
        sample_result = observe_result.counts()
        register_names = sample_result.register_names
        if '__global__' in register_names:
            register_names.remove('__global__')
        # Check that each register is in the proper state.
        for index, sub_term in enumerate(hamiltonian):
            # Extract the register name from the spin term.
            got_name = str(sub_term).split(" ")[1].rstrip()
            # Pull the counts for that hamiltonian sub term from the
            # `ObserveResult::counts` overload.
            sub_term_counts = observe_result.counts(sub_term=sub_term)
            # Pull the counts for that hamiltonian sub term from the
            # `SampleResult` dictionary by its name.
            sub_register_counts = sample_result.get_register_counts(got_name)
            # Sub-term should have an expectation value proportional to the
            # expectation over the entire system.
            assert sub_term_counts.expectation_z(
            ) == want_expectation / qubit_count
            assert sub_register_counts.expectation_z(
            ) == want_expectation / qubit_count
            # Should have `shots_count` results for each.
            assert sum(sub_term_counts.values()) == shots_count
            assert sum(sub_register_counts.values()) == shots_count
            # Check the state.
            assert want_state in sub_term_counts
            assert want_state in sub_register_counts

    # Make sure that we throw an exception if user provides no/the wrong args.
    with pytest.raises(RuntimeError) as error:
        # None.
        cudaq.observe(kernel, hamiltonian)
    with pytest.raises(RuntimeError) as error:
        # Too many.
        cudaq.observe(kernel, hamiltonian, np.pi, np.pi)


@pytest.mark.parametrize(
    "angle_0, angle_1, angles, want_state, want_expectation",
    [[np.pi, np.pi, [np.pi, np.pi], "1", -4.0],
     [0.0, 0.0, [0.0, 0.0], "0", 4.0]])
@pytest.mark.parametrize("shots_count", [-1, 10])
def test_observe_multi_param(angle_0, angle_1, angles, want_state,
                             want_expectation, shots_count):
    """
    Test `cudaq.observe()` on a parameterized circuit that takes
    multiple arguments of different types. Checks with shots mode 
    turned both on and off.

    First round we test a kernel with rx gates by np.pi. This should
    result in the 1-state for all qubits and `<Z> = -4.0`.

    Second round we test a kernel with rx gates by 0.0. This should
    result in the 0-state for all qubits and `<Z> = 4.0`.
    """
    qubit_count = 4
    kernel, theta_0, theta_1, thetas = cudaq.make_kernel(float, float, list)
    qreg = kernel.qalloc(qubit_count)

    # Rotate each qubit by their respective angles.
    kernel.rx(theta_0, qreg[0])
    kernel.rx(theta_1, qreg[1])
    kernel.rx(thetas[0], qreg[2])
    kernel.rx(thetas[1], qreg[3])

    # Measure each qubit in the Z-basis.
    hamiltonian = spin.z(0) + spin.z(1) + spin.z(2) + spin.z(3)

    # Call `cudaq.observe()` at the specified number of shots.
    observe_result = cudaq.observe(kernel,
                                   hamiltonian,
                                   angle_0,
                                   angle_1,
                                   angles,
                                   shots_count=shots_count)
    got_expectation = observe_result.expectation_z()
    assert want_expectation == got_expectation

    # If shots mode was enabled, check those results.
    if shots_count != -1:
        sample_result = observe_result.counts()
        register_names = sample_result.register_names
        if '__global__' in register_names:
            register_names.remove('__global__')
        # Check that each register is in the proper state.
        for index, sub_term in enumerate(hamiltonian):
            # Extract the register name from the spin term.
            got_name = str(sub_term).split(" ")[1].rstrip()
            # Pull the counts for that hamiltonian sub term from the
            # `ObserveResult::counts` overload.
            sub_term_counts = observe_result.counts(sub_term=sub_term)
            # Pull the counts for that hamiltonian sub term from the
            # `SampleResult` dictionary by its name.
            sub_register_counts = sample_result.get_register_counts(got_name)
            # Sub-term should have an expectation value proportional to the
            # expectation over the entire system.
            assert sub_term_counts.expectation_z(
            ) == want_expectation / qubit_count
            assert sub_register_counts.expectation_z(
            ) == want_expectation / qubit_count
            # Should have `shots_count` results for each.
            assert sum(sub_term_counts.values()) == shots_count
            assert sum(sub_register_counts.values()) == shots_count
            # Check the state.
            assert want_state in sub_term_counts
            assert want_state in sub_register_counts

    # Make sure that we throw an exception if user provides no/the wrong args.
    with pytest.raises(RuntimeError) as error:
        # None.
        cudaq.observe(kernel, hamiltonian)
    with pytest.raises(RuntimeError) as error:
        # Too few.
        cudaq.observe(kernel, hamiltonian, np.pi, np.pi)
    with pytest.raises(RuntimeError) as error:
        # Too many list elements.
        cudaq.observe(kernel, hamiltonian, np.pi, np.pi, [np.pi, np.pi, np.pi])


@pytest.mark.parametrize("want_state, want_expectation",
                         [["0", 1.0], ["1", -1.0]])
@pytest.mark.parametrize("shots_count", [-1, 10])
def test_observe_async_no_params(want_state, want_expectation, shots_count):
    """
    Test `cudaq.observe_async()` when no parameters are provided for 
    two instances: 
    1. We leave the qubit in the 0 state and call `observe()`
    2. We rotate the qubit to the 1 state and call `observe()`

    Tests both with and without shots mode.
    """
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    if want_state == "0":
        # Keep qubit in the 0-state.
        # <kernel |H| kernel> = 1.0
        pass
    else:
        # Place the qubit in the 1-state.
        # <kernel |H| kernel> = -1.0
        kernel.x(qubit)

    # Measuring in the Z-basis.
    hamiltonian = spin.z(0)

    # Call `cudaq.observe()` at the specified number of shots.
    future = cudaq.observe_async(kernel=kernel,
                                 spin_operator=hamiltonian,
                                 qpu_id=0,
                                 shots_count=shots_count)
    observe_result = future.get()
    got_expectation = observe_result.expectation_z()
    assert want_expectation == got_expectation

    # Test that this throws an exception, the problem here
    # is we are on a quantum platform with 1 QPU, and we're asking
    # to run an async job on the 13th QPU with device id 12.
    with pytest.raises(Exception) as error:
        future = cudaq.observe_async(kernel, hamiltonian, qpu_id=12)


@pytest.mark.parametrize("angle, want_state, want_expectation",
                         [[np.pi, "1", -2.0], [0.0, "0", 2.0]])
@pytest.mark.parametrize("shots_count", [-1, 10])
def test_observe_async_single_param(angle, want_state, want_expectation,
                                    shots_count):
    """
    Test `cudaq.observe_async()` on a parameterized circuit that takes
    one argument. Checks with shots mode turned both on and off.

    First round we test a kernel with rx gates by np.pi. This should
    result in the 1-state for both qubits and `<Z> = -2.0`.

    Second round we test a kernel with rx gates by 0.0. This should
    result in the 0-state for both qubits and `<Z> = 2.0`.
    """
    qubit_count = 2
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(qubit_count)

    # Rotate both qubits by the provided `theta`.
    kernel.rx(theta, qreg[0])
    kernel.rx(theta, qreg[1])

    # Measure both qubits in the Z-basis.
    hamiltonian = spin.z(0) + spin.z(1)

    # Call `cudaq.observe()` at the specified number of shots.
    future = cudaq.observe_async(kernel,
                                 hamiltonian,
                                 angle,
                                 shots_count=shots_count)
    observe_result = future.get()
    got_expectation = observe_result.expectation_z()
    assert want_expectation == got_expectation

    # If shots mode was enabled, check those results.
    if shots_count != -1:
        sample_result = observe_result.counts()
        register_names = sample_result.register_names
        if '__global__' in register_names:
            register_names.remove('__global__')
        # Check that each register is in the proper state.
        for index, sub_term in enumerate(hamiltonian):
            # Extract the register name from the spin term.
            got_name = str(sub_term).split(" ")[1].rstrip()
            # Pull the counts for that hamiltonian sub term from the
            # `ObserveResult::counts` overload.
            sub_term_counts = observe_result.counts(sub_term=sub_term)
            # Pull the counts for that hamiltonian sub term from the
            # `SampleResult` dictionary by its name.
            sub_register_counts = sample_result.get_register_counts(got_name)
            # Sub-term should have an expectation value proportional to the
            # expectation over the entire system.
            assert sub_term_counts.expectation_z(
            ) == want_expectation / qubit_count
            assert sub_register_counts.expectation_z(
            ) == want_expectation / qubit_count
            # Should have `shots_count` results for each.
            assert sum(sub_term_counts.values()) == shots_count
            assert sum(sub_register_counts.values()) == shots_count
            # Check the state.
            assert want_state in sub_term_counts
            assert want_state in sub_register_counts

    # Make sure that we throw an exception if user provides no/the wrong args.
    with pytest.raises(RuntimeError) as error:
        # None.
        cudaq.observe_async(kernel, hamiltonian)
    with pytest.raises(RuntimeError) as error:
        # Too many.
        cudaq.observe_async(kernel, hamiltonian, np.pi, np.pi)
    with pytest.raises(Exception) as error:
        # Bad QPU id.
        future = cudaq.observe_async(kernel, hamiltonian, np.pi, qpu_id=12)


@pytest.mark.parametrize(
    "angle_0, angle_1, angles, want_state, want_expectation",
    [[np.pi, np.pi, [np.pi, np.pi], "1", -4.0],
     [0.0, 0.0, [0.0, 0.0], "0", 4.0]])
@pytest.mark.parametrize("shots_count", [-1, 10])
def test_observe_async_multi_param(angle_0, angle_1, angles, want_state,
                                   want_expectation, shots_count):
    """
    Test `cudaq.observe_async()` on a parameterized circuit that takes
    multiple arguments of different types. Checks with shots mode 
    turned both on and off.

    First round we test a kernel with rx gates by np.pi. This should
    result in the 1-state for all qubits and `<Z> = -4.0`.

    Second round we test a kernel with rx gates by 0.0. This should
    result in the 0-state for all qubits and `<Z> = 4.0`.
    """
    qubit_count = 4
    kernel, theta_0, theta_1, thetas = cudaq.make_kernel(float, float, list)
    qreg = kernel.qalloc(qubit_count)

    # Rotate each qubit by their respective angles.
    kernel.rx(theta_0, qreg[0])
    kernel.rx(theta_1, qreg[1])
    kernel.rx(thetas[0], qreg[2])
    kernel.rx(thetas[1], qreg[3])

    # Measure each qubit in the Z-basis.
    hamiltonian = spin.z(0) + spin.z(1) + spin.z(2) + spin.z(3)

    # Call `cudaq.observe()` at the specified number of shots.
    future = cudaq.observe_async(kernel,
                                 hamiltonian,
                                 angle_0,
                                 angle_1,
                                 angles,
                                 shots_count=shots_count)
    observe_result = future.get()
    got_expectation = observe_result.expectation_z()
    assert want_expectation == got_expectation

    # If shots mode was enabled, check those results.
    if shots_count != -1:
        sample_result = observe_result.counts()
        register_names = sample_result.register_names
        if '__global__' in register_names:
            register_names.remove('__global__')
        # Check that each register is in the proper state.
        for index, sub_term in enumerate(hamiltonian):
            # Extract the register name from the spin term.
            got_name = str(sub_term).split(" ")[1].rstrip()
            # Pull the counts for that hamiltonian sub term from the
            # `ObserveResult::counts` overload.
            sub_term_counts = observe_result.counts(sub_term=sub_term)
            # Pull the counts for that hamiltonian sub term from the
            # `SampleResult` dictionary by its name.
            sub_register_counts = sample_result.get_register_counts(got_name)
            # Sub-term should have an expectation value proportional to the
            # expectation over the entire system.
            assert sub_term_counts.expectation_z(
            ) == want_expectation / qubit_count
            assert sub_register_counts.expectation_z(
            ) == want_expectation / qubit_count
            # Should have `shots_count` results for each.
            assert sum(sub_term_counts.values()) == shots_count
            assert sum(sub_register_counts.values()) == shots_count
            # Check the state.
            assert want_state in sub_term_counts
            assert want_state in sub_register_counts

    # Make sure that we throw an exception if user provides no/the wrong args.
    with pytest.raises(RuntimeError) as error:
        # None.
        cudaq.observe_async(kernel, hamiltonian)
    with pytest.raises(RuntimeError) as error:
        # Too few.
        cudaq.observe_async(kernel, hamiltonian, np.pi, np.pi)
    with pytest.raises(Exception) as error:
        # Too many list elements.
        future = cudaq.observe_async(kernel,
                                     hamiltonian,
                                     np.pi,
                                     np.pi, [np.pi, np.pi, np.pi],
                                     qpu_id=12)
    with pytest.raises(Exception) as error:
        # Bad QPU id.
        future = cudaq.observe_async(kernel,
                                     hamiltonian,
                                     np.pi,
                                     np.pi, [np.pi, np.pi],
                                     qpu_id=12)


@pytest.mark.parametrize("angles, want_state, want_expectation",
                         [[[np.pi, np.pi, np.pi, np.pi], "1", -4.0],
                          [[0.0, 0.0, 0.0, 0.0], "0", 4.0]])
def test_observe_numpy_array(angles, want_state, want_expectation):
    """
    Tests that a numpy array can be passed to `cudaq.observe` in place of
    a list.
    """
    qubit_count = 4
    shots_count = 10
    kernel, thetas = cudaq.make_kernel(list)
    qreg = kernel.qalloc(qubit_count)

    # Rotate each qubit by their respective angles.
    kernel.rx(thetas[0], qreg[0])
    kernel.rx(thetas[1], qreg[1])
    kernel.rx(thetas[2], qreg[2])
    kernel.rx(thetas[3], qreg[3])

    print(cudaq.get_state(kernel, angles))
    # Measure each qubit in the Z-basis.
    hamiltonian = spin.z(0) + spin.z(1) + spin.z(2) + spin.z(3)

    # Convert our angles values to a numpy array from a list.
    numpy_angles = np.asarray(angles)
    # Try calling the kernel function at those angles.
    kernel(numpy_angles)

    # Call `cudaq.observe()` on the numpy array with 10 shots.
    observe_result = cudaq.observe(kernel,
                                   hamiltonian,
                                   numpy_angles,
                                   shots_count=10)
    got_expectation = observe_result.expectation_z()
    assert want_expectation == got_expectation

    # Since shots mode was enabled, check the results.
    sample_result = observe_result.counts()
    register_names = sample_result.register_names
    if '__global__' in register_names:
        register_names.remove('__global__')
        # Check that each register is in the proper state.
    for index, sub_term in enumerate(hamiltonian):
        # Extract the register name from the spin term.
        got_name = str(sub_term).split(" ")[1].rstrip()
        # Pull the counts for that hamiltonian sub term from the
        # `ObserveResult::counts` overload.
        sub_term_counts = observe_result.counts(sub_term=sub_term)
        # Pull the counts for that hamiltonian sub term from the
        # `SampleResult` dictionary by its name.
        sub_register_counts = sample_result.get_register_counts(got_name)
        # Sub-term should have an expectation value proportional to the
        # expectation over the entire system.
        assert sub_term_counts.expectation_z() == want_expectation / qubit_count
        assert sub_register_counts.expectation_z(
        ) == want_expectation / qubit_count
        # Should have `shots_count` results for each.
        assert sum(sub_term_counts.values()) == shots_count
        assert sum(sub_register_counts.values()) == shots_count
        # Check the state.
        assert want_state in sub_term_counts
        assert want_state in sub_register_counts

    # Cannot pass numpy array that is not a vector.
    bad_params = np.random.uniform(low=-np.pi, high=np.pi, size=(2, 2))
    with pytest.raises(Exception) as error:
        # Test kernel call.
        kernel(bad_params)
    with pytest.raises(Exception) as error:
        # Test observe call.
        cudaq.observe(kernel, hamiltonian, bad_params, qpu_id=0, shots_count=10)
    with pytest.raises(Exception) as error:
        # Test too few elements in array.
        bad_params = np.random.uniform(low=-np.pi, high=np.pi, size=(2,))
        cudaq.observe(kernel, hamiltonian, bad_params, qpu_id=0, shots_count=10)
    with pytest.raises(Exception) as error:
        # Test too many elements in array.
        bad_params = np.random.uniform(low=-np.pi, high=np.pi, size=(8,))
        cudaq.observe(kernel, hamiltonian, bad_params, qpu_id=0, shots_count=10)


def test_observe_n():
    """
    Test that we can broadcast the observe call over a number of argument sets
    """
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    angles = np.linspace(-np.pi, np.pi, 50)

    circuit, theta = cudaq.make_kernel(float)
    q = circuit.qalloc(2)
    circuit.x(q[0])
    circuit.ry(theta, q[1])
    circuit.cx(q[1], q[0])

    results = cudaq.observe(circuit, hamiltonian, angles)
    energies = np.array([r.expectation_z() for r in results])
    print(energies)
    expected = np.array([
        12.250289999999993, 12.746369918061657, 13.130147571153335,
        13.395321340821365, 13.537537081098929, 13.554459613462432,
        13.445811070398316, 13.213375457979938, 12.860969362537181,
        12.39437928241443, 11.821266613827706, 11.151041850950664,
        10.39471006586037, 9.56469020555809, 8.674611173202855,
        7.7390880418983645, 6.773482075596711, 5.793648497568958,
        4.815676148077341, 3.8556233060630225, 2.929254012649781,
        2.051779226024591, 1.2376070579247536, 0.5001061928414527,
        -0.14861362540995326, -0.6979004353486014, -1.1387349627411503,
        -1.4638787168353469, -1.6679928461780573, -1.7477258024084987,
        -1.701768372589711, -1.5308751764487525, -1.2378522755416648,
        -0.8275110978002891, -0.30658943401863836, 0.3163591964856498,
        1.0311059944220289, 1.8259148371286382, 2.687734985381901,
        3.6024153761738114, 4.55493698277526, 5.529659426739748,
        6.510577792485027, 7.481585427564503, 8.42673841345514,
        9.330517364258766, 10.178082254589516, 10.955516092380341,
        11.650053435508049, 12.250289999999993
    ])
    assert np.allclose(energies, expected)

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(
            1) + 9.625 - 9.625 * spin.z(2) - 3.913119 * spin.x(1) * spin.x(
                2) - 3.913119 * spin.y(1) * spin.y(2)
    kernel, theta, phi = cudaq.make_kernel(float, float)
    qubits = kernel.qalloc(3)
    kernel.x(qubits[0])
    kernel.ry(theta, qubits[1])
    kernel.ry(phi, qubits[2])
    kernel.cx(qubits[2], qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.ry(theta * -1., qubits[1])
    kernel.cx(qubits[0], qubits[1])
    kernel.cx(qubits[1], qubits[0])

    runtimeAngles = np.random.uniform(low=-np.pi, high=np.pi, size=(50, 2))
    print(runtimeAngles)

    results = cudaq.observe(kernel, hamiltonian, runtimeAngles[:, 0],
                            runtimeAngles[:, 1])
    energies = np.array([r.expectation_z() for r in results])
    print(energies)
    assert len(energies) == 50

    kernel, thetas = cudaq.make_kernel(list)
    qubits = kernel.qalloc(3)
    kernel.x(qubits[0])
    kernel.ry(thetas[0], qubits[1])
    kernel.ry(thetas[1], qubits[2])
    kernel.cx(qubits[2], qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.ry(thetas[0] * -1., qubits[1])
    kernel.cx(qubits[0], qubits[1])
    kernel.cx(qubits[1], qubits[0])

    runtimeAngles = np.random.uniform(low=-np.pi, high=np.pi, size=(50, 2))
    print(runtimeAngles)

    results = cudaq.observe(kernel, hamiltonian, runtimeAngles)
    energies = np.array([r.expectation_z() for r in results])
    print(energies)
    assert len(energies) == 50


def test_observe_list():
    hamiltonianList = [
        -2.1433 * spin.x(0) * spin.x(1), -2.1433 * spin.y(0) * spin.y(1),
        .21829 * spin.z(0), -6.125 * spin.z(1)
    ]

    circuit, theta = cudaq.make_kernel(float)
    q = circuit.qalloc(2)
    circuit.x(q[0])
    circuit.ry(theta, q[1])
    circuit.cx(q[1], q[0])

    results = cudaq.observe(circuit, hamiltonianList, .59)

    sum = 5.907
    for r in results:
        sum += r.expectation_z() * np.real(r.get_spin().get_coefficient())
    print(sum)
    want_expectation_value = -1.7487948611472093
    assert assert_close(want_expectation_value, sum, tolerance=1e-2)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
