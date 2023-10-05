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


@pytest.mark.parametrize("qubit_count", [1, 5, 9])
@pytest.mark.parametrize("shots_count", [10, 100, 1000])
def test_sample_result_single_register(qubit_count, shots_count):
    """
    Tests the `SampleResult` data-type on a simple circuit
    of varying sizes.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(qubit_count)
    # Place every qubit in the 1-state.
    kernel.x(qreg)
    kernel.mz(qreg)

    # Get the QPU result from a call to `sample`.
    # Check at a varying number of shots.
    sample_result = cudaq.sample(kernel, shots_count=shots_count)

    # Check for correctness on each member function of `SampleResult`
    want_bitstring = "1" * qubit_count
    # `::dump()`
    sample_result.dump()
    # `__str__`
    print(str(sample_result))
    # `__iter__`
    for sub_counts in sample_result:
        # Should just be the `want_bitstring`
        assert sub_counts == want_bitstring
    # `__getitem__`
    # The `want_bitstring` should have `shots_count` observations.
    assert sample_result[want_bitstring] == shots_count
    # Should have 1 global register.
    assert sample_result.register_names == ["__global__"]

    # The full `SampleResult` and the extracted counts for the
    # global register should be the same in this case.
    for counts in [
            sample_result,
            sample_result.get_register_counts("__global__")
    ]:
        # `__len__`
        # Should have only measured 1 different state.
        assert len(counts) == 1
        # `expectation_z`
        # The `qubit_count` is always odd so we should always have
        # an expectation of -1. for the 1-state.
        assert counts.expectation_z() == -1.
        # `probability`
        assert counts.probability(want_bitstring) == 1.
        # `most_probable`
        assert counts.most_probable() == want_bitstring
        # `count`
        assert counts.count(want_bitstring) == shots_count
        # Check the results marginalized over each qubit.
        for qubit in range(qubit_count):
            marginal_counts = counts.get_marginal_counts([qubit])
            print(marginal_counts)
            assert marginal_counts.expectation_z() == -1.
            # Should be in the 1-state.
            assert marginal_counts.probability("1") == 1
            assert marginal_counts.most_probable() == "1"
    # `get_sequential_data`
    # In this case, should just contain the single bitstring in a list.
    assert sample_result.get_sequential_data() == [want_bitstring] * shots_count

    # `::items()`
    for key, value in sample_result.items():
        assert key == want_bitstring
        assert value == shots_count

    # `::values()`
    for value in sample_result.values():
        assert value == shots_count

    # `::clear()`
    sample_result.clear()
    # Counts should now be empty.
    assert str(sample_result) == "{ }\n"

    with pytest.raises(RuntimeError) as error:
        # Too many args.
        result = cudaq.sample(kernel, 0.0)


@pytest.mark.parametrize("qubit_count", [3, 5, 9])
@pytest.mark.parametrize("shots_count", [10, 100, 1000])
def test_sample_result_single_register_float_param(qubit_count, shots_count):
    """
    Tests the `SampleResult` data-type on a simple circuit
    of varying sizes. The circuit in this case is parameterized
    by a single float value.
    """
    kernel, angle = cudaq.make_kernel(float)
    qreg = kernel.qalloc(qubit_count)
    # Place every qubit in the 1-state, parameterized by
    # the `angle`.
    for index in range(qubit_count):
        kernel.rx(angle, qreg[index])
    kernel.mz(qreg)

    # Get the QPU result from a call to `sample`, at the concrete
    # angle of `np.pi`. Should be equivalent to the previous test
    # case.
    # Check at a varying number of shots.
    sample_result = cudaq.sample(kernel, np.pi, shots_count=shots_count)

    # Check for correctness on each member function of `SampleResult`
    want_bitstring = "1" * qubit_count
    # `::dump()`
    sample_result.dump()
    # `__str__`
    print(str(sample_result))
    # `__iter__`
    for sub_counts in sample_result:
        # Should just be the `want_bitstring`
        assert sub_counts == want_bitstring
    # `__getitem__`
    # The `want_bitstring` should have `shots_count` observations.
    assert sample_result[want_bitstring] == shots_count
    # Should have 1 global register.
    assert sample_result.register_names == ["__global__"]

    # The full `SampleResult` and the extracted counts for the
    # global register should be the same in this case.
    for counts in [
            sample_result,
            sample_result.get_register_counts("__global__")
    ]:
        # `__len__`
        # Should have only measured 1 different state.
        assert len(counts) == 1
        # `expectation_z`
        # The `qubit_count` is always odd so we should always have
        # an expectation of -1. for the 1-state.
        assert counts.expectation_z() == -1.
        # `probability`
        assert counts.probability(want_bitstring) == 1.
        # `most_probable`
        assert counts.most_probable() == want_bitstring
        # `count`
        assert counts.count(want_bitstring) == shots_count
        # Check the results marginalized over each qubit.
        for qubit in range(qubit_count):
            marginal_counts = counts.get_marginal_counts([qubit])
            print(marginal_counts)
            assert marginal_counts.expectation_z() == -1.
            # Should be in the 1-state.
            assert marginal_counts.probability("1") == 1
            assert marginal_counts.most_probable() == "1"
    # `get_sequential_data`
    # In this case, should just contain the single bitstring in a list.
    assert sample_result.get_sequential_data() == [want_bitstring] * shots_count

    # `::items()`
    for key, value in sample_result.items():
        assert key == want_bitstring
        assert value == shots_count

    # `::values()`
    for value in sample_result.values():
        assert value == shots_count

    # `::clear()`
    sample_result.clear()
    # Counts should now be empty.
    assert str(sample_result) == "{ }\n"

    with pytest.raises(RuntimeError) as error:
        # Too few args.
        result = cudaq.sample(kernel)


@pytest.mark.parametrize("qubit_count", [3, 5, 9])
@pytest.mark.parametrize("shots_count", [10, 100, 1000])
def test_sample_result_single_register_list_param(qubit_count, shots_count):
    """
    Tests the `SampleResult` data-type on a simple circuit
    of varying sizes. The circuit in this case is parameterized
    by a list.
    """
    kernel, angles = cudaq.make_kernel(list)
    qreg = kernel.qalloc(qubit_count)
    # Place every qubit in the 1-state, parameterized by
    # the `angle`.
    for index in range(qubit_count):
        kernel.rx(angles[0], qreg[index])
    kernel.mz(qreg)

    # Get the QPU result from a call to `sample`, at the concrete
    # angle of `np.pi`. Should be equivalent to the previous test
    # case.
    # Check at a varying number of shots.
    sample_result = cudaq.sample(kernel, [np.pi], shots_count=shots_count)

    # Check for correctness on each member function of `SampleResult`
    want_bitstring = "1" * qubit_count
    # `::dump()`
    sample_result.dump()
    # `__str__`
    print(str(sample_result))
    # `__iter__`
    for sub_counts in sample_result:
        # Should just be the `want_bitstring`
        assert sub_counts == want_bitstring
    # `__getitem__`
    # The `want_bitstring` should have `shots_count` observations.
    assert sample_result[want_bitstring] == shots_count
    # Should have 1 global register.
    assert sample_result.register_names == ["__global__"]

    # The full `SampleResult` and the extracted counts for the
    # global register should be the same in this case.
    for counts in [
            sample_result,
            sample_result.get_register_counts("__global__")
    ]:
        # `__len__`
        # Should have only measured 1 different state.
        assert len(counts) == 1
        # `expectation_z`
        # The `qubit_count` is always odd so we should always have
        # an expectation of -1. for the 1-state.
        assert counts.expectation_z() == -1.
        # `probability`
        assert counts.probability(want_bitstring) == 1.
        # `most_probable`
        assert counts.most_probable() == want_bitstring
        # `count`
        assert counts.count(want_bitstring) == shots_count
        # Check the results marginalized over each qubit.
        for qubit in range(qubit_count):
            marginal_counts = counts.get_marginal_counts([qubit])
            print(marginal_counts)
            assert marginal_counts.expectation_z() == -1.
            # Should be in the 1-state.
            assert marginal_counts.probability("1") == 1
            assert marginal_counts.most_probable() == "1"
    # `get_sequential_data`
    # In this case, should just contain the single bitstring in a list.
    assert sample_result.get_sequential_data() == [want_bitstring] * shots_count

    # `::items()`
    for key, value in sample_result.items():
        assert key == want_bitstring
        assert value == shots_count

    # `::values()`
    for value in sample_result.values():
        assert value == shots_count

    # `::clear()`
    sample_result.clear()
    # Counts should now be empty.
    assert str(sample_result) == "{ }\n"

    with pytest.raises(RuntimeError) as error:
        # Wrong arg type.
        result = cudaq.sample(kernel, 0.0)


@pytest.mark.skip(
    reason=
    "Mid-circuit measurements not currently supported without the use of `c_if`."
)
@pytest.mark.parametrize("qubit_count", [1, 5, 9])
@pytest.mark.parametrize("shots_count", [10, 100, 1000])
def test_sample_result_multiple_registers(qubit_count, shots_count):
    """
    Tests the `SampleResult` data-type on a simple circuit
    of varying sizes. The circuit provides a `register_name`
    on the measurements in this case.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(qubit_count)
    # Place every qubit in the 1-state.
    kernel.x(qreg)
    # Name the measurement register.
    kernel.mz(qreg, register_name="test_measurement")

    # Get the QPU result from a call to `sample`.
    # Check at a varying number of shots.
    sample_result = cudaq.sample(kernel, shots_count=shots_count)

    # Check for correctness on each member function of `SampleResult`
    want_bitstring = "1" * qubit_count
    # `::dump()`
    sample_result.dump()
    # `__str__`
    print(str(sample_result))
    # `__iter__`
    for sub_counts in sample_result:
        # Should just be the `want_bitstring`
        assert sub_counts == want_bitstring
    # `__getitem__`
    # The `want_bitstring` should have `shots_count` observations.
    assert sample_result[want_bitstring] == shots_count

    # TODO: once mid-circuit measurements are supported, finish out
    # the rest of this test.


@pytest.mark.parametrize("shots_count", [10, 100])
def test_sample_result_observe(shots_count):
    """
    Test `cudaq.SampleResult` as its returned from a call
    to `cudaq.observe()`.
    """
    qubit_count = 3
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(qubit_count)
    kernel.x(qreg)
    hamiltonian = spin.z(0) + spin.z(1) + spin.z(2)
    want_expectation = -3.0
    want_state = "111"

    # Test via call to `cudaq.sample()`.
    observe_result = cudaq.observe(kernel, hamiltonian, shots_count=shots_count)
    # Return the entire `cudaq.SampleResult` data from observe_result.
    sample_result = observe_result.counts()

    # If shots mode was enabled, check those results.
    if shots_count != -1:
        sample_result = observe_result.counts()
        sample_result.dump()
        # Should just have 3 measurement registers, one for each spin term.
        want_register_names = ["IIZ", "IZI", "ZII"]
        got_register_names = sample_result.register_names
        if '__global__' in got_register_names:
            got_register_names.remove('__global__')
        for want_name in want_register_names:
            assert want_name in got_register_names

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
            # Sub-term should have the an expectation proportional to the entire
            # system.
            assert sub_term_counts.expectation_z(
            ) == want_expectation / qubit_count
            assert sub_register_counts.expectation_z(
            ) == want_expectation / qubit_count
            # Should have `shots_count` results for each.
            assert sum(sub_term_counts.values()) == shots_count
            assert sum(sub_register_counts.values()) == shots_count
            print(sub_term_counts)
            # Check the state.
            assert "1" in sub_term_counts
            assert "1" in sub_register_counts

    sample_result.dump()
    # `::items()`
    for key, value in sample_result.items():
        assert key == "1"
        assert value == shots_count

    # `::values()`
    for value in sample_result.values():
        assert value == shots_count

    # `::clear()`
    sample_result.clear()
    # Counts should now be empty.
    assert str(sample_result) == "{ }\n"


def test_sample_async():
    """Tests `cudaq.sample_async` on a simple kernel with no args."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    # Invalid QPU
    with pytest.raises(Exception) as error:
        future = cudaq.sample_async(kernel, qpu_id=1)

    # Default 0th qpu
    future = cudaq.sample_async(kernel)
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    # Can specify qpu id
    future = cudaq.sample_async(kernel, qpu_id=0)
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    with pytest.raises(Exception) as error:
        # Invalid qpu_id type.
        result = cudaq.sample_async(kernel, qpu_id=12)


def test_sample_async_params():
    """Tests `cudaq.sample_async` on a simple kernel that accepts args."""
    kernel, theta, phi = cudaq.make_kernel(float, float)
    qubits = kernel.qalloc(2)
    kernel.rx(theta, qubits[0])
    kernel.ry(phi, qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    # Creating the bell state with rx and ry instead of hadamard
    # need a pi rotation and a pi/2 rotation
    future = cudaq.sample_async(kernel, np.pi, np.pi / 2.)
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    with pytest.raises(Exception) as error:
        # Invalid qpu_id type.
        result = cudaq.sample_async(kernel, 0.0, 0.0, qpu_id=12)


def test_sample_marginalize():
    """
    A more thorough test of the functionality of `SampleResult::get_marginal_counts`.
    """
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(4)
    # Place register in `0101` state.
    kernel.x(qubits[1])
    kernel.x(qubits[3])

    want_bitstring = "0101"

    sample_result = cudaq.sample(kernel)

    # Marginalize over each qubit and check that it's correct.
    for qubit in range(4):
        marginal_result = sample_result.get_marginal_counts([qubit])
        # Check the individual qubits state.
        assert marginal_result.most_probable() == want_bitstring[qubit]

    # Marginalize the qubit over pairs and check if correct.
    qubit = 0
    for other_qubit in [1, 2, 3]:
        new_bitstring = want_bitstring[qubit] + want_bitstring[other_qubit]
        # Check that qubit paired with every other qubit.
        marginal_result = sample_result.get_marginal_counts(
            [qubit, other_qubit])
        assert marginal_result.most_probable() == new_bitstring

    # Marginalize over the first 3 qubits.
    marginal_result = sample_result.get_marginal_counts([0, 1, 2])
    assert marginal_result.most_probable() == "010"

    # Marginalize over the last 3 qubits.
    marginal_result = sample_result.get_marginal_counts([1, 2, 3])
    assert marginal_result.most_probable() == "101"


def test_swap_2q():
    """
    Tests the simple case of swapping the states of two qubits.
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # Place qubit 0 in the 1-state.
    kernel.x(qubit_0)
    # Swap states with qubit 1.
    kernel.swap(qubit_0, qubit_1)
    # Check their states.
    kernel.mz(qreg)

    want_state = "01"
    result = cudaq.sample(kernel)
    assert (want_state in result)
    assert (result[want_state] == 1000)


def test_qubit_reset():
    """
    Basic test that we can apply a qubit reset.
    """
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.x(qubit)
    kernel.reset(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    assert (len(counts) == 1)
    assert ('0' in counts)


def test_qreg_reset():
    """
    Basic test that we can apply a qreg reset.
    """
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.x(qubits)
    kernel.reset(qubits)
    kernel.mz(qubits)

    counts = cudaq.sample(kernel)
    assert (len(counts) == 1)
    assert ('00' in counts)


def test_for_loop():
    """
    Test that we can build a kernel expression with a for loop.
    """
    circuit, inSize = cudaq.make_kernel(int)
    qubits = circuit.qalloc(inSize)
    circuit.h(qubits[0])
    # can pass concrete integers for both
    circuit.for_loop(0, inSize - 1,
                     lambda index: circuit.cx(qubits[index], qubits[index + 1]))
    print(circuit)
    counts = cudaq.sample(circuit, 5)
    assert len(counts) == 2
    assert '0' * 5 in counts
    assert '1' * 5 in counts

    counts.dump()


def test_sample_n():
    """
    Test that we can broadcast the sample call over a number of argument sets
    """
    circuit, inSize = cudaq.make_kernel(int)
    qubits = circuit.qalloc(inSize)
    circuit.h(qubits[0])
    # can pass concrete integers for both
    circuit.for_loop(0, inSize - 1,
                     lambda index: circuit.cx(qubits[index], qubits[index + 1]))
    # circuit.mz(qubits)
    print(circuit)

    allCounts = cudaq.sample(circuit, [3, 4, 5, 6, 7])
    first0 = '000'
    first1 = '111'
    for c in allCounts:
        print(c)
        assert first0 in c and first1 in c
        first0 += '0'
        first1 += '1'

    testNpArray = np.random.randint(3, high=8, size=6)
    print(testNpArray)
    allCounts = cudaq.sample(circuit, testNpArray)
    for i, c in enumerate(allCounts):
        print(c)
        assert '0' * testNpArray[i] in c and '1' * testNpArray[i] in c

    circuit, angles = cudaq.make_kernel(list)
    q = circuit.qalloc(2)
    circuit.rx(angles[0], q[0])
    circuit.ry(angles[1], q[0])
    circuit.cx(q[0], q[1])

    runtimeAngles = np.random.uniform(low=1.0, high=np.pi, size=(10, 2))
    print(runtimeAngles)
    allCounts = cudaq.sample(circuit, runtimeAngles)
    for i, c in enumerate(allCounts):
        print(runtimeAngles[i, :], c)
        assert len(c) == 2


def test_index_out_of_range():
    """
    Test the `cudaq.kernel` for out-of-range errors
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 3.
    qreg = kernel.qalloc(3)
    kernel.x(qreg[99])

    with pytest.raises(Exception) as error:
        # Index out of range
        result = cudaq.sample(kernel)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
