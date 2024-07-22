# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import os
import sys
import requests
import subprocess
import time
import psutil
import numpy as np

import cudaq
from cudaq import spin
import numpy as np

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="This feature is supported on Python 3.9+")


def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


def kill_proc_and_child_processes(parent_proc: subprocess.Popen):
    try:
        parent = psutil.Process(parent_proc.pid)
    except psutil.NoSuchProcess:
        return

    # Try to kill the children processes, giving them 3 seconds for a graceful
    # exit, and then a forceful kill after that.
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue

    _, still_alive = psutil.wait_procs(children, timeout=3)

    for child in still_alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            continue

    # Now kill the parent process
    parent.terminate()
    _, still_alive = psutil.wait_procs([parent], timeout=3)
    for p in still_alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            continue


def wait_until_port_active(port: int) -> bool:
    port_up = False
    retries = 0
    port_url = 'http://localhost:' + str(port)
    while (not port_up):
        try:
            ping_response = requests.get(port_url)
            port_up = (ping_response.status_code == 200)
        except:
            port_up = False
        if not port_up:
            retries += 1
            if retries > 100:
                print("EXIT: TOO MANY RETRIES!")
                return False
            time.sleep(0.1)
    return True


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    os.environ['CUDAQ_SER_CODE_EXEC'] = '1'
    cudaq_qpud = os.path.dirname(cudaq.__file__) + "/../bin/cudaq-qpud.py"
    nvqc_proxy = os.path.dirname(cudaq.__file__) + "/../bin/nvqc_proxy.py"
    p1 = subprocess.Popen([sys.executable, nvqc_proxy])
    p2 = subprocess.Popen([sys.executable, cudaq_qpud, '--port', '3031'])
    cudaq.set_target("remote-mqpu", url="localhost:3030")
    proxy_up = wait_until_port_active(3030)
    qpud_up = wait_until_port_active(3031)

    # Shutdown servers if either one fails to come up. The tests will fail
    # downstream.
    if not proxy_up or not qpud_up:
        kill_proc_and_child_processes(p1)
        kill_proc_and_child_processes(p2)

    yield
    cudaq.reset_target()
    kill_proc_and_child_processes(p1)
    kill_proc_and_child_processes(p2)


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_setup():
    target = cudaq.get_target()
    numQpus = target.num_qpus()
    assert numQpus == 1


@skipIfPythonLessThan39
def test_optimizer():
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Verify that variables can be captured by kernels
    testVar = 0

    @cudaq.kernel
    def kernel(angles: list[float]):
        qvector = cudaq.qvector(2)
        x(qvector[0])
        ry(angles[0] + testVar, qvector[1])
        x.ctrl(qvector[1], qvector[0])

    optimizer = cudaq.optimizers.Adam()
    gradient = cudaq.gradients.CentralDifference()

    def objective_function(parameter_vector: list[float],
                           hamiltonian=hamiltonian,
                           gradient_strategy=gradient,
                           kernel=kernel) -> tuple[float, list[float]]:
        get_result = lambda parameter_vector: cudaq.observe(
            kernel, hamiltonian, parameter_vector).expectation()
        cost = get_result(parameter_vector)
        gradient_vector = gradient_strategy.compute(parameter_vector,
                                                    get_result, cost)
        return cost, gradient_vector

    energy, parameter = optimizer.optimize(dimensions=1,
                                           function=objective_function)
    print(f"\nminimized <H> = {round(energy,16)}")
    print(f"optimal theta = {round(parameter[0],16)}")
    assert assert_close(energy, -1.7483830311526454, 1e-3)
    assert assert_close(parameter[0], 0.5840908448487905, 1e-3)


@skipIfPythonLessThan39
def test_optimizer_nested_kernels():
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel
    def kernelA(qvector: cudaq.qview):
        x(qvector[0])

    @cudaq.kernel
    def kernelB(angles: list[float]):
        qvector = cudaq.qvector(2)
        # This x() is done in a nested kernel
        # x(qvector[0])
        kernelA(qvector)
        ry(angles[0], qvector[1])
        x.ctrl(qvector[1], qvector[0])

    optimizer = cudaq.optimizers.Adam()
    gradient = cudaq.gradients.CentralDifference()

    def nested_obj_func():
        print('Calling nested_obj_func')

    def objective_function(parameter_vector: list[float],
                           hamiltonian=hamiltonian,
                           gradient_strategy=gradient,
                           kernel=kernelB) -> tuple[float, list[float]]:
        nested_obj_func()  # gratuitous call to a nested function (for testing)

        def another_nested_obj_func():
            print('I am in another_nested_obj_func')

        another_nested_obj_func()
        get_result = lambda parameter_vector: cudaq.observe(
            kernel, hamiltonian, parameter_vector).expectation()
        cost = get_result(parameter_vector)
        gradient_vector = gradient_strategy.compute(parameter_vector,
                                                    get_result, cost)
        return cost, gradient_vector

    energy, parameter = optimizer.optimize(dimensions=1,
                                           function=objective_function)
    print(f"\nminimized <H> = {round(energy,16)}")
    print(f"optimal theta = {round(parameter[0],16)}")
    assert assert_close(energy, -1.7483830311526454, 1e-3)
    assert assert_close(parameter[0], 0.5840908448487905, 1e-3)


@skipIfPythonLessThan39
@pytest.mark.parametrize(
    "optimizer", [cudaq.optimizers.COBYLA(),
                  cudaq.optimizers.NelderMead()])
def test_simple_vqe(optimizer):
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel
    def kernel(angles: list[float]):
        qvector = cudaq.qvector(2)
        x(qvector[0])
        ry(angles[0], qvector[1])
        x.ctrl(qvector[1], qvector[0])

    energy, parameter = cudaq.vqe(kernel=kernel,
                                  spin_operator=hamiltonian,
                                  optimizer=optimizer,
                                  parameter_count=1)

    print(f"\nminimized <H> = {round(energy,16)}")
    print(f"optimal theta = {round(parameter[0],16)}")
    want_expectation_value = -1.7487948611472093
    want_optimal_parameters = [0.59]
    assert assert_close(want_expectation_value, energy, tolerance=1e-2)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 parameter))


@skipIfPythonLessThan39
@pytest.mark.parametrize(
    "optimizer", [cudaq.optimizers.COBYLA(),
                  cudaq.optimizers.NelderMead()])
def test_simple_vqe_nested_kernels(optimizer):
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel
    def kernelA(qvector: cudaq.qview):
        x(qvector[0])

    @cudaq.kernel
    def kernelB(angles: list[float]):
        qvector = cudaq.qvector(2)
        # This x() is done in a nested kernel
        # x(qvector[0])
        kernelA(qvector)
        ry(angles[0], qvector[1])
        x.ctrl(qvector[1], qvector[0])

    energy, parameter = cudaq.vqe(kernel=kernelB,
                                  spin_operator=hamiltonian,
                                  optimizer=optimizer,
                                  parameter_count=1)

    print(f"\nminimized <H> = {round(energy,16)}")
    print(f"optimal theta = {round(parameter[0],16)}")
    want_expectation_value = -1.7487948611472093
    want_optimal_parameters = [0.59]
    assert assert_close(want_expectation_value, energy, tolerance=1e-2)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 parameter))


@skipIfPythonLessThan39
def test_complex_vqe_inline_lambda():
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel
    def kernel(angles: list[float], num_qubits: int):
        qvector = cudaq.qvector(num_qubits)
        x(qvector[0])
        ry(angles[0], qvector[1])
        x.ctrl(qvector[1], qvector[0])

    optimizer = cudaq.optimizers.Adam()
    grad = cudaq.gradients.CentralDifference()

    num_qubits = 2
    energy, parameter = cudaq.vqe(kernel=kernel,
                                  gradient_strategy=grad,
                                  spin_operator=hamiltonian,
                                  optimizer=optimizer,
                                  argument_mapper=lambda x: (x, num_qubits),
                                  parameter_count=1)

    print(f"\nminimized <H> = {round(energy,16)}")
    print(f"optimal theta = {round(parameter[0],16)}")
    assert assert_close(energy, -1.7488648395275948, 1e-3)
    assert assert_close(parameter[0], 0.5840908448487905, 1e-3)


@skipIfPythonLessThan39
def test_vqe_perf_warning():
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel
    def kernel(num_qubits: int, angles: list[float]):
        qvector = cudaq.qvector(num_qubits)
        x(qvector[0])
        ry(angles[0], qvector[1])
        x.ctrl(qvector[1], qvector[0])

    optimizer = cudaq.optimizers.Adam()
    grad = cudaq.gradients.CentralDifference()

    num_qubits = 2
    with pytest.raises(RuntimeError) as error:
        energy, parameter = cudaq.vqe(kernel=kernel,
                                      gradient_strategy=grad,
                                      spin_operator=hamiltonian,
                                      optimizer=optimizer,
                                      argument_mapper=lambda x: (num_qubits, x),
                                      parameter_count=1)


# This is a helper function used by parameterized tests below.
@pytest.mark.skip
def test_complex_vqe_named_lambda(optimizer, gradient):
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel
    def kernel(angles: list[float], num_qubits: int):
        qvector = cudaq.qvector(num_qubits)
        x(qvector[0])
        ry(angles[0], qvector[1])
        x.ctrl(qvector[1], qvector[0])

    num_qubits = 2
    arg_mapper = lambda x: (x, num_qubits)
    energy, parameter = cudaq.vqe(kernel=kernel,
                                  gradient_strategy=gradient,
                                  spin_operator=hamiltonian,
                                  optimizer=optimizer,
                                  argument_mapper=arg_mapper,
                                  parameter_count=1)

    print(f"\nminimized <H> = {round(energy,16)}")
    print(f"optimal theta = {round(parameter[0],16)}")
    want_expectation_value = -1.7487948611472093
    want_optimal_parameters = [0.59]
    assert assert_close(want_expectation_value, energy, tolerance=1e-2)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 parameter))


@skipIfPythonLessThan39
@pytest.mark.parametrize("optimizer", [
    cudaq.optimizers.LBFGS(),
    cudaq.optimizers.Adam(),
    cudaq.optimizers.GradientDescent(),
    cudaq.optimizers.SGD(),
])
def test_complex_vqe_named_lambda_sweep_opt(optimizer):
    test_complex_vqe_named_lambda(optimizer,
                                  cudaq.gradients.CentralDifference())


@skipIfPythonLessThan39
@pytest.mark.parametrize("gradient", [
    cudaq.gradients.CentralDifference(),
    cudaq.gradients.ParameterShift(),
    cudaq.gradients.ForwardDifference()
])
def test_complex_vqe_named_lambda_sweep_grad(gradient):
    test_complex_vqe_named_lambda(cudaq.optimizers.Adam(), gradient)

@skipIfPythonLessThan39
def test_state_preparation():

    @cudaq.kernel
    def kernel(vec: list[complex]):
        qubits = cudaq.qvector(vec)

    state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
    counts = cudaq.sample(kernel, state)
    assert '00' in counts
    assert '10' in counts
    assert not '01' in counts
    assert not '11' in counts

@skipIfPythonLessThan39
def test_state_preparation_builder():
    kernel, state = cudaq.make_kernel(list[complex])
    qubits = kernel.qalloc(state)

    state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
    counts = cudaq.sample(kernel, state)
    assert '00' in counts
    assert '10' in counts
    assert not '01' in counts
    assert not '11' in counts

@skipIfPythonLessThan39
@pytest.mark.skip(reason="https://github.com/NVIDIA/cuda-quantum/issues/1924")
def test_arbitrary_unitary_synthesis():
    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def bell(angles: list[float]):
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])
        ry(angles[0], qubits[1])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    optimizer = cudaq.optimizers.Adam()
    energy, parameter = cudaq.vqe(kernel=bell,
                                  spin_operator=hamiltonian,
                                  optimizer=optimizer,
                                  parameter_count=1)
    print(f"\nminimized <H> = {round(energy,16)}")
    print(f"optimal theta = {round(parameter[0],16)}")


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
