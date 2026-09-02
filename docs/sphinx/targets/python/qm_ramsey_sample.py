import cudaq
import numpy as np

# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
cudaq.set_target(
    "quantum_machines",
    url="http://host.docker.internal:8000",
    #url="http://172.16.32.154:8000",
    #api_key="1234567890",
    qubit_mapping_mode="backend",
    executor="sim")


# `wait` is implemented by the backend rather than the compiler.
@cudaq.extern_kernel
def wait(duration: float, q: cudaq.qubit) -> None:
    ...


# A Ramsey sequence. Sweeping the delay traces out the qubit's detuning.
@cudaq.kernel
def ramsey_single(wait_duration: float):
    qubit = cudaq.qubit()

    reset(qubit)
    rx(np.pi / 2, qubit)
    wait(wait_duration, qubit)
    rx(np.pi / 2, qubit)
    mz(qubit)


# The duration folds into the payload as a constant, so each point of the sweep
# is a separate compilation and submission.
shots_count = 1000
for wait_duration in [x * 1.0 for x in range(1, 10)]:
    result = cudaq.sample(ramsey_single, wait_duration, shots_count=shots_count)
    print(f"wait {wait_duration:5.1f} -> "
          f"P(1) = {result.count('1') / shots_count:.3f}")
