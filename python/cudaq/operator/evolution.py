from __future__ import annotations
import numpy, scipy, sys, uuid
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .expressions import Operator
from .helpers import _OperatorHelpers
from .schedule import Schedule
from ..kernel.register_op import register_operation
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
# FIXME: PyKernelDecorator is documented but not accessible in the API due to shadowing by the kernel module.
#from ..kernel.kernel_decorator import PyKernelDecorator
from ..kernel.kernel_builder import PyKernel, make_kernel
from ..runtime.observe import observe
from .integrator import BaseIntegrator
from .cuso_solver import evolve_me

noise = cudaq_runtime.NoiseModel()


def _register_evolution_kernels(
    hamiltonian: Operator,
    schedule: Schedule,
    collapse_operators: Sequence[Operator] | None = None
) -> Generator[(str, Mapping[str, NumericType])]:
    # Evolution kernels can only be defined for qubits.
    dimensions = dict([(i, 2) for i in hamiltonian.degrees])
    qubits = [i for i in range(len(hamiltonian.degrees))]
    # We could make operators hashable and try to use that to do some kernel caching,
    # but there is no guarantee that if the hash is equal, the operator is equal.
    # Overall it feels like a better choice to just take a uuid here.
    kernel_base_name = "".join(filter(str.isalnum, str(uuid.uuid4())))
    current_time = 0.0
    for step_idx, parameters in enumerate(schedule):
        kernel_name = f"evolve_{kernel_base_name}_{step_idx}"
        if step_idx == 0:
            evolution_matrix = numpy.eye(2**len(qubits), dtype=numpy.complex128)
            register_operation(kernel_name, evolution_matrix)
            current_time = schedule.current_step
        else:
            op_matrix = hamiltonian.to_matrix(dimensions, **parameters)
            dt = schedule.current_step - current_time
            current_time = schedule.current_step
            evolution_matrix = scipy.linalg.expm(-1j * op_matrix * dt)
            register_operation(kernel_name, evolution_matrix)

            if collapse_operators is not None and len(
                    collapse_operators) > 0:
                global noise
                for collapse_op in collapse_operators:
                    L = collapse_op.to_matrix(dimensions)
                    G = -0.5 * numpy.dot(L.conj().T, L)
                    M0 = G * dt + numpy.eye(2**len(dimensions))
                    M1 = numpy.sqrt(dt) * L
                    try:
                        noise.add_channel(kernel_name, qubits,
                                      cudaq_runtime.KrausChannel([M0, M1]))
                    except Exception as e:
                        print(f"Error adding noise channel, perhaps the time step is too large.")
                        raise e

        yield kernel_name, parameters

def _evolution_kernel(num_qubits: int, operation_names: Iterable[str]) -> PyKernel:
    evolution, initial_state = make_kernel(cudaq_runtime.State)
    qs = evolution.qalloc(initial_state)
    for operation_name in operation_names:
        # FIXME: #(*qs) causes infinite loop
        # FIXME: It would be nice if a registered operation could take a vector of qubits?
        targets = [qs[i] for i in range(num_qubits)]
        evolution.__getattr__(f"{operation_name}")(*targets)
    return evolution

def _create_kernel(name: str, 
                   hamiltonian: Operator, 
                   schedule: Schedule,
                   collapse_operators: Sequence[Operator] | None = None) -> tuple[PyKernel, Mapping[str, NumericType]]:
    evolution = _register_evolution_kernels(hamiltonian, schedule, collapse_operators)
    operation_names = []
    for operation_name, parameters in evolution:
        operation_names.append(operation_name)
    return _evolution_kernel(len(hamiltonian.degrees), operation_names), parameters

def _create_kernels(name: str, 
                    hamiltonian: Operator, 
                    schedule: Schedule,
                    collapse_operators: Sequence[Operator] | None = None) -> Generator[tuple[PyKernel, Mapping[str, NumericType]]]:
    evolution = _register_evolution_kernels(hamiltonian, schedule, collapse_operators)
    for op_idx, (operation_name, parameters) in enumerate(evolution):
        yield _evolution_kernel(len(hamiltonian.degrees), [operation_name]), parameters


# Top level API for the CUDA-Q master equation solver.
def evolve(
    hamiltonian: Operator,
    dimensions: Mapping[int, int],
    schedule: Schedule,
    initial_state: cudaq_runtime.State | Sequence[cudaq_runtime.State],
    collapse_operators: Sequence[Operator] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results=False,
    integrator: Optional[BaseIntegrator] = None
) -> cudaq_runtime.EvolveResult | Sequence[cudaq_runtime.EvolveResult]:
    """
    Computes the time evolution of one or more initial state(s) under the defined 
    operator(s). 

    Arguments:
        hamiltonian: Operator that describes the behavior of a quantum system
            without noise.
        dimensions: A mapping that specifies the number of levels, that is
            the dimension, of each degree of freedom that any of the operator 
            arguments acts on.
        schedule: A sequence that generates a mapping of keyword arguments 
            to their respective value. The keyword arguments are the parameters
            needed to evaluate any of the operators passed to `evolve`.
            All required parameters for evaluating an operator and their
            documentation, if available, can be queried by accessing the
            `parameter` property of the operator.
        initial_state: A single state or a sequence of states of a quantum system.
        collapse_operators: A sequence of operators that describe the influence of 
            noise on the quantum system.
        observables: A sequence of operators for which to compute their expectation
            value during evolution. If `store_intermediate_results` is set to True,
            the expectation values are computed after each step in the schedule, 
            and otherwise only the final expectation values at the end of the 
            evolution are computed.

    Returns:
        A single evolution result if a single initial state is provided, or a sequence
        of evolution results representing the data computed during the evolution of each
        initial state. See `EvolveResult` for more information about the data computed
        during evolution.
    """
    target_name = cudaq_runtime.get_target().name
    if target_name == "nvidia-dynamics":
        return evolve_me(hamiltonian, dimensions, schedule, initial_state,
                         collapse_operators, observables,
                         store_intermediate_results, integrator)

    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError(
            "time evolution is currently only supported on simulator targets")

    # Unless we are using cuSuperoperator for the execution,
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        # FIXME: tensornet can potentially handle qudits
        raise ValueError(
            "computing the time evolution is only possible for qubits; use the nvidia-dynamics target to simulate time evolution of arbitrary d-level systems"
        )
    # Unless we are using cuSuperoperator for the execution,
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0 and simulator != "dm":
        raise ValueError(
            "collapse operators can only be defined when using the nvidia-dynamics or density-matrix-cpu target"
        )

    # FIXME: deal with a sequence of initial states
    if store_intermediate_results:
        evolution = _create_kernels("time_evolution", hamiltonian, schedule,
                                    collapse_operators)
        global noise
        cudaq_runtime.set_noise(noise)
        kernels, observable_spinops = [], []
        for kernel, parameters in evolution:
            kernels.append(kernel)
            observable_spinops.append([lambda: op._to_spinop(dimensions, **parameters) for op in observables])
        if len(observables) == 0: return cudaq_runtime.evolve(initial_state, kernels)
        return cudaq_runtime.evolve(initial_state, kernels, observable_spinops)
    else:
        kernel, parameters = _create_kernel("time_evolution", hamiltonian,
                                            schedule, collapse_operators)
        global noise
        cudaq_runtime.set_noise(noise)
        if len(observables) == 0:
            return cudaq_runtime.evolve(initial_state, kernel)
        # FIXME: permit to compute expectation values for operators defined as matrix
        observable_spinops = [lambda: op._to_spinop(dimensions, **parameters) for op in observables]
        return cudaq_runtime.evolve(initial_state, kernel, observable_spinops)


def evolve_async(
    hamiltonian: Operator,
    dimensions: Mapping[int, int],
    schedule: Schedule,
    initial_state: cudaq_runtime.State | Sequence[cudaq_runtime.State],
    collapse_operators: Sequence[Operator] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results=False
) -> cudaq_runtime.AsyncEvolveResult | Sequence[
        cudaq_runtime.AsyncEvolveResult]:
    """
    Asynchronously computes the time evolution of one or more initial state(s) 
    under the defined operator(s). See `cudaq.evolve` for more details about the
    parameters passed here.
    
    Returns:
        The handle to a single evolution result if a single initial state is provided, 
        or a sequence of handles to the evolution results representing the data computed 
        during the evolution of each initial state. See the `EvolveResult` for more 
        information about the data computed during evolution.
    """
    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError(
            "time evolution is currently only supported on simulator targets")
    elif simulator == "nvidia-dynamics":  # FIXME: update once we know the target name
        raise NotImplementedError(f"{simulator} backend does not yet exist")

    # Unless we are using cuSuperoperator for the execution,
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        # FIXME: tensornet can potentially handle qudits
        raise ValueError(
            "computing the time evolution is only possible for qubits; use the nvidia-dynamics target to simulate time evolution of arbitrary d-level systems"
        )
    # Unless we are using cuSuperoperator for the execution,
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0 and simulator != "dm":
        raise ValueError(
            "collapse operators can only be defined when using the nvidia-dynamics or density-matrix-cpu target"
        )

    # FIXME: deal with a sequence of initial states
    if store_intermediate_results:
        evolution = _create_kernels("time_evolution", hamiltonian, schedule, collapse_operators)
        global noise
        cudaq_runtime.set_noise(noise)
        kernels, observable_spinops = [], []
        for kernel, parameters in evolution:
            kernels.append(kernel)
            observable_spinops.append([lambda: op._to_spinop(dimensions, **parameters) for op in observables])
        if len(observables) == 0: return cudaq_runtime.evolve_async(initial_state, kernels)
        return cudaq_runtime.evolve_async(initial_state, kernels, observable_spinops)
    else:
        kernel, parameters = _create_kernel("time_evolution", hamiltonian,
                                            schedule, collapse_operators)
        global noise
        cudaq_runtime.set_noise(noise)
        if len(observables) == 0:
            return cudaq_runtime.evolve_async(initial_state, kernel)
        # FIXME: permit to compute expectation values for operators defined as matrix
        observable_spinops = [lambda: op._to_spinop(dimensions, **parameters) for op in observables]
        return cudaq_runtime.evolve_async(initial_state, kernel, observable_spinops)

