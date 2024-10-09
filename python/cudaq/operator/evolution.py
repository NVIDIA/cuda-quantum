# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import numpy, scipy, sys, uuid
from numpy.typing import NDArray
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .expressions import Operator
from .helpers import _OperatorHelpers, NumericType
from .schedule import Schedule
from ..kernel.register_op import register_operation
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
# FIXME: `PyKernelDecorator` is documented but not accessible in the API due to shadowing by the kernel module.
#from ..kernel.kernel_decorator import PyKernelDecorator
from ..kernel.kernel_builder import PyKernel, make_kernel
from ..runtime.observe import observe
from .integrator import BaseIntegrator


def _compute_step_matrix(hamiltonian: Operator, dimensions: Mapping[int, int],
                         parameters: Mapping[str, NumericType],
                         dt: float) -> NDArray[complexfloating]:
    op_matrix = hamiltonian.to_matrix(dimensions, **parameters)
    # FIXME: Use `approximative` approach (series expansion, integrator),
    # and maybe use GPU acceleration for matrix manipulations if it makes sense.
    return scipy.linalg.expm(-1j * op_matrix * dt)


def _add_noise_channel_for_step(step_kernel_name: str,
                                noise_model: cudaq_runtime.NoiseModel,
                                collapse_operators: Sequence[Operator],
                                dimensions: Mapping[int, int],
                                parameters: Mapping[str,
                                                    NumericType], dt: float):
    for collapse_op in collapse_operators:
        L = collapse_op.to_matrix(dimensions, **parameters)
        G = -0.5 * numpy.dot(L.conj().T, L)
        M0 = G * dt + numpy.eye(2**len(dimensions))
        M1 = numpy.sqrt(dt) * L
        try:
            noise_model.add_all_qubit_channel(
                step_kernel_name, cudaq_runtime.KrausChannel([M0, M1]))
        except Exception as e:
            print(
                f"Error adding noise channel, perhaps the time step is too large."
            )
            raise e


# FIXME: move to C++
def _evolution_kernel(
    num_qubits: int,
    compute_step_matrix: Callable[[Mapping[str, NumericType], float],
                                  NDArray[numpy.complexfloating]],
    tlist: Sequence[float],
    schedule: Iterable[Mapping[str, NumericType]],
    split_into_steps=False,
    register_kraus_channel: Optional[Callable[
        [str, Mapping[str, NumericType], float], None]] = None
) -> Generator[PyKernel]:
    kernel_base_name = "".join(filter(str.isalnum, str(uuid.uuid4())))

    def register_operations():
        for step_idx, parameters in enumerate(schedule):
            # We could make operators `hashable` and try to use that to do some kernel caching,
            # but there is no guarantee that if the hash is equal, the operator is equal.
            # Overall it feels like a better choice to just take a `uuid` here.
            operation_name = f"evolve_{kernel_base_name}_{step_idx}"
            # Note: the first step is expected to be the identity matrix, i.e., initial state.
            if step_idx == 0:
                evolution_matrix = numpy.eye(2**num_qubits,
                                             dtype=numpy.complex128)
                register_operation(operation_name, evolution_matrix)
            else:
                dt = tlist[step_idx] - tlist[step_idx - 1]
                register_operation(operation_name,
                                   compute_step_matrix(parameters, dt))
                register_kraus_channel(operation_name, parameters, dt)
            yield operation_name

    operation_names = register_operations()

    evolution, initial_state = make_kernel(cudaq_runtime.State)
    qs = evolution.qalloc(initial_state)
    for operation_name in operation_names:
        # FIXME: `#(*qs)` causes infinite loop
        # FIXME: It would be nice if a registered operation could take a vector of qubits?
        targets = [qs[i] for i in range(num_qubits)]
        evolution.__getattr__(f"{operation_name}")(*targets)
        if split_into_steps:
            yield evolution
            evolution, initial_state = make_kernel(cudaq_runtime.State)
            qs = evolution.qalloc(initial_state)
    if not split_into_steps:
        yield evolution


def evolve_single(
        hamiltonian: Operator,
        dimensions: Mapping[int, int],
        schedule: Schedule,
        initial_state: cudaq_runtime.State,
        collapse_operators: Sequence[Operator] = [],
        observables: Sequence[Operator] = [],
        store_intermediate_results=False,
        integrator: Optional[BaseIntegrator] = None
) -> cudaq_runtime.EvolveResult:
    target_name = cudaq_runtime.get_target().name
    if target_name == "nvidia-dynamics":
        try:
            from .cuso_solver import evolve_dynamics
        except:
            raise ImportError(
                "Failed to load nvidia-dynamics solver. Please check your installation"
            )
        return evolve_dynamics(hamiltonian, dimensions, schedule, initial_state,
                               collapse_operators, observables,
                               store_intermediate_results, integrator)

    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError(
            "time evolution is currently only supported on simulator targets")

    # Unless we are using `cuSuperoperator` for the execution,
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        # FIXME: `tensornet` can potentially handle qudits
        raise ValueError(
            "computing the time evolution is only possible for qubits; use the nvidia-dynamics target to simulate time evolution of arbitrary d-level systems"
        )
    # Unless we are using `cuSuperoperator` for the execution,
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0 and simulator != "dm":
        raise ValueError(
            "collapse operators can only be defined when using the nvidia-dynamics or density-matrix-cpu target"
        )

    num_qubits = len(hamiltonian.degrees)
    parameters = [mapping for mapping in schedule]
    schedule.reset()
    tlist = [schedule.current_step for _ in schedule]
    observable_spinops = [
        lambda step_parameters, op=op: op._to_spinop(
            dimensions, **step_parameters) for op in observables
    ]
    compute_step_matrix = lambda step_parameters, dt: _compute_step_matrix(
        hamiltonian, dimensions, step_parameters, dt)
    noise = cudaq_runtime.NoiseModel()
    add_noise_channel_for_step = lambda step_kernel_name, step_parameters, dt: _add_noise_channel_for_step(
        step_kernel_name, noise, collapse_operators, dimensions,
        step_parameters, dt)

    if store_intermediate_results:
        evolution = _evolution_kernel(
            num_qubits,
            compute_step_matrix,
            tlist,
            parameters,
            split_into_steps=True,
            register_kraus_channel=add_noise_channel_for_step)
        kernels = [kernel for kernel in evolution]
        if len(observables) == 0:
            return cudaq_runtime.evolve(initial_state, kernels)
        if len(collapse_operators) > 0:
            cudaq_runtime.set_noise(noise)
        result = cudaq_runtime.evolve(initial_state, kernels, parameters,
                                      observable_spinops)
        cudaq_runtime.unset_noise()
        return result
    else:
        kernel = next(
            _evolution_kernel(
                num_qubits,
                compute_step_matrix,
                tlist,
                parameters,
                register_kraus_channel=add_noise_channel_for_step))
        if len(observables) == 0:
            return cudaq_runtime.evolve(initial_state, kernel)
        # FIXME: permit to compute expectation values for operators defined as matrix
        if len(collapse_operators) > 0:
            cudaq_runtime.set_noise(noise)
        result = cudaq_runtime.evolve(initial_state, kernel, parameters[-1],
                                      observable_spinops)
        cudaq_runtime.unset_noise()
        return result


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
        `hamiltonian`: Operator that describes the behavior of a quantum system
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
        `observables`: A sequence of operators for which to compute their expectation
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
    if isinstance(initial_state, Sequence):
        return [
            evolve_single(hamiltonian, dimensions, schedule, state,
                          collapse_operators, observables,
                          store_intermediate_results, integrator)
            for state in initial_state
        ]
    else:
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             collapse_operators, observables,
                             store_intermediate_results, integrator)


def evolve_single_async(
    hamiltonian: Operator,
    dimensions: Mapping[int, int],
    schedule: Schedule,
    initial_state: cudaq_runtime.State,
    collapse_operators: Sequence[Operator] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results=False,
    integrator: Optional[BaseIntegrator] = None
) -> cudaq_runtime.AsyncEvolveResult:
    target_name = cudaq_runtime.get_target().name
    if target_name == "nvidia-dynamics":
        try:
            from .cuso_solver import evolve_dynamics
        except:
            raise ImportError(
                "Failed to load nvidia-dynamics solver. Please check your installation"
            )

        return cudaq_runtime.evolve_async(lambda: evolve_dynamics(
            hamiltonian, dimensions, schedule, initial_state,
            collapse_operators, observables, store_intermediate_results,
            integrator))

    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError(
            "time evolution is currently only supported on simulator targets")
    elif simulator == "nvidia-dynamics":  # FIXME: update once we know the target name
        raise NotImplementedError(f"{simulator} backend does not yet exist")

    # Unless we are using `cuSuperoperator` for the execution,
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        # FIXME: `tensornet` can potentially handle qudits
        raise ValueError(
            "computing the time evolution is only possible for qubits; use the nvidia-dynamics target to simulate time evolution of arbitrary d-level systems"
        )
    # Unless we are using `cuSuperoperator` for the execution,
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0 and simulator != "dm":
        raise ValueError(
            "collapse operators can only be defined when using the nvidia-dynamics or density-matrix-cpu target"
        )

    num_qubits = len(hamiltonian.degrees)
    parameters = [mapping for mapping in schedule]
    schedule.reset()
    tlist = [schedule.current_step for _ in schedule]
    observable_spinops = [
        lambda step_parameters, op=op: op._to_spinop(
            dimensions, **step_parameters) for op in observables
    ]
    compute_step_matrix = lambda step_parameters, dt: _compute_step_matrix(
        hamiltonian, dimensions, step_parameters, dt)
    noise = cudaq_runtime.NoiseModel()
    add_noise_channel_for_step = lambda step_kernel_name, step_parameters, dt: _add_noise_channel_for_step(
        step_kernel_name, noise, collapse_operators, dimensions,
        step_parameters, dt)

    # FIXME: deal with a sequence of initial states
    if store_intermediate_results:
        evolution = _evolution_kernel(
            num_qubits,
            compute_step_matrix,
            tlist,
            parameters,
            split_into_steps=True,
            register_kraus_channel=add_noise_channel_for_step)
        kernels = [kernel for kernel in evolution]
        if len(observables) == 0:
            return cudaq_runtime.evolve_async(initial_state, kernels)
        if len(collapse_operators) > 0:
            return cudaq_runtime.evolve_async(initial_state,
                                              kernels,
                                              parameters,
                                              observable_spinops,
                                              noise_model=noise)
        return cudaq_runtime.evolve_async(initial_state, kernels, parameters,
                                          observable_spinops)
    else:
        kernel = next(
            _evolution_kernel(
                num_qubits,
                compute_step_matrix,
                tlist,
                parameters,
                register_kraus_channel=add_noise_channel_for_step))
        if len(observables) == 0:
            return cudaq_runtime.evolve_async(initial_state, kernel)
        # FIXME: permit to compute expectation values for operators defined as matrix
        if len(collapse_operators) > 0:
            cudaq_runtime.evolve_async(initial_state,
                                       kernel,
                                       parameters[-1],
                                       observable_spinops,
                                       noise_model=noise)
        return cudaq_runtime.evolve_async(initial_state, kernel, parameters[-1],
                                          observable_spinops)


def evolve_async(
    hamiltonian: Operator,
    dimensions: Mapping[int, int],
    schedule: Schedule,
    initial_state: cudaq_runtime.State | Sequence[cudaq_runtime.State],
    collapse_operators: Sequence[Operator] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results=False,
    integrator: Optional[BaseIntegrator] = None
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
    if isinstance(initial_state, Sequence):
        return [
            evolve_single_async(hamiltonian, dimensions, schedule, state,
                                collapse_operators, observables,
                                store_intermediate_results, integrator)
            for state in initial_state
        ]
    else:
        return evolve_single_async(hamiltonian, dimensions, schedule,
                                   initial_state, collapse_operators,
                                   observables, store_intermediate_results,
                                   integrator)
