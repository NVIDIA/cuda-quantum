# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import numpy
import random
import string
import warnings
import uuid
from numpy.typing import NDArray
from typing import Callable, Iterable, Mapping, Optional, Sequence

from cudaq.kernel.kernel_builder import PyKernel, make_kernel
from cudaq.kernel.register_op import register_operation
from cudaq.kernel.utils import ahkPrefix
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from ..operators import NumericType, Operator, RydbergHamiltonian, SuperOperator
from .helpers import InitialState, InitialStateArgT, IntermediateResultSave
from .integrator import BaseIntegrator
from .schedule import Schedule

analog_targets = ["pasqal", "quera"]


def _taylor_series_expm(op_matrix: NDArray[numpy.complexfloating],
                        order: int = 20) -> NDArray[numpy.complexfloating]:
    """
    Approximate the matrix exponential using the Taylor series expansion.
    """
    result = numpy.eye(op_matrix.shape[0], dtype=op_matrix.dtype)
    op_matrix_n = numpy.eye(op_matrix.shape[0], dtype=op_matrix.dtype)
    factorial = 1

    for n in range(1, order + 1):
        op_matrix_n = numpy.dot(op_matrix_n, op_matrix)
        factorial *= n
        result += op_matrix_n / factorial

    return result


def _compute_step_matrix(hamiltonian: Operator,
                         dimensions: Mapping[int, int],
                         parameters: Mapping[str, NumericType],
                         dt: float,
                         use_gpu: bool = False) -> NDArray[complexfloating]:
    op_matrix = hamiltonian.to_matrix(dimensions, **parameters)
    op_matrix = -1j * op_matrix * dt

    if use_gpu:
        import cupy as cp
        op_matrix_gpu = cp.array(op_matrix)
        return cp.asnumpy(cp.exp(op_matrix_gpu))
    else:
        return _taylor_series_expm(op_matrix)


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


def _launch_analog_hamiltonian_kernel(target_name: str,
                                      hamiltonian: RydbergHamiltonian,
                                      schedule: Schedule,
                                      shots_count: int,
                                      is_async: bool = False):
    # Get the operators to generate time series
    tlist = [schedule.current_step for _ in schedule]
    amp_ts = []
    ph_ts = []
    dg_ts = []
    for t in tlist:
        for ts, op in zip([amp_ts, ph_ts, dg_ts], [
                hamiltonian.amplitude, hamiltonian.phase,
                hamiltonian.delta_global
        ]):
            if op is not None:
                param_names = op.parameters.keys()
                if len(param_names) == 0:
                    evaluated = op.evaluate()
                elif len(param_names) == 1:
                    param_map = {next(iter(param_names)): t}
                    evaluated = op.evaluate(**param_map)
                else:
                    raise ValueError(
                        "generator for tunable parameter must not take more than one argument"
                    )
                if abs(evaluated.imag) != 0:
                    raise ValueError("tunable parameter must be real")
                ts.append((evaluated.real, t))

    atoms = cudaq_runtime.ahs.AtomArrangement()
    atoms.sites = hamiltonian.atom_sites
    atoms.filling = hamiltonian.atom_filling

    omega = cudaq_runtime.ahs.PhysicalField()
    omega.time_series = cudaq_runtime.ahs.TimeSeries(amp_ts)

    phi = cudaq_runtime.ahs.PhysicalField()
    phi.time_series = cudaq_runtime.ahs.TimeSeries(ph_ts)

    delta = cudaq_runtime.ahs.PhysicalField()
    delta.time_series = cudaq_runtime.ahs.TimeSeries(dg_ts)

    drive = cudaq_runtime.ahs.DrivingField()
    drive.amplitude = omega
    drive.phase = phi
    drive.detuning = delta

    program = cudaq_runtime.ahs.Program()
    program.setup.ahs_register = atoms
    program.hamiltonian.drivingFields = [drive]
    program.hamiltonian.localDetuning = []

    funcName = '{}{}_{}'.format(
        ahkPrefix, target_name, ''.join(
            random.choice(string.ascii_uppercase + string.digits)
            for _ in range(10)))

    ctx = cudaq_runtime.ExecutionContext("sample", shots_count)
    ctx.asyncExec = is_async
    cudaq_runtime.setExecutionContext(ctx)
    cudaq_runtime.pyAltLaunchAnalogKernel(funcName, program.to_json())
    if is_async:
        return ctx.asyncResult
    res = ctx.result
    cudaq_runtime.resetExecutionContext()
    return res


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
        initial_state: InitialStateArgT,
        collapse_operators: Sequence[Operator] = [],
        observables: Sequence[Operator] = [],
        store_intermediate_results:
    IntermediateResultSave = IntermediateResultSave.NONE,
        integrator: Optional[BaseIntegrator] = None,
        shots_count: Optional[int] = None) -> cudaq_runtime.EvolveResult:
    target_name = cudaq_runtime.get_target().name
    if not isinstance(store_intermediate_results, IntermediateResultSave):
        raise ValueError(
            f"Invalid argument `store_intermediate_results` for target {cudaq_runtime.get_target().name}."
        )

    if target_name in analog_targets:
        ## TODO: Convert result from `sample_result` to `evolve_result`
        return _launch_analog_hamiltonian_kernel(target_name, hamiltonian,
                                                 schedule, shots_count)

    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError(
            "time evolution is currently only supported on simulator targets")

    # Unless we are using `cuquantum.densitymat` for the execution,
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        # FIXME: `tensornet` can potentially handle qudits
        raise ValueError(
            "computing the time evolution is only possible for qubits; use the dynamics target to simulate time evolution of arbitrary d-level systems"
        )
    # Unless we are using `cuquantum.densitymat` for the execution,
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0 and simulator != "dm":
        raise ValueError(
            "collapse operators can only be defined when using the dynamics or density-matrix-cpu target"
        )

    num_qubits = len(hamiltonian.degrees)
    parameters = [mapping for mapping in schedule]
    schedule.reset()
    tlist = [schedule.current_step for _ in schedule]
    observable_spinops = [
        lambda step_parameters, op=op: op.evaluate(**step_parameters)
        for op in observables
    ]
    compute_step_matrix = lambda step_parameters, dt: _compute_step_matrix(
        hamiltonian, dimensions, step_parameters, dt)
    noise = cudaq_runtime.NoiseModel()
    add_noise_channel_for_step = lambda step_kernel_name, step_parameters, dt: _add_noise_channel_for_step(
        step_kernel_name, noise, collapse_operators, dimensions,
        step_parameters, dt)
    if shots_count is None:
        shots_count = -1
    if isinstance(initial_state, InitialState):
        # This is an initial state enum, create concrete state.
        state_size = 2**num_qubits
        if initial_state == InitialState.ZERO:
            state_data = numpy.zeros(state_size, dtype=numpy.complex128)
            state_data[0] = 1.0
        elif initial_state == InitialState.UNIFORM:
            state_data = (1. / numpy.sqrt(state_size)) * numpy.ones(
                state_size, dtype=numpy.complex128)
        else:
            raise ValueError("Unsupported initial state type")

        sim_name = cudaq_runtime.get_target().simulator.strip()
        if sim_name == "dm":
            initial_state = cudaq_runtime.State.from_data(
                numpy.outer(state_data, numpy.conj(state_data)))
        else:
            initial_state = cudaq_runtime.State.from_data(state_data)

    if store_intermediate_results != IntermediateResultSave.NONE:
        evolution = _evolution_kernel(
            num_qubits,
            compute_step_matrix,
            tlist,
            parameters,
            split_into_steps=True,
            register_kraus_channel=add_noise_channel_for_step)
        kernels = [kernel for kernel in evolution]
        save_intermediate_states = store_intermediate_results == IntermediateResultSave.ALL
        if len(observables) == 0:
            return cudaq_runtime.evolve(initial_state, kernels,
                                        save_intermediate_states)
        if len(collapse_operators) > 0:
            cudaq_runtime.set_noise(noise)
        result = cudaq_runtime.evolve(initial_state, kernels, parameters,
                                      observable_spinops, shots_count,
                                      save_intermediate_states)
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
                                      observable_spinops, shots_count)
        cudaq_runtime.unset_noise()
        return result


# Top level API for the CUDA-Q master equation solver.
def evolve(
    hamiltonian: Operator | SuperOperator | Sequence[Operator] |
    Sequence[SuperOperator],
    dimensions: Mapping[int, int] = {},
    schedule: Schedule = None,
    initial_state: InitialStateArgT | Sequence[InitialStateArgT] = None,
    collapse_operators: Sequence[Operator] | Sequence[Sequence[Operator]] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results: IntermediateResultSave |
    bool = IntermediateResultSave.NONE,
    integrator: Optional[BaseIntegrator] = None,
    shots_count: Optional[int] = None,
    max_batch_size: Optional[int] = None
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
            value during evolution. If `store_intermediate_results` is not None,
            the expectation values are computed after each step in the schedule, 
            and otherwise only the final expectation values at the end of the 
            evolution are computed.
        shots_count: Optional integer, if provided, it is the number of shots to use
            for QPU execution.

    Returns:
        A single evolution result if a single initial state is provided, or a sequence
        of evolution results representing the data computed during the evolution of each
        initial state. See `EvolveResult` for more information about the data computed
        during evolution.
    """
    target_name = cudaq_runtime.get_target().name
    if not isinstance(schedule, Schedule):
        raise ValueError(
            f"Invalid argument `schedule` for target {target_name}.")

    if isinstance(store_intermediate_results, bool):
        warnings.warn(
            "deprecated - use an `IntermediateResultSave` enum value instead",
            DeprecationWarning)
        store_intermediate_results = IntermediateResultSave.ALL if store_intermediate_results else IntermediateResultSave.NONE

    if target_name in analog_targets:
        if not isinstance(hamiltonian, RydbergHamiltonian):
            raise ValueError(
                f"Invalid argument `hamiltonian` for target {target_name}. Must be `RydbergHamiltonian` operator."
            )
        if bool(dimensions):
            raise ValueError(
                f"Unexpected argument `dimensions` for target {target_name}.")
        if initial_state is not None:
            raise ValueError(
                f"Unexpected argument `initial_state` for target {target_name}."
            )
        if integrator is not None:
            raise ValueError(
                f"Unexpected argument `integrator` for target {target_name}.")
        if len(collapse_operators) != 0:
            raise ValueError(
                f"Unexpected argument `collapse_operators` for target {target_name}."
            )
        if len(observables) != 0:
            raise ValueError(
                f"Unexpected argument `observables` for target {target_name}.")
        if store_intermediate_results != IntermediateResultSave.NONE:
            raise ValueError(
                f"Unexpected argument `store_intermediate_results` for target {target_name}."
            )
        if shots_count is None:
            shots_count = 100
    else:
        if dimensions is None or not bool(dimensions):
            raise RuntimeError(
                f"Valid `dimensions` must be provided for target {target_name}")
        if initial_state is None:
            raise RuntimeError(
                f"Valid `initial_state` must be provided for target {target_name}"
            )

    if target_name == "dynamics" and shots_count is not None:
        warnings.warn(f"`shots_count` will be ignored on target {target_name}")

    if target_name != "dynamics" and max_batch_size is not None:
        warnings.warn(f"`batch_size` will be ignored on target {target_name}")

    if max_batch_size is not None and max_batch_size < 1:
        raise ValueError(
            f"Invalid max_batch_size {max_batch_size}. It must be at least 1.")

    if isinstance(hamiltonian, Sequence):
        if len(hamiltonian) == 0:
            raise ValueError(
                "If `hamiltonian` is a sequence, then it must not be empty.")

        # This is batched operators evolve.
        # Broadcast the initial state to the same length as the hamiltonian if it is a single state.
        if not isinstance(initial_state, Sequence):
            initial_state = [initial_state] * len(hamiltonian)

        if len(hamiltonian) != len(initial_state):
            raise ValueError(
                "If `hamiltonian` is a sequence, then `initial_state` must be a sequence of the same length."
            )

        if isinstance(hamiltonian[0], Operator):
            if len(collapse_operators) == 0:
                collapse_operators = [[] for _ in range(len(hamiltonian))]

            if len(hamiltonian) != len(collapse_operators):
                raise ValueError(
                    "If `hamiltonian` is a sequence, then `collapse_operators` must be a sequence of the same length."
                )

            for collapse_ops in collapse_operators:
                if not isinstance(collapse_ops, Sequence):
                    raise ValueError(
                        "If `hamiltonian` is a sequence, then `collapse_operators` must be a sequence of lists of collapse operators (nested sequence)."
                    )

    if target_name == "dynamics":
        try:
            from .cudm_solver import evolve_dynamics
        except:
            raise ImportError(
                "Failed to load dynamics solver. Please check your installation"
            )
        return evolve_dynamics(hamiltonian, dimensions, schedule, initial_state,
                               collapse_operators, observables,
                               store_intermediate_results, integrator,
                               max_batch_size)
    else:
        if isinstance(initial_state, Sequence):
            return [
                evolve_single(ham, dimensions, schedule, state, collapse_ops,
                              observables, store_intermediate_results,
                              integrator, shots_count)
                for ham, state, collapse_ops in zip(hamiltonian, initial_state,
                                                    collapse_operators)
            ]
        else:
            return evolve_single(hamiltonian, dimensions, schedule,
                                 initial_state, collapse_operators, observables,
                                 store_intermediate_results, integrator,
                                 shots_count)


def evolve_single_async(
        hamiltonian: Operator,
        dimensions: Mapping[int, int],
        schedule: Schedule,
        initial_state: InitialStateArgT,
        collapse_operators: Sequence[Operator] = [],
        observables: Sequence[Operator] = [],
        store_intermediate_results:
    IntermediateResultSave = IntermediateResultSave.NONE,
        integrator: Optional[BaseIntegrator] = None,
        shots_count: Optional[int] = None) -> cudaq_runtime.AsyncEvolveResult:
    if not isinstance(store_intermediate_results, IntermediateResultSave):
        raise ValueError(
            f"Invalid argument `store_intermediate_results` for target {cudaq_runtime.get_target().name}."
        )
    target_name = cudaq_runtime.get_target().name
    if target_name == "dynamics":
        try:
            from .cudm_solver import evolve_dynamics
        except:
            raise ImportError(
                "Failed to load dynamics solver. Please check your installation"
            )
        return cudaq_runtime.evolve_async(lambda: evolve_dynamics(
            hamiltonian, dimensions, schedule, initial_state,
            collapse_operators, observables, store_intermediate_results,
            integrator))

    if target_name in analog_targets:
        return _launch_analog_hamiltonian_kernel(target_name, hamiltonian,
                                                 schedule, shots_count, True)

    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError(
            "time evolution is currently only supported on simulator targets")
    elif simulator == "dynamics":
        raise NotImplementedError(f"{simulator} backend does not yet exist")

    # Unless we are using `cuquantum.densitymat` for the execution,
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        # FIXME: `tensornet` can potentially handle qudits
        raise ValueError(
            "computing the time evolution is only possible for qubits; use the dynamics target to simulate time evolution of arbitrary d-level systems"
        )
    # Unless we are using a simulator that supports noisy simulation,
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0 and simulator != "dm":
        raise ValueError(
            "collapse operators can only be defined when using the dynamics or density-matrix-cpu target"
        )
    num_qubits = len(hamiltonian.degrees)
    parameters = [mapping for mapping in schedule]
    schedule.reset()
    tlist = [schedule.current_step for _ in schedule]
    observable_spinops = [
        lambda step_parameters, op=op: op.evaluate(**step_parameters)
        for op in observables
    ]
    compute_step_matrix = lambda step_parameters, dt: _compute_step_matrix(
        hamiltonian, dimensions, step_parameters, dt)
    noise = cudaq_runtime.NoiseModel()
    add_noise_channel_for_step = lambda step_kernel_name, step_parameters, dt: _add_noise_channel_for_step(
        step_kernel_name, noise, collapse_operators, dimensions,
        step_parameters, dt)
    if shots_count is None:
        shots_count = -1
    if store_intermediate_results != IntermediateResultSave.NONE:
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
                                              noise_model=noise,
                                              shots_count=shots_count)
        return cudaq_runtime.evolve_async(initial_state,
                                          kernels,
                                          parameters,
                                          observable_spinops,
                                          shots_count=shots_count)
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
            return cudaq_runtime.evolve_async(initial_state,
                                              kernel,
                                              parameters[-1],
                                              observable_spinops,
                                              noise_model=noise,
                                              shots_count=shots_count)
        return cudaq_runtime.evolve_async(initial_state,
                                          kernel,
                                          parameters[-1],
                                          observable_spinops,
                                          shots_count=shots_count)


def evolve_async(
    hamiltonian: Operator,
    dimensions: Mapping[int, int] = {},
    schedule: Schedule = None,
    initial_state: InitialStateArgT | Sequence[InitialStateArgT] = None,
    collapse_operators: Sequence[Operator] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results: IntermediateResultSave |
    bool = IntermediateResultSave.NONE,
    integrator: Optional[BaseIntegrator] = None,
    shots_count: Optional[int] = None
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
    target_name = cudaq_runtime.get_target().name

    if not isinstance(schedule, Schedule):
        raise ValueError(
            f"Invalid argument `schedule` for target {target_name}.")

    if isinstance(store_intermediate_results, bool):
        warnings.warn(
            "deprecated - use an `IntermediateResultSave` enum value instead",
            DeprecationWarning)
        store_intermediate_results = IntermediateResultSave.ALL if store_intermediate_results else IntermediateResultSave.NONE

    if target_name in analog_targets:
        if not isinstance(hamiltonian, RydbergHamiltonian):
            raise ValueError(
                f"Invalid argument `hamiltonian` for target {target_name}. Must be `RydbergHamiltonian` operator."
            )
        if bool(dimensions):
            raise ValueError(
                f"Unexpected argument `dimensions` for target {target_name}.")
        if initial_state is not None:
            raise ValueError(
                f"Unexpected argument `initial_state` for target {target_name}."
            )
        if integrator is not None:
            raise ValueError(
                f"Unexpected argument `integrator` for target {target_name}.")
        if len(collapse_operators) != 0:
            raise ValueError(
                f"Unexpected argument `collapse_operators` for target {target_name}."
            )
        if len(observables) != 0:
            raise ValueError(
                f"Unexpected argument `observables` for target {target_name}.")
        if store_intermediate_results != IntermediateResultSave.NONE:
            raise ValueError(
                f"Unexpected argument `store_intermediate_results` for target {target_name}."
            )
        if shots_count is None:
            shots_count = 100
    else:
        if dimensions is None or not bool(dimensions):
            raise RuntimeError(
                f"Valid `dimensions` must be provided for target {target_name}")
        if initial_state is None:
            raise RuntimeError(
                f"Valid `initial_state` must be provided for target {target_name}"
            )

    if target_name == "dynamics" and shots_count is not None:
        warnings.warn(f"`shots_count` will be ignored on target {target_name}")

    if isinstance(initial_state, Sequence):
        return [
            evolve_single_async(hamiltonian, dimensions, schedule, state,
                                collapse_operators, observables,
                                store_intermediate_results, integrator,
                                shots_count) for state in initial_state
        ]
    else:
        return evolve_single_async(hamiltonian, dimensions, schedule,
                                   initial_state, collapse_operators,
                                   observables, store_intermediate_results,
                                   integrator, shots_count)
