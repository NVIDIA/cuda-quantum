from __future__ import annotations
import numpy, scipy, sys, uuid
from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Iterable

from .expressions import Operator
from .helpers import _OperatorHelpers
from .schedule import Schedule
from ..kernel.register_op import register_operation
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from ..kernel.kernel_decorator import PyKernelDecorator
from ..runtime.observe import observe

# To be implemented in C++ and bindings will be generated.
# If multiple initial states were passed, a sequence of evolution results is returned.
class EvolveResult:
    """
    Stores the execution data from an invocation of `cudaq.evolve`.
    """

    # Shape support in the type annotation for ndarray data type is still under development:
    # https://github.com/numpy/numpy/issues/16544
    def __init__(self: EvolveResult, 
                 state: cudaq_runtime.State | Iterable[cudaq_runtime.State],
                 expectation: Optional[NDArray[cudaq_runtime.ObserveResult] | Iterable[NDArray[cudaq_runtime.ObserveResult]]] = None) -> None:
        """
        Instantiates an EvolveResult representing the output generated when evolving a single 
        initial state under a set of operators. See `cudaq.evolve` for more detail.

        Arguments:
            state: A single state or a sequence of states of a quantum system. If a single 
                state is given, it represents the final state of the system after time evolution.
                If a sequence of states are given, they represent the state of the system after
                each steps in the schedule specified in the call to `cudaq.evolve`.
            expectation: An one-dimensional array of results from the call to 
                `cudaq.observe` or a sequence of arrays thereof. If a single array 
                is provided, it contains the results from the calls to `cudaq.observe`
                at the end of the evolution for each observable defined in the call 
                to `cudaq.evolve`. If a sequence of arrays is provided, they represent 
                the results computed at each step in the schedule passed to 
                `cudaq.evolve`.
        """
        if isinstance(state, cudaq_runtime.State):
            self._intermediate_states = None
            self._final_state = state
        else:
            if len(state) == 0:
                raise ValueError("got an empty sequence of states")
            self._intermediate_states = state
            self._final_state = state[-1]
        if expectation is None or len(expectation) == 0:
            self._expectation_values = None
            self._final_expectation_values : Optional[NDArray[cudaq_runtime.ObserveResult]] = None
        else:
            if isinstance(expectation[0], cudaq_runtime.ObserveResult):
                if self._intermediate_states is not None:
                    raise ValueError("intermediate states were defined but no intermediate expectation values are provided")
                self._expectation_values = None
                self._final_expectation_values = expectation # type: ignore
            else:
                if self._intermediate_states is None:
                    raise ValueError("no intermediate states were defined but intermediate expectation values are provided")
                self._expectation_values = expectation
                self._final_expectation_values = expectation[-1]

    @property
    def intermediate_states(self: EvolveResult) -> Optional[Iterable[cudaq_runtime.State]]:
        """
        Stores all intermediate states, meaning the state after each step in a defined 
        schedule, produced by a call to `cudaq.evolve`, including the final state. 
        This property is only populated saving intermediate results was requested in the 
        call to `cudaq.evolve`.
        """
        return self._intermediate_states

    @property
    def final_state(self: EvolveResult) -> cudaq_runtime.State:
        """
        Stores the final state produced by a call to `cudaq.evolve`.
        Represent the state of a quantum system after time evolution under a set of 
        operators, see the `cudaq.evolve` documentation for more detail.
        """
        return self._final_state

    @property
    def expectation_values(self: EvolveResult) -> Optional[Iterable[NDArray[cudaq_runtime.ObserveResult]]]:
        """
        Stores the expectation values, that is the results from the calls to 
        `cudaq.observe`, at each step in the schedule produced by a call to 
        `cudaq.evolve`, including the final expectation values. Each entry 
        corresponds to one observable provided in the `cudaq.evolve` call. 
        This property is only populated saving intermediate results was requested in the 
        call to `cudaq.evolve`. This value will be None if no intermediate results were 
        requested, or if no observables were specified in the call.
        """
        return self._expectation_values

    @property
    def final_expectation_values(self: EvolveResult) -> Optional[NDArray[cudaq_runtime.ObserveResult]]:
        """
        Stores the final expectation values, that is the results produced by
        calls to `cudaq.observe`, from a call to `cudaq.evolve`. Each entry 
        corresponds to one observable provided in the `cudaq.evolve` call. 
        This value will be None if no observables were specified in the call.
        """
        return self._final_expectation_values

# To be implemented in C++ and bindings will be generated.
class AsyncEvolveResult:
    """
    Stores the execution data from an invocation of `cudaq.evolve_async`.
    """

    def __init__(self: AsyncEvolveResult, 
                 state: cudaq_runtime.AsyncStateResult | Iterable[cudaq_runtime.AsyncStateResult],
                 expectation: Optional[NDArray[cudaq_runtime.AsyncObserveResult] | Iterable[NDArray[cudaq_runtime.AsyncObserveResult]]] = None) -> None:
        """
        Creates a class instance that can be used to retrieve the evolution
        result produces by an calling the asynchronously executing function
        `cudaq.evolve_async`. It models a future-like type whose 
        `EvolveResult` may be accessed via an invocation of the `get`
        method. 

        Arguments:
            state: A single handle for retrieving the state of a quantum system, 
                or a sequence of handle. If a single handle is given, its value 
                represents the final state of the system after time evolution.
                If a sequence of handles are given, they represent the state of 
                the system after each steps in the schedule specified in the call 
                to `cudaq.evolve_async`.
            expectation: An one-dimensional array of results from the call to 
                `cudaq.observe_async` or a sequence of arrays thereof. If a single 
                array is provided, it contains the results from the calls to 
                `cudaq.observe_async` at the end of the evolution for each observable 
                defined in the call to `cudaq.evolve_async`. If a sequence of arrays 
                is provided, they represent the results computed at each step in the 
                schedule passed to `cudaq.evolve_async`.
        """
        if isinstance(state, cudaq_runtime.AsyncStateResult):
            self._intermediate_states = None
            self._final_state = state
        else:
            if len(state) == 0:
                raise ValueError("got an empty sequence of states")
            self._intermediate_states = state
            self._final_state = state[-1]
        if expectation is None or len(expectation) == 0:
            self._expectation_values = None
            self._final_expectation_values : Optional[NDArray[cudaq_runtime.ObserveResult]] = None
        else:
            if isinstance(expectation[0], cudaq_runtime.AsyncObserveResult):
                if self._intermediate_states is not None:
                    raise ValueError("intermediate states were defined but no intermediate expectation values are provided")
                self._expectation_values = None
                self._final_expectation_values = expectation # type: ignore
            else:
                if self._intermediate_states is None:
                    raise ValueError("no intermediate states were defined but intermediate expectation values are provided")
                self._expectation_values = expectation
                self._final_expectation_values = expectation[-1]

    @property
    def intermediate_states(self: EvolveResult) -> Optional[Iterable[cudaq_runtime.AsyncStateResult]]:
        """
        Stores the handle to all intermediate states, meaning the state after each step 
        in a defined schedule, produced by a call to `cudaq.evolve_async`, including the 
        final state. This property is only populated saving intermediate results was 
        requested in the call to `cudaq.evolve_async`.
        """
        return self._intermediate_states

    @property
    def final_state(self: EvolveResult) -> cudaq_runtime.AsyncStateResult:
        """
        Stores the handle to the final state produced by a call to `cudaq.evolve_async`.
        Its value represent the state of a quantum system after time evolution under a 
        set of operators, see the `cudaq.evolve_async` documentation for more detail.
        """
        return self._final_state

    @property
    def expectation_values(self: EvolveResult) -> Optional[Iterable[NDArray[cudaq_runtime.AsyncObserveResult]]]:
        """
        Stores the handles to the expectation values, that is the results from the calls 
        to `cudaq.observe_async`, at each step in the schedule produced by a call to 
        `cudaq.evolve_async`, including the final expectation values. Each entry 
        corresponds to one observable provided in the `cudaq.evolve_async` call. This 
        property is only populated saving intermediate results was requested in the 
        call to `cudaq.evolve_async`. This value will be None if no intermediate results 
        were requested, or if no observables were specified in the call.
        """
        return self._expectation_values

    @property
    def final_expectation_values(self: EvolveResult) -> Optional[NDArray[cudaq_runtime.AsyncObserveResult]]:
        """
        Stores the handles to the final expectation values, that is the results produced 
        by calls to `cudaq.observe_async`, from a call to `cudaq.evolve_async`. Each 
        entry corresponds to one observable provided in the `cudaq.evolve_async` call. 
        This value will be None if no observables were specified in the call.
        """
        return self._final_expectation_values

def _register_evolution_kernels(hamiltonian: Operator, schedule: Schedule) -> Generator[(str, Mapping[str, NumericType])]:
    # Evolution kernels can only be defined for qubits.
    dimensions = dict([(i, 2) for i in hamiltonian.degrees])
    # FIXME: MAKE OPERATORS HASHABLE
    # kernel_base_name = hash(hamiltonian)
    kernel_base_name = "".join(filter(str.isalnum, str(uuid.uuid4())))
    for step_idx, parameters in enumerate(schedule):
        kernel_name = f"evolve_{kernel_base_name}_{step_idx}"
        op_matrix = hamiltonian.to_matrix(dimensions, **parameters)
        # FIXME: Use GPU acceleration for matrix manipulations if possible.
        # Alternative/possibly better: do the same thing we'll do for hardware
        # and decompose directly into gates.
        evolution_matrix = scipy.linalg.expm(-1j * op_matrix)
        register_operation(kernel_name, evolution_matrix)
        yield kernel_name, parameters

def _create_kernel(name: str, 
                   hamiltonian: Operator, 
                   schedule: Schedule, 
                   initial_state = None) -> tuple[PyKernelDecorator, Mapping[str, NumericType]]:
    evolution = _register_evolution_kernels(hamiltonian, schedule)
    num_qubits = len(hamiltonian.degrees)
    srcCode = f"def {name}():\n"
    if initial_state is None:
        srcCode += f"\tqs = cudaq.qvector({num_qubits})\n"
    else:
        # FIXME: precision
        statevector = ', '.join([str(value) for value in initial_state])
        srcCode += f"\tqs = cudaq.qvector([{statevector}])\n"
    for operation_name, parameters in evolution:
        # FIXME: It would be nice if a registered operation could take a vector of qubits.
        arguments = [f"qs[{i}]" for i in range(num_qubits)]
        srcCode += f"\t{operation_name}({', '.join(arguments)})\n"
    # FIXME: PyKernelDecorator is documented but not accessible in the API due to shadowing by the kernel module.
    # See also https://stackoverflow.com/questions/6810999/how-to-determine-file-function-and-line-number
    kernel = PyKernelDecorator("evolution_kernel",
                               kernelName = name,
                               funcSrc = srcCode,
                               signature = {},
                               location = (__file__, sys._getframe().f_lineno))
    return kernel, parameters

def _create_kernels(name: str, 
                    hamiltonian: Operator, 
                    schedule: Schedule, 
                    initial_state = None) -> Generator[tuple[PyKernelDecorator, Mapping[str, NumericType]]]:
    evolution = _register_evolution_kernels(hamiltonian, schedule)
    num_qubits = len(hamiltonian.degrees)
    for op_idx, (operation_name, parameters) in enumerate(evolution):
        if op_idx == 0: srcCode = f"def {name}_{op_idx}():\n"
        else: srcCode = f"def {name}_{op_idx}(init_state: cudaq.State):\n"

        if op_idx == 0 and initial_state is None: qvec_arg = str(num_qubits)
        elif op_idx == 0: qvec_arg = f"[{', '.join([str(value) for value in initial_state])}]" # FIXME: precision
        else: qvec_arg = "init_state"
        srcCode += f"\tqs = cudaq.qvector({qvec_arg})\n"

        arguments = [f"qs[{i}]" for i in range(num_qubits)]
        srcCode += f"\t{operation_name}({', '.join(arguments)})\n"

        signature : dict[str, Any] = {}
        if op_idx != 0: signature["init_state"] = cudaq_runtime.State
        kernel = PyKernelDecorator(f"evolution_kernel",
                                   kernelName = f"{name}_{op_idx}",
                                   funcSrc = srcCode,
                                   signature = signature,
                                   location = (__file__, sys._getframe().f_lineno))
        yield kernel, parameters

def _state_to_kernel():
    kernel_name = "gen_" + "".join(filter(str.isalnum, str(uuid.uuid4())))
    srcCode = f"def {kernel_name}(init_state: cudaq.State):\n"
    srcCode += f"\tqs = cudaq.qvector(init_state)\n"
    return PyKernelDecorator(f"state_kernel",
                             kernelName = kernel_name,
                             funcSrc = srcCode,
                             signature = {"init_state": cudaq_runtime.State},
                             location = (__file__, sys._getframe().f_lineno))

# Top level API for the CUDA-Q master equation solver.
def evolve(hamiltonian: Operator, 
           dimensions: Mapping[int, int], 
           schedule: Schedule,
           initial_state: cudaq_runtime.State | Iterable[cudaq_runtime.States],
           collapse_operators: Iterable[Operator] = [],
           observables: Iterable[Operator] = [], 
           store_intermediate_results = False) -> EvolveResult | Iterable[EvolveResult]:
    """
    Computes the time evolution of one or more initial state(s) under the defined 
    operator(s). 

    Arguments:
        hamiltonian: Operator that describes the behavior of a quantum system
            without noise.
        dimensions: A mapping that specifies the number of levels, that is
            the dimension, of each degree of freedom that any of the operator 
            arguments acts on.
        schedule: An iterable that generates a mapping of keyword arguments 
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
    simulator = cudaq_runtime.get_target().simulator.strip()
    if simulator == "":
        raise NotImplementedError("time evolution is currently only supported on simulator targets")
    elif simulator == "nvidia-dynamics": # FIXME: update here and below once we know the target name
        raise NotImplementedError(f"{simulator} backend does not yet exist")

    # Unless we are using cuSuperoperator for the execution, 
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        raise ValueError("computing the time evolution is only possible for qubits; use the nvidia-dynamics target to simulate time evolution of arbitrary d-level systems")
    # Unless we are using cuSuperoperator for the execution, 
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0:
        raise ValueError("collapse operators can only be defined when using the nvidia-dynamics target")

    state_to_kernel = _state_to_kernel()
    def compute_expectations(cudaq_state: cudaq_runtime.State, parameters: Mapping[str, NumericType]) -> Iterable[cudaq_runtime.ObserveResult]:
        observable_spinops = [op._to_spinop(dimensions, **parameters) for op in observables]
        expectation_values: list[cudaq_runtime.ObserveResult] = []
        for observable in observable_spinops:
            expectation = observe(state_to_kernel, observable, cudaq_state)
            expectation_values.append(expectation)
        return expectation_values

    if store_intermediate_results:
        evolution = _create_kernels("time_evolution", hamiltonian, schedule, initial_state)
        states, expectations = [], []
        for kernel, parameters in evolution:
            if len(states) == 0: intermediate_state = cudaq_runtime.get_state(kernel)
            else: intermediate_state = cudaq_runtime.get_state(kernel, states[-1])
            states.append(intermediate_state)
            if len(observables) > 0: expectations.append(compute_expectations(intermediate_state, parameters))
        return EvolveResult(states, expectations)
    else:
        kernel, parameters = _create_kernel("time_evolution", hamiltonian, schedule, initial_state)
        final_state = cudaq_runtime.get_state(kernel)
        if len(observables) == 0: return EvolveResult(final_state)
        else: return EvolveResult(final_state, compute_expectations(final_state, parameters))

def evolve_async(hamiltonian: Operator, 
           dimensions: Mapping[int, int], 
           schedule: Schedule,
           initial_state: cudaq_runtime.State | Iterable[cudaq_runtime.State],
           collapse_operators: Iterable[Operator] = [],
           observables: Iterable[Operator] = [], 
           store_intermediate_results = False) -> AsyncEvolveResult | Iterable[AsyncEvolveResult]:
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
        raise NotImplementedError("time evolution is currently only supported on simulator targets")
    elif simulator == "nvidia-dynamics": # FIXME: update once we know the target name
        raise NotImplementedError(f"{simulator} backend does not yet exist")

    # Unless we are using cuSuperoperator for the execution, 
    # we can only handle qubits at this time.
    if any(dimension != 2 for dimension in dimensions.values()):
        raise ValueError("computing the time evolution is only possible for qubits; use the nvidia-dynamics target to simulate time evolution of arbitrary d-level systems")
    # Unless we are using cuSuperoperator for the execution, 
    # we cannot handle simulating the effect of collapse operators.
    if len(collapse_operators) > 0:
        raise ValueError("collapse operators can only be defined when using the nvidia-dynamics target")

    state_to_kernel = _state_to_kernel()
    def compute_expectations(cudaq_state: cudaq_runtime.State, parameters: Mapping[str, NumericType]) -> Iterable[cudaq_runtime.ObserveResult]:
        observable_spinops = [op._to_spinop(dimensions, **parameters) for op in observables]
        expectation_values: list[cudaq_runtime.ObserveResult] = []
        for observable in observable_spinops:
            expectation = cudaq_runtime.observe_async(state_to_kernel, observable, cudaq_state)
            expectation_values.append(expectation)
        return expectation_values

    if store_intermediate_results:
        evolution = _create_kernels("time_evolution", hamiltonian, schedule, initial_state)
        states, expectations = [], []
        for kernel, parameters in evolution:
            if len(states) == 0: intermediate_state = cudaq_runtime.get_state_async(kernel)
            else: # FIXME: can we manually create a AsyncStateResult to not wait here?
                # FIXME: inlining the expression to get the previous state here causes a segfault
                previous_state = states[-1].get()
                intermediate_state = cudaq_runtime.get_state_async(kernel, previous_state)
            states.append(intermediate_state)
            # FIXME: can we make this so that we don't have to get the state here?
            if len(observables) > 0: expectations.append(compute_expectations(intermediate_state.get(), parameters))
        return AsyncEvolveResult(states, expectations)
    else:
        kernel, parameters = _create_kernel("time_evolution", hamiltonian, schedule, initial_state)
        final_state = cudaq_runtime.get_state_async(kernel)
        if len(observables) > 0:
            # FIXME: can we make this so that we don't have to get the state here?
            return AsyncEvolveResult(final_state, compute_expectations(final_state.get(), parameters))
        return AsyncEvolveResult(final_state)
