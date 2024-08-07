from __future__ import annotations
import numpy
from collections.abc import Mapping, Sequence
from typing import Optional, Iterable

from .expressions import Operator
from .mlir._mlir_libs._quakeDialects import cudaq_runtime
from .schedule import Schedule

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
                 expectation: Optional[NDArray[numpy.complexfloating] | Iterable[NDArray[numpy.complexfloating]]] = None) -> None:
        """
        Instantiates an EvolveResult representing the output generated when evolving a single 
        initial state under a set of operators. See `cudaq.evolve` for more detail.

        Arguments:
            state: A single state or a sequence of states of a quantum system. If a single 
                state is given, it represents the final state of the system after time evolution.
                If a sequence of states are given, they represent the state of the system after
                each steps in the schedule specified in the call to `cudaq.evolve`.
            expectation: A single one-dimensional array of complex values or a sequence of 
                one-dimensional arrays of complex values representing the expectation values
                computed during the evolution. If a single array is provided, it contains the
                expectation values computed at the end of the evolution. If a sequence of arrays
                is given, they represent the expectation values computed at each step in the 
                schedule passed to `cudaq.evolve`.
        """
        # This implementation is just a mock up - probably not very robust.
        *_, final_state = iter(state) # assumes cudaq.State is iterable - check if the type check here works
        if isinstance(final_state, cudaq_runtime.State):
            self._states = state
            self._final_state = final_state
        else:
            self._states = None
            self._final_state = state
        if expectation is None:
            self._expectation_values = None
            self._final_expectation = None
        else:
            *_, final_expectation = iter(expectation)
            if isinstance(final_expectation, numpy.ndarray):
                if self._states is None:
                    raise ValueError("intermediate states were defined but no intermediate expectation values are provided")
                self._expectation_values = expectation
                self._final_expectation = final_expectation
            else:
                self._expectation_values = None
                self._final_expectation = expectation

    @property
    def intermediate_states(self: EvolveResult) -> Optional[Iterable[cudaq_runtime.State]]:
        """
        Stores all intermediate states, meaning the state after each step in a defined 
        schedule, produced by a call to `cudaq.evolve`, including the final state. 
        This property is only populated if the corresponding saving intermediate results 
        was requested in the call to `cudaq.evolve`.
        """
        return self._states

    @property
    def final_state(self: EvolveResult) -> cudaq_runtime.State:
        """
        Stores the final state produced by a call to `cudaq.evolve`.
        Represent the state of a quantum system after time evolution under a set of 
        operators, see the `cudaq.evolve` documentation for more detail.
        """
        return self._state

    @property
    def expectation_values(self: EvolveResult) -> Optional[Iterable[NDArray[numpy.complexfloating]]]:
        """
        Stores the expectation values at each step in the schedule produced by a call to 
        `cudaq.evolve`, including the final expectation values. Each expectation value 
        corresponds to one observable provided in the `cudaq.evolve` call. 
        This property is only populated if the corresponding saving intermediate results 
        was requested in the call to `cudaq.evolve`. This value will be None if no 
        intermediate results were requested, or if no observables were specified in the 
        call to `cudaq.evolve`.
        """
        return self._expectation_values

    @property
    def final_expectation(self: EvolveResult) -> Optional[NDArray[numpy.complexfloating]]:
        """
        Stores the final expectation values produced by a call to `cudaq.evolve`.
        Each expectation value corresponds to one observable provided in the `cudaq.evolve` call. 
        This value will be None if no observables were specified in the call to `cudaq.evolve`.
        """
        return self._final_expectation

# To be implemented in C++ and bindings will be generated.
class AsyncEvolveResult:
    """
    Stores the execution data from an invocation of `cudaq.evolve_async`.
    """

    def __init__(handle: str) -> None:
        """
        Creates a class instance that can be used to retrieve the evolution
        result produces by an calling the asynchronously executing function
        `cudaq.evolve_async`. It models a future-like type whose 
        `EvolveResult` may be accessed via an invocation of the `get`
        method. 
        """
        raise NotImplementedError()

    def get(self: AsyncEvolveResult) -> EvolutionResult:
        """
        Retrieves the `EvolveResult` from the asynchronous evolve execution.
        This causes the current thread to wait until the time evolution
        execution has completed. 
        """
        raise NotImplementedError()

    def __str__(self: AsyncEvolveResult) -> str:
        pass


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
    operators. 

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
    raise NotImplementedError()

def evolve(hamiltonian: Operator, 
           dimensions: Mapping[int, int], 
           schedule: Schedule,
           initial_state: cudaq.State | Iterable[cudaq.States],
           collapse_operators: Iterable[Operator] = [],
           observables: Iterable[Operator] = [], 
           store_intermediate_results = False) -> AsyncEvolveResult | Iterable[AsyncEvolveResult]:
    """
    Asynchronously computes the time evolution of one or more initial state(s) 
    under the defined operators. See `cudaq.evolve` for more details about the
    parameters passed here.
    
    Returns:
        The handle to a single evolution result if a single initial state is provided, 
        or a sequence of handles to the evolution results representing the data computed 
        during the evolution of each initial state. See the `EvolveResult` for more 
        information about the data computed during evolution.
    """
    raise NotImplementedError()
