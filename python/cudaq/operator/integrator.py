
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence, Mapping
from .expressions import Operator


TState = TypeVar('TState')

class BaseTimeStepper(ABC, Generic[TState]):
    @abstractmethod
    def compute(self, state: TState, t: float):
        pass

class BaseIntegrator(ABC, Generic[TState]):
    """
    An abstract wrapper around ODE integrator to ensure a common interface for master equation solver usage.
    """
    integrator_options = {}
    def __init__(self, **kwargs):
        self.state = None
        self.integrator_options.update(kwargs)
        self.t = None
        self.dimensions = None
        self.hamiltonian = None
        self.stepper = None
        self.collapse_operators = None
        self.__post_init__()
    
    @abstractmethod
    def __post_init__(self):
        """
        Initialize the integrator: any implementation-specific initialization actions
        """
        pass

    def set_state(self, state: TState, t: float = 0.0):
        self.state = state
        self.t = t

    def set_system(self, dimensions: Mapping[int, int], hamiltonian: Operator, collapse_operators: Sequence[Operator] = []):
        self.dimensions = dimensions
        self.hamiltonian = hamiltonian
        self.collapse_operators = collapse_operators

    @abstractmethod
    def integrate(self, t):
        """
        Evolve to t.

        Before calling `integrate` for the first time, the initial state should
        be set with `set_state`.
        """  
        pass


    def get_state(self) -> tuple[float, TState]:
        """
        Obtain the state of the integrator as a pair (t, state).
        """
        return (self.t, self.state)

