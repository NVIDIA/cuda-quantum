# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence, Mapping, Tuple
from ..operators import Operator, SuperOperator
from .schedule import Schedule

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
        self.schedule = None
        self.hamiltonian = None
        self.stepper = None
        self.collapse_operators = None
        self.super_op = None
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

    def set_system(self,
                   dimensions: Mapping[int, int],
                   schedule: Schedule,
                   hamiltonian: Operator | SuperOperator | Sequence[Operator] |
                   Sequence[SuperOperator],
                   collapse_operators: Sequence[Operator] |
                   Sequence[Sequence[Operator]] = []):
        self.dimensions = tuple(dimensions[d] for d in range(len(dimensions)))
        self.schedule = schedule
        if isinstance(
                hamiltonian,
                SuperOperator) or (isinstance(hamiltonian, Sequence) and
                                   isinstance(hamiltonian[0], SuperOperator)):
            self.super_op = hamiltonian
        else:
            self.hamiltonian = hamiltonian

        self.collapse_operators = collapse_operators
        self.stepper = None

    @abstractmethod
    def integrate(self, t):
        """
        Evolve to t.

        Before calling `integrate` for the first time, the initial state should
        be set with `set_state`.
        """
        pass

    def get_state(self) -> Tuple[float, TState]:
        """
        Obtain the state of the integrator as a pair (t, state).
        """
        return (self.t, self.state)

    def support_distributed_state(self):
        """
        Returns true if the integrator supports distributed state else returns false. Default is set to false.
        """
        return False

    def is_native(self):
        """
        Returns true if the integrator is a native C-API implementation. Default is set to false.
        """
        return False
