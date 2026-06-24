# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import typing

from .helpers import NumericType
from .spin import SpinOperator, SpinOperatorTerm, SpinOperatorElement
from .fermion import FermionOperator, FermionOperatorTerm, FermionOperatorElement
from .boson import BosonOperator, BosonOperatorTerm, BosonOperatorElement
from .custom import MatrixOperator, MatrixOperatorTerm, MatrixOperatorElement
from .scalar import ScalarOperator

OperatorSum = MatrixOperator | SpinOperator | BosonOperator | FermionOperator
ProductOperator = MatrixOperatorTerm | SpinOperatorTerm | BosonOperatorTerm | FermionOperatorTerm
ElementaryOperator = SpinOperatorElement | BosonOperatorElement | FermionOperatorElement | MatrixOperatorElement

# Doc strings for type alias are not supported in Python.
# The string below hence merely serves to document it here;
# within the Python AST it is not associated with the type alias.
Operator = OperatorSum | ProductOperator | ScalarOperator
"""
Type of an arbitrary operator expression. 
Operator expressions cannot be used within quantum kernels, but 
they provide methods to convert them to data types that can.
"""


class RydbergHamiltonian:
    """
    Representation for the time-dependent Hamiltonian which is simulated by
    analog neutral-atom machines such as QuEra's Aquila and Pasqal's Fresnel.
    Ref: https://docs.aws.amazon.com/braket/latest/developerguide/braket-quera-submitting-analog-program-aquila.html#braket-quera-ahs-program-schema
    """

    def __init__(
        self,
        atom_sites: typing.Iterable[tuple[float, float]],
        amplitude: ScalarOperator,
        phase: ScalarOperator,
        delta_global: ScalarOperator,
        atom_filling: typing.Optional[typing.Iterable[int]] = [],
        delta_local: typing.Optional[tuple[ScalarOperator,
                                           typing.Iterable[float]]] = None):
        """
        Instantiate an operator consumable by `evolve` API using the supplied 
        parameters.
        
        Arguments:
            atom_sites: List of 2-d coordinates where the tweezers trap atoms.            
            amplitude: time and value points of driving amplitude, Omega(t).
            phase: time and value points of driving phase, phi(t).
            delta_global: time and value points of driving detuning, 
                          Delta_global(t).
            atom_filling: typing.Optional. Marks atoms that occupy the trap sites with
                          1, and empty sites with 0. If not provided, all are
                          set to 1, i.e. filled.
            delta_local: typing.Optional. A tuple of time and value points of the 
                         time-dependent factor of the local detuning magnitude,
                         Delta_local(t), and site-dependent factor of the local
                         detuning  magnitude, h_k, a dimensionless number 
                         between 0.0 and 1.0
        """
        if len(atom_filling) == 0:
            atom_filling = [1] * len(atom_sites)
        elif len(atom_sites) != len(atom_filling):
            raise ValueError(
                "Size of `atom_sites` and `atom_filling` must be equal")

        if delta_local is not None:
            raise NotImplementedError(
                "Local detuning is experimental feature not yet supported in CUDA-Q"
            )

        self.atom_sites = atom_sites
        self.atom_filling = atom_filling
        self.amplitude = amplitude
        self.phase = phase
        self.delta_global = delta_global
        self.delta_local = delta_local

        ## TODO [FUTURE]: Construct the Hamiltonian terms from the supplied parameters using
        # the following 'fixed' values: s_minus, s_plus, n_op, C6
        # Also, the spatial pattern of Omega(t), phi(t) and Delta_global(t) must be 'uniform'
