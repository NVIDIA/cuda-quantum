# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

# The qudit level must be explicitly defined
globalQuditLevel = None


@dataclass
class PyQudit:
    """
    A data structure to represent `qudit` which models a general d-level 
    quantum system.

    Args:
        level (int): An integer representing the level of quantum system to 
            which this qudit belongs to.
        id (int): An integer representing the unique identifier for this qudit
            in the system.
    """
    level: int
    id: int

    def __del__(self):
        try:
            cudaq_runtime.photonics.release_qudit(self.level, self.id)
        except Exception:
            pass


def _is_qudit_type(q: any) -> bool:
    """
    Utility function to check whether the input argument is instance of 
    `PyQudit` class.

    Returns:
        bool: `True` if input argument is instance or a list of `PyQudit` 
               class, else `False`
    """
    if isinstance(q, PyQudit):
        return True
    if isinstance(q, List):
        return all(isinstance(i, PyQudit) for i in q)
    return False


def _check_args(q: any):
    """
    Utility function to verify the arguments to a photonic quantum operation.

    Args:
        q: A single instance or a list of objects of `PyQudit` class.

    Raises:
        RuntimeError: If the qudit level is not set.
        Exception: If input argument is not instance of `PyQudit` class.
    """
    if globalQuditLevel is None:
        raise RuntimeError(
            "Qudit level not set. Define a qudit (`qudit(level=N)`) or list of qudits."
        )

    if not _is_qudit_type(q):
        raise Exception(
            "Invalid quantum type. Use qudit (`qudit(level=N)`) or a list of qudits."
        )


def qudit(level: int) -> PyQudit:
    """
    API to define a d-level qudit in a `PhotonicsHandler`. All the qudits within
    a kernel are of the same level.

    Args:
        level (int): An integer representing the level of quantum system.
    
    Returns:
        An instance of `PyQudit`.
    
    Raises:
        RuntimeError: If a qudit of level different than one already defined 
            in the kernel is requested.
    """
    global globalQuditLevel

    if globalQuditLevel is None:
        globalQuditLevel = level
    elif level != globalQuditLevel:
        raise RuntimeError("The qudits must be of same level within a kernel.")

    id = cudaq_runtime.photonics.allocate_qudit(globalQuditLevel)
    return PyQudit(globalQuditLevel, id)


def plus(qudit: PyQudit):
    """
    Apply plus gate on the input qudit. 
    U|0> -> |1>, U|1> -> |2>, ..., and U|d> -> |0>

    Args: 
        qudit: An instance of `PyQudit` class.
    
    Raises:
        RuntimeError: If the qudit level is not set.
        Exception: If input argument is not instance of `PyQudit` class.
    """
    _check_args(qudit)
    cudaq_runtime.photonics.apply_operation('plusGate', [],
                                            [[qudit.level, qudit.id]])


def phase_shift(qudit: PyQudit, phi: float):
    """
    Apply phase shift gate.
    TBD
    
    Args:
        qudit: An instance of `PyQudit` class.
        phi: A floating point number for the angle.

    Raises:
        RuntimeError: If the qudit level is not set.
        Exception: If input argument is not instance of `PyQudit` class.
    """
    _check_args(qudit)
    cudaq_runtime.photonics.apply_operation('phaseShiftGate', [phi],
                                            [[qudit.level, qudit.id]])


def beam_splitter(q: PyQudit, r: PyQudit, theta: float):
    """
    Apply beam splitter gate.
    TBD
    
    Args:
        q: An instance of `PyQudit` class TBD.
        r: An instance of `PyQudit` class TBD.
        theta: A floating point number for the angle.
    
    Raises:
        RuntimeError: If the qudit level is not set.
        Exception: If input argument is not instance of `PyQudit` class.
    """
    _check_args([q, r])
    cudaq_runtime.photonics.apply_operation('beamSplitterGate', [theta],
                                            [[q.level, q.id], [r.level, r.id]])


def mz(qudits: PyQudit | List[PyQudit], register_name=''):
    """
    Measure a single qudit or list of qudits.

    Args:
        qudits: A single instance or a list of objects of `PyQudit` class.
    
    Returns:
        Measurement results.
    
    Raises:
        RuntimeError: If the qudit level is not set.
        Exception: If input argument is not instance of `PyQudit` class.
    """
    _check_args(qudits)
    if isinstance(qudits, PyQudit):
        return cudaq_runtime.photonics.measure(qudits.level, qudits.id,
                                               register_name)
    if isinstance(qudits, List):
        return [
            cudaq_runtime.photonics.measure(q.level, q.id, register_name)
            for q in qudits
        ]


class PhotonicsHandler(object):
    """
    The `PhotonicsHandler` class serves as to process CUDA-Q kernels for the 
    `photonics` target.
    The target must be set to `photonics` prior to invoking a `PhotonicsHandler`.

    The quantum operations in this kernel apply to qudits defined by 
    `qudit(level=N)` or a list of qudits. The qudits within a kernel must be of
    the same level.

    Allowed quantum operations are: `plus`, `phase_shift`, `beam_splitter`, and `mz`.
    """

    def __init__(self, function):

        if 'photonics' != cudaq_runtime.get_target().name:
            raise RuntimeError(
                "A photonics kernel can only be used with 'photonics' target.")

        global globalQuditLevel
        globalQuditLevel = None

        self.kernelFunction = function

        self.kernelFunction.__globals__['qudit'] = qudit
        self.kernelFunction.__globals__['plus'] = plus
        self.kernelFunction.__globals__['phase_shift'] = phase_shift
        self.kernelFunction.__globals__['beam_splitter'] = beam_splitter
        self.kernelFunction.__globals__['mz'] = mz

    def __call__(self, *args):
        return self.kernelFunction(*args)
