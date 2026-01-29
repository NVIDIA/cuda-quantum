# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


@dataclass(slots=True)
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


class QuditManager(object):
    """
    A class to explicitly manage resource allocation for qudits within a 
    `PhotonicsKernel`.    
    """
    qudit_level = None
    allocated_ids = []

    @classmethod
    def reset(cls):
        cls.qudit_level = None
        cls.allocated_ids = []

    @classmethod
    def allocate(cls, level: int):
        if cls.qudit_level is None:
            cls.qudit_level = level
        elif level != cls.qudit_level:
            raise RuntimeError(
                "The qudits must be of same level within a kernel.")
        id = cudaq_runtime.photonics.allocate_qudit(cls.qudit_level)
        cls.allocated_ids.append(id)
        return PyQudit(cls.qudit_level, id)

    def __enter__(cls):
        cls.reset()

    def __exit__(cls, exc_type, exc_val, exc_tb):
        while cls.allocated_ids:
            cudaq_runtime.photonics.release_qudit(cls.allocated_ids.pop(),
                                                  cls.qudit_level)
        cls.reset()


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
    if QuditManager.qudit_level is None:
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
    return QuditManager.allocate(level)


def create(qudit: PyQudit):
    """
    Apply create gate on the input qudit.
    U|0> -> |1>, U|1> -> |2>, ..., and U|d> -> |d>

    Args:
        qudit: An instance of `PyQudit` class.

    Raises:
        RuntimeError: If the qudit level is not set.
        Exception: If input argument is not instance of `PyQudit` class.
    """
    _check_args(qudit)
    cudaq_runtime.photonics.apply_operation("create", [],
                                            [[qudit.level, qudit.id]])


def annihilate(qudit: PyQudit):
    """
    Apply annihilate gate on the input qudit.
    U|0> -> |0>, U|1> -> |0>, ..., and U|d> -> |d-1>

    Args:
        qudit: An instance of `PyQudit` class.

    Raises:
        RuntimeError: If the qudit level is not set.
        Exception: If input argument is not instance of `PyQudit` class.
    """
    _check_args(qudit)
    cudaq_runtime.photonics.apply_operation("annihilate", [],
                                            [[qudit.level, qudit.id]])


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
    cudaq_runtime.photonics.apply_operation("plus", [],
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
    cudaq_runtime.photonics.apply_operation("phase_shift", [phi],
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
    cudaq_runtime.photonics.apply_operation("beam_splitter", [theta],
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
    `orca-photonics` target.
    The target must be set to `orca-photonics` prior to invoking a `PhotonicsHandler`.

    The quantum operations in this kernel apply to qudits defined by 
    `qudit(level=N)` or a list of qudits. The qudits within a kernel must be of
    the same level.

    Allowed quantum operations are: `create`, `annihilate`, `plus`,
    `phase_shift`, `beam_splitter`, and `mz`.
    """

    def __init__(self, function):

        if 'orca-photonics' != cudaq_runtime.get_target().name:
            raise RuntimeError(
                "This kernel can only be used with 'orca-photonics' target.")

        QuditManager.reset()
        self.kernelFunction = function

        self.kernelFunction.__globals__["qudit"] = qudit
        self.kernelFunction.__globals__["create"] = create
        self.kernelFunction.__globals__["annihilate"] = annihilate
        self.kernelFunction.__globals__["plus"] = plus
        self.kernelFunction.__globals__["phase_shift"] = phase_shift
        self.kernelFunction.__globals__["beam_splitter"] = beam_splitter
        self.kernelFunction.__globals__["mz"] = mz

    def __call__(self, *args):
        with QuditManager():
            return self.kernelFunction(*args)
