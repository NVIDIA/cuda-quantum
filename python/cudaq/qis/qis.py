# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Stub definitions for quantum operations


def raise_error():
    raise RuntimeError("This function is here for documentation purposes only.\
         Its implementation is populated by the CUDA-Q toolchain.")


def h(*args):
    """
    This operation is a rotation by π about the X+Z axis, and enables one to create a superposition of computational basis states.
    """
    raise_error


def x(*args):
    """
    This operation implements the transformation defined by the Pauli-X matrix. It is also known as the quantum version of a NOT-gate.
    """
    raise_error


def y(*args):
    """
    This operation implements the transformation defined by the Pauli-Y matrix.
    """
    raise_error


def z(*args):
    """
    This operation implements the transformation defined by the Pauli-Z matrix.
    """
    raise_error


def s(*args):
    """
    This operation applies to its target a rotation by π/2 about the Z axis.
    """
    raise_error


def t(*args):
    """
    This operation applies to its target a π/4 rotation about the Z axis.
    """
    raise_error


def rx(*args):
    """
    This operation is an arbitrary rotation about the X axis.
    """
    raise_error


def ry(*args):
    """
    This operation is an arbitrary rotation about the Y axis.
    """
    raise_error


def rz(*args):
    """
    This operation is an arbitrary rotation about the Z axis.
    """
    raise_error


def r1(*args):
    """
    This operation is an arbitrary rotation about the |1> state.
    """
    raise_error


def ch(*args):
    """
    This operation applies a controlled-h operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def cx(*args):
    """
    This operation applies a controlled-x operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def cy(*args):
    """
    This operation applies a controlled-y operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def cz(*args):
    """
    This operation applies a controlled-z operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def ct(*args):
    """
    This operation applies a controlled-t operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def cs(*args):
    """
    This operation applies a controlled-s operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def sdg(*args):
    """
    This operation applies a rotation on the z-axis of negative 90 degrees to the given target qubit/s.
    """
    raise_error


def tdg(*args):
    """
    This operation applies a rotation on the z-axis of negative 45 degrees to the given target qubit/s.
    """
    raise_error


def crx(*args):
    """
    This operation applies a `controlled-rx` operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def cry(*args):
    """
    This operation applies a `controlled-ry` operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def crz(*args):
    """
    This operation applies a `controlled-rz` operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def cr1(*args):
    """
    This operation applies a `controlled-r1` operation to the given target qubit, with the provided control qubit/s.
    """
    raise_error


def swap(*args):
    """
    This operation swaps the states of the provided qubits. Can be controlled on any number of qubits via the `ctrl` method.
    """
    raise_error


def exp_pauli(theta, qubits, pauliWord):
    """
    Apply a general Pauli tensor product rotation, `exp(i theta P)`,
    """
    raise_error


def mz(*args, register_name=''):
    """
    Measure the qubit along the z-axis.
    """
    raise_error


def my(*args, register_name=''):
    """
    Measure the qubit along the y-axis.
    """
    raise_error


def mx(*args, register_name=''):
    """
    Measure the qubit along the x-axis.
    """
    raise_error


def adjoint(kernel, *args):
    """
    Apply the adjoint of the given kernel at the provided runtime arguments.
    """
    raise_error


def control(kernel, controls, *args):
    """
    Apply the general control version of the given kernel at the provided runtime arguments.
    """
    raise_error


def compute_action(compute, action):
    """
    Apply the U V U^dag given U and V unitaries.
    """
    raise_error
