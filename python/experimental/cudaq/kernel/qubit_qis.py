# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Library Mode QIS
# qubit, qvector, qview all defined in C++
# (better tracking of construction / destruction)

from abc import abstractmethod, ABCMeta
import inspect
import numpy as np
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime
from .utils import globalRegisteredUnitaries
from typing import Callable

qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
qubit = cudaq_runtime.qubit
SpinOperator = cudaq_runtime.SpinOperator


def processQubitIds(opName, *args):
    """
    Return the qubit unique ID integers for a general tuple of 
    kernel arguments, where all arguments are assumed to be qubit-like 
    (qvector, qview, qubit).
    """
    qubitIds = []
    for a in args:
        if isinstance(a, qubit):
            qubitIds.append(a.id())
        elif isinstance(a, qvector) or isinstance(a, qview):
            [qubitIds.append(q.id()) for q in a]
        else:
            raise Exception(
                "invalid argument type passed to {}.__call__".format(opName))
    return qubitIds


class quantum_operation(object):
    """
    A quantum_operation provides a base class interface for invoking 
    a specific quantum gate, as well as controlled and adjoint versions 
    of the gate.
    """

    @staticmethod
    @abstractmethod
    def get_name():
        """
        Return the name of this operation.
        """
        pass

    @classmethod
    def get_num_parameters(cls):
        """
        Return the number of rotational parameters this operation requires.
        """
        return 0

    @classmethod
    def get_unitary(cls):
        return []

    @classmethod
    def __validateAndProcessUnitary(cls, unitary):
        if not hasattr(unitary, 'shape'):
            raise RuntimeError('custom unitary must be a numpy array.')

        shape = unitary.shape
        if len(unitary.shape) != 2:
            raise RuntimeError("custom unitary must be a 2D numpy array.")

        if shape[0] != shape[1]:
            raise RuntimeError("custom unitary must be square matrix.")

        numTargets = np.log2(shape[0])

        # Flatten the array
        unitary = list(unitary.flat)
        return unitary, numTargets

    @classmethod
    def __call__(cls, *args):
        """
        Invoke the quantum operation. The args can contain float parameters (of the
        correct number according to get_num_parameters) and quantum types (qubit, qvector, qview).
        """
        opName = cls.get_name()
        unitary = cls.get_unitary()
        parameters = list(args)[:cls.get_num_parameters()]
        quantumArguments = list(args)[cls.get_num_parameters():]
        qubitIds = [q for q in processQubitIds(opName, *quantumArguments)]

        # If the unitary is callable, evaluate it
        if isinstance(unitary, Callable):
            unitary = unitary(*parameters)

        if len(unitary) > 0:
            unitary, numTargets = quantum_operation.__validateAndProcessUnitary(
                unitary)
            # Disable operation broadcasting for custom unitaries
            if numTargets != len(qubitIds):
                raise RuntimeError(
                    "incorrect number of target qubits provided.")

            cudaq_runtime.applyQuantumOperation(opName, parameters,
                                                [], qubitIds, False,
                                                SpinOperator(), unitary)
            return

        # Not a custom unitary, handle basic quantum operation,
        # with optional broadcasting
        [
            cudaq_runtime.applyQuantumOperation(opName, parameters, [], [q],
                                                False, SpinOperator())
            for q in qubitIds
        ]

    @classmethod
    def ctrl(cls, *args):
        """
        Invoke the general controlled version of the quantum operation. 
        The args can contain float parameters (of the correct number according
        to get_num_parameters) and quantum types (qubit, qvector, qview).
        """
        opName = cls.get_name()
        unitary = cls.get_unitary()
        parameters = list(args)[:cls.get_num_parameters()]
        quantumArguments = list(args)[cls.get_num_parameters():]
        qubitIds = processQubitIds(opName, *quantumArguments)
        controls = qubitIds[:len(qubitIds) - 1]
        targets = [qubitIds[-1]]
        # If the unitary is callable, evaluate it
        if isinstance(unitary, Callable):
            unitary = unitary(*parameters)

        if len(unitary) > 0:
            unitary, numTargets = quantum_operation.__validateAndProcessUnitary(
                unitary)
            controls = qubitIds[:len(qubitIds) - int(numTargets)]
            targets = qubitIds[-int(numTargets):]
            # Disable operation broadcasting for custom unitaries
            if numTargets != len(targets):
                raise RuntimeError(
                    "incorrect number of target qubits provided.")

        for q in quantumArguments:
            if isinstance(q, qubit) and q.is_negated():
                x()(q)

        cudaq_runtime.applyQuantumOperation(opName, parameters,
                                            controls, targets, False,
                                            SpinOperator(), unitary)
        for q in quantumArguments:
            if isinstance(q, qubit) and q.is_negated():
                x()(q)
                q.reset_negation()

    @classmethod
    def adj(cls, *args):
        """
        Invoke the general adjoint version of the quantum operation. 
        The args can contain float parameters (of the correct number according
        to get_num_parameters) and quantum types (qubit, qvector, qview).
        """
        opName = cls.get_name()
        unitary = cls.get_unitary()
        parameters = list(args)[:cls.get_num_parameters()]
        quantumArguments = list(args)[cls.get_num_parameters():]
        qubitIds = [q for q in processQubitIds(opName, *quantumArguments)]

        # If the unitary is callable, evaluate it
        if isinstance(unitary, Callable):
            unitary = unitary(*parameters)

        if len(unitary) > 0:
            unitary, numTargets = quantum_operation.__validateAndProcessUnitary(
                unitary)
            # Disable operation broadcasting for custom unitaries
            if numTargets != len(qubitIds):
                raise RuntimeError(
                    "incorrect number of target qubits provided.")

            cudaq_runtime.applyQuantumOperation(opName,
                                                [-1 * p for p in parameters],
                                                [], qubitIds, False,
                                                SpinOperator(), unitary)
            return

        # Not a custom unitary, handle basic quantum operation,
        # with optional broadcasting
        [
            cudaq_runtime.applyQuantumOperation(opName,
                                                [-1 * p
                                                 for p in parameters], [], [q],
                                                False, SpinOperator())
            for q in qubitIds
        ]


# Define our quantum operatations
h = type('h', (quantum_operation,), {'get_name': staticmethod(lambda: 'h')})
x = type('x', (quantum_operation,), {'get_name': staticmethod(lambda: 'x')})
y = type('y', (quantum_operation,), {'get_name': staticmethod(lambda: 'y')})
z = type('z', (quantum_operation,), {'get_name': staticmethod(lambda: 'z')})
s = type('s', (quantum_operation,), {'get_name': staticmethod(lambda: 's')})
t = type('t', (quantum_operation,), {'get_name': staticmethod(lambda: 't')})

rx = type(
    'rx', (quantum_operation,), {
        'get_name': staticmethod(lambda: 'rx'),
        'get_num_parameters': staticmethod(lambda: 1)
    })
ry = type(
    'ry', (quantum_operation,), {
        'get_name': staticmethod(lambda: 'ry'),
        'get_num_parameters': staticmethod(lambda: 1)
    })
rz = type(
    'rz', (quantum_operation,), {
        'get_name': staticmethod(lambda: 'rz'),
        'get_num_parameters': staticmethod(lambda: 1)
    })
r1 = type(
    'r1', (quantum_operation,), {
        'get_name': staticmethod(lambda: 'r1'),
        'get_num_parameters': staticmethod(lambda: 1)
    })


class swap(object):
    """
    The swap operation. Can be controlled on any number of qubits via the `ctrl` method.
    """

    @staticmethod
    def __call__(first, second):
        cudaq_runtime.applyQuantumOperation(
            __class__.__name__, [], [], [first.id(), second.id()])

    @staticmethod
    def ctrl(*args):
        qubitIds = processQubitIds(__class__.__name__, *args)
        cudaq_runtime.applyQuantumOperation(__class__.__name__, [],
                                            qubitIds[:len(qubitIds) - 2],
                                            [qubitIds[-2], qubitIds[-1]])


def exp_pauli(theta, qubits, pauliWord):
    """
    Apply a general Pauli tensor product rotation, `exp(i theta P)`,
    """
    cudaq_runtime.applyQuantumOperation('exp_pauli', [theta], [],
                                        [q.id() for q in qubits], False,
                                        SpinOperator.from_word(pauliWord))


def mz(*args, register_name=''):
    """
    Measure the qubit along the z-axis.
    """
    qubitIds = processQubitIds('mz', *args)
    res = [cudaq_runtime.measure(q, register_name) for q in qubitIds]
    if len(res) == 1:
        return res[0]
    else:
        return res


def my(*args, register_name=''):
    """
    Measure the qubit along the y-axis.
    """
    s.adj(*args)
    h()(*args)
    return mz(*args, register_name)


def mx(*args, register_name=''):
    """
    Measure the qubit along the x-axis.
    """
    h()(*args)
    return mz(*args, register_name)


def adjoint(kernel, *args):
    """
    Apply the adjoint of the given kernel at the provided runtime arguments.
    """
    if not isinstance(kernel, Callable):
        raise RuntimeError(
            "cudaq.adjoint: first argument must be a callable kernel.")

    cudaq_runtime.startAdjointRegion()
    kernel(*args)
    cudaq_runtime.endAdjointRegion()


def control(kernel, controls, *args):
    """
    Apply the general control version of the given kernel at the provided runtime arguments.
    """
    if not isinstance(kernel, Callable):
        raise RuntimeError(
            "cudaq.control: first argument must be a callable kernel.")

    cudaq_runtime.startCtrlRegion([c.id() for c in controls])
    kernel(*args)
    cudaq_runtime.endCtrlRegion(len(controls))


def compute_action(compute, action):
    """
    Apply the U V U^dag given U and V unitaries.
    """
    compute()
    action()
    adjoint(compute)


def register_operation(unitary, operation_name=None):
    global globalRegisteredUnitaries
    """
    Register a new quantum operation at runtime. Users must 
    provide the unitary matrix as a 2D NumPy array. The operation 
    name is inferred from the name of the assigned variable. 

    .. code:: python 

        myOp = cudaq.register_operation(unitary)

        @cudaq.kernel
        def kernel():
            ...
            myOp(...)
            ...

    """
    if operation_name == None:
        lastFrame = inspect.currentframe().f_back
        frameInfo = inspect.getframeinfo(lastFrame)
        codeContext = frameInfo.code_context[0]
        if not '=' in codeContext:
            raise RuntimeError(
                "[register_operation] operation_name not given and variable name not set."
            )
        operation_name = codeContext.split('=')[0].strip()

    numParameters = 0
    if isinstance(unitary, Callable):
        numParameters = len(inspect.getfullargspec(unitary).args)

    # register a new function for kernels of the given
    # name, have it apply the unitary data
    registeredOp = type(
        operation_name, (quantum_operation,), {
            'get_name': staticmethod(lambda: operation_name),
            'get_unitary': staticmethod(lambda: unitary),
            'get_num_parameters': staticmethod(lambda: numParameters)
        })

    # Register the operation name so JIT AST can
    # get it.
    globalRegisteredUnitaries[operation_name] = unitary
    return registeredOp()
