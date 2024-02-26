# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Stub implementation for quantum operations

from abc import abstractmethod
from typing import Callable

from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
qubit = cudaq_runtime.qubit
SpinOperator = cudaq_runtime.SpinOperator


def processQubitIds(opName, *args):
    """
    Return the qubit unique ID integers for a general tuple of 
    kernel arguments, where all arguments are assumed to be qubit-like 
    (`qvector`, `qview`, `qubit`).
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
    def __call__(cls, *args):
        """
        Invoke the quantum operation. The arguments can contain float parameters (of the
        correct number according to `get_num_parameters`) and quantum types (`qubit`, `qvector`, `qview`).
        """
        opName = cls.get_name()
        parameters = list(args)[:cls.get_num_parameters()]
        quantumArguments = list(args)[cls.get_num_parameters():]
        qubitIds = [q for q in processQubitIds(opName, *quantumArguments)]

        [
            cudaq_runtime.applyQuantumOperation(opName, parameters, [], [q],
                                                False, SpinOperator())
            for q in qubitIds
        ]

    @classmethod
    def ctrl(cls, *args):
        """
        Invoke the general controlled version of the quantum operation. 
        The arguments can contain float parameters (of the correct number according
        to `get_num_parameters`) and quantum types (`qubit`, `qvector`, `qview`).
        """
        opName = cls.get_name()
        parameters = list(args)[:cls.get_num_parameters()]
        quantumArguments = list(args)[cls.get_num_parameters():]
        qubitIds = processQubitIds(opName, *quantumArguments)
        controls = qubitIds[:len(qubitIds) - 1]
        targets = [qubitIds[-1]]

        for q in quantumArguments:
            if isinstance(q, qubit) and q.is_negated():
                x()(q)

        cudaq_runtime.applyQuantumOperation(opName, parameters, controls,
                                            targets, False, SpinOperator())
        for q in quantumArguments:
            if isinstance(q, qubit) and q.is_negated():
                x()(q)
                q.reset_negation()

    @classmethod
    def adj(cls, *args):
        """
        Invoke the general adjoint version of the quantum operation. 
        The arguments can contain float parameters (of the correct number according
        to `get_num_parameters`) and quantum types (`qubit`, `qvector`, `qview`).
        """
        opName = cls.get_name()
        parameters = list(args)[:cls.get_num_parameters()]
        quantumArguments = list(args)[cls.get_num_parameters():]
        qubitIds = [q for q in processQubitIds(opName, *quantumArguments)]

        [
            cudaq_runtime.applyQuantumOperation(opName,
                                                [-1 * p
                                                 for p in parameters], [], [q],
                                                False, SpinOperator())
            for q in qubitIds
        ]


# Define our quantum operations
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

ch = type('ch', (quantum_operation,), {'get_name': staticmethod(lambda: 'ch')})
cx = type('cx', (quantum_operation,), {'get_name': staticmethod(lambda: 'cx')})
cy = type('cy', (quantum_operation,), {'get_name': staticmethod(lambda: 'cy')})
cz = type('cz', (quantum_operation,), {'get_name': staticmethod(lambda: 'cz')})
cs = type('cs', (quantum_operation,), {'get_name': staticmethod(lambda: 'cs')})
ct = type('ct', (quantum_operation,), {'get_name': staticmethod(lambda: 'ct')})

crx = type(
    'crx', (quantum_operation,), {
        'get_name': staticmethod(lambda: 'crx'),
        'get_num_parameters': staticmethod(lambda: 1)
    })
cry = type(
    'cry', (quantum_operation,), {
        'get_name': staticmethod(lambda: 'cry'),
        'get_num_parameters': staticmethod(lambda: 1)
    })
crz = type(
    'crz', (quantum_operation,), {
        'get_name': staticmethod(lambda: 'crz'),
        'get_num_parameters': staticmethod(lambda: 1)
    })
cr1 = type(
    'r1', (quantum_operation,), {
        'get_name': staticmethod(lambda: 'cr1'),
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
