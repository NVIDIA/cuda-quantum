# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, inspect, numpy, operator, types, uuid
from cudaq.operator import *
from typing import Any, Optional
import os, pytest

@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()

def test_all():
    dims = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}

    print(f'pauliX(1): {spin.x(1).to_matrix(dims)}')
    print(f'pauliY(2): {spin.y(2).to_matrix(dims)}')

    print(f'pauliZ(0) * pauliZ(0): {(spin.z(0) * spin.z(0)).to_matrix(dims)}')
    print(f'pauliZ(0) * pauliZ(1): {(spin.z(0) * spin.z(1)).to_matrix(dims)}')
    print(f'pauliZ(0) * pauliY(1): {(spin.z(0) * spin.y(1)).to_matrix(dims)}')

    op1 = ProductOperator([spin.x(0), spin.i(1)])
    op2 = ProductOperator([spin.i(0), spin.x(1)])
    print(f'pauliX(0) + pauliX(1): {op1.to_matrix(dims) + op2.to_matrix(dims)}')
    op3 = ProductOperator([spin.x(1), spin.i(0)])
    op4 = ProductOperator([
        spin.i(1),
        spin.x(0),
    ])
    print(f'pauliX(1) + pauliX(0): {op1.to_matrix(dims) + op2.to_matrix(dims)}')

    print(f'pauliX(0) + pauliX(1): {(spin.x(0) + spin.x(1)).to_matrix(dims)}')
    print(f'pauliX(0) * pauliX(1): {(spin.x(0) * spin.x(1)).to_matrix(dims)}')
    print(
        f'pauliX(0) * pauliI(1) * pauliI(0) * pauliX(1): {(op1 * op2).to_matrix(dims)}'
    )

    print(f'pauliX(0) * pauliI(1): {op1.to_matrix(dims)}')
    print(f'pauliI(0) * pauliX(1): {op2.to_matrix(dims)}')
    print(
        f'pauliX(0) * pauliI(1) + pauliI(0) * pauliX(1): {(op1 + op2).to_matrix(dims)}'
    )

    op5 = spin.x(0) * spin.x(1)
    op6 = spin.z(0) * spin.z(1)
    print(f'pauliX(0) * pauliX(1): {op5.to_matrix(dims)}')
    print(f'pauliZ(0) * pauliZ(1): {op6.to_matrix(dims)}')
    print(
        f'pauliX(0) * pauliX(1) + pauliZ(0) * pauliZ(1): {(op5 + op6).to_matrix(dims)}'
    )

    op7 = spin.x(0) + spin.x(1)
    op8 = spin.z(0) + spin.z(1)
    print(f'pauliX(0) + pauliX(1): {op7.to_matrix(dims)}')
    print(f'pauliZ(0) + pauliZ(1): {op8.to_matrix(dims)}')
    print(
        f'pauliX(0) + pauliX(1) + pauliZ(0) + pauliZ(1): {(op7 + op8).to_matrix(dims)}'
    )
    print(
        f'(pauliX(0) + pauliX(1)) * (pauliZ(0) + pauliZ(1)): {(op7 * op8).to_matrix(dims)}'
    )

    print(
        f'pauliX(0) * (pauliZ(0) + pauliZ(1)): {(spin.x(0) * op8).to_matrix(dims)}'
    )
    print(
        f'(pauliZ(0) + pauliZ(1)) * pauliX(0): {(op8 * spin.x(0)).to_matrix(dims)}'
    )

    op9 = spin.z(1) + spin.z(2)
    print(
        f'(pauliX(0) + pauliX(1)) * pauliI(2): {numpy.kron(op7.to_matrix(dims), spin.i(2).to_matrix(dims))}'
    )
    print(
        f'(pauliX(0) + pauliX(1)) * pauliI(2): {(op7 * spin.i(2)).to_matrix(dims)}'
    )
    print(
        f'(pauliX(0) + pauliX(1)) * pauliI(2): {(spin.i(2) * op7).to_matrix(dims)}'
    )
    print(
        f'pauliI(0) * (pauliZ(1) + pauliZ(2)): {numpy.kron(spin.i(0).to_matrix(dims), op9.to_matrix(dims))}'
    )
    print(
        f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(op7 * op9).to_matrix(dims)}'
    )

    so0 = ScalarOperator(lambda: 1.0j)
    print(f'Scalar op (t -> 1.0)(): {so0.to_matrix()}')

    so1 = ScalarOperator(lambda t: t)
    print(f'Scalar op (t -> t)(1.): {so1.to_matrix(t = 1.0)}')
    print(
        f'Trivial prod op (t -> t)(1.): {(ProductOperator([so1])).to_matrix({}, t = 1.)}'
    )
    print(
        f'Trivial prod op (t -> t)(2.): {(ProductOperator([so1])).to_matrix({}, t = 2.)}'
    )

    print(
        f'(t -> t)(1j) * pauliX(0): {(so1 * spin.x(0)).to_matrix(dims, t = 1j)}'
    )
    print(
        f'pauliX(0) * (t -> t)(1j): {(spin.x(0) * so1).to_matrix(dims, t = 1j)}'
    )
    print(
        f'pauliX(0) + (t -> t)(1j): {(spin.x(0) + so1).to_matrix(dims, t = 1j)}'
    )
    print(
        f'(t -> t)(1j) + pauliX(0): {(so1 + spin.x(0)).to_matrix(dims, t = 1j)}'
    )
    print(
        f'pauliX(0) + (t -> t)(1j): {(spin.x(0) + so1).to_matrix(dims, t = 1j)}'
    )
    print(
        f'(t -> t)(1j) + pauliX(0): {(so1 + spin.x(0)).to_matrix(dims, t = 1j)}'
    )
    print(
        f'(t -> t)(2.) * (pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(so1 * op7 * op9).to_matrix(dims, t = 2.)}'
    )
    print(
        f'(pauliX(0) + pauliX(1)) * (t -> t)(2.) * (pauliZ(1) + pauliZ(2)): {(op7 * so1 * op9).to_matrix(dims, t = 2.)}'
    )
    print(
        f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)) * (t -> t)(2.): {(op7 * op9 * so1).to_matrix(dims, t = 2.)}'
    )

    op10 = so1 * spin.x(0)
    so1.generator = lambda t: 1. / t
    print(f'(t -> 1/t)(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')
    so1_gen2 = so1.generator
    so1.generator = lambda t: so1_gen2(2 * t)
    print(f'(t -> 1/(2t))(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')
    so1.generator = lambda t: so1_gen2(t)
    print(f'(t -> 1/t)(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')

    so2 = ScalarOperator(lambda t: t**2)
    op11 = spin.z(1) * so2
    print(f'pauliZ(0) * (t -> t^2)(2.): {op11.to_matrix(dims, t = 2.)}')

    so3 = ScalarOperator(lambda t: 1. / t)
    so4 = ScalarOperator(lambda t: t**2)
    print(f'((t -> 1/t) * (t -> t^2))(2.): {(so3 * so4).to_matrix(t = 2.)}')
    so5 = so3 + so4
    so3.generator = lambda field: 1. / field
    print(
        f'((f -> 1/f) + (t -> t^2))(f=2, t=1.): {so5.to_matrix(t = 1., field = 2)}'
    )

    def generator(field, **kwargs):
        print(f'generator got kwargs: {kwargs}')
        return field

    so3.generator = generator
    print(
        f'((f -> f) + (t -> t^2))(f=3, t=2): {so5.to_matrix(field = 3, t = 2, dummy = 10)}'
    )

    so6 = ScalarOperator(lambda foo, *, bar: foo * bar)
    print(
        f'((f,t) -> f*t)(f=3, t=2): {so6.to_matrix(foo = 3, bar = 2, dummy = 10)}'
    )
    so7 = ScalarOperator(lambda foo, *, bar, **kwargs: foo * bar)
    print(
        f'((f,t) -> f*t)(f=3, t=2): {so6.to_matrix(foo = 3, bar = 2, dummy = 10)}'
    )

    def get_parameter_value(parameter_name: str, time: float):
        if parameter_name == "foo":
            return time
        elif parameter_name == "bar":
            return 2 * time
        else:
            raise NotImplementedError(
                f'No value defined for parameter {parameter_name}.')

    schedule = Schedule([0.0, 0.5, 1.0], so6.parameters, get_parameter_value)
    for parameters in schedule:
        print(f'step {schedule.current_step}')
        print(
            f'((f,t) -> f*t)({parameters}): {so6.to_matrix({}, **parameters)}')

    print(
        f'(pauliX(0) + i*pauliY(0))/2: {0.5 * (spin.x(0) + operators.const(1j) * spin.y(0)).to_matrix(dims)}'
    )
    print(f'pauli+(0): {spin.plus(0).to_matrix(dims)}')
    print(
        f'(pauliX(0) - i*pauliY(0))/2: {0.5 * (spin.x(0) - operators.const(1j) * spin.y(0)).to_matrix(dims)}'
    )
    print(f'pauli-(0): {spin.minus(0).to_matrix(dims)}')

    op12 = operators.squeeze(0) + operators.displace(0)
    print(f'create<3>(0): {operators.create(0).to_matrix({0:3})}')
    print(f'annihilate<3>(0): {operators.annihilate(0).to_matrix({0:3})}')
    print(
        f'squeeze<3>(0)[squeezing = 0.5]: {operators.squeeze(0).to_matrix({0:3}, squeezing=0.5)}'
    )
    print(
        f'displace<3>(0)[displacement = 0.5]: {operators.displace(0).to_matrix({0:3}, displacement=0.5)}'
    )
    print(
        f'(squeeze<3>(0) + displace<3>(0))[squeezing = 0.5, displacement = 0.5]: {op12.to_matrix({0:3}, displacement=0.5, squeezing=0.5)}'
    )
    print(
        f'squeeze<4>(0)[squeezing = 0.5]: {operators.squeeze(0).to_matrix({0:4}, squeezing=0.5)}'
    )
    print(
        f'displace<4>(0)[displacement = 0.5]: {operators.displace(0).to_matrix({0:4}, displacement=0.5)}'
    )
    print(
        f'(squeeze<4>(0) + displace<4>(0))[squeezing = 0.5, displacement = 0.5]: {op12.to_matrix({0:4}, displacement=0.5, squeezing=0.5)}'
    )

    so8 = ScalarOperator(lambda my_param: my_param - 1)
    so9 = so7 * so8
    print(f'parameter descriptions: {operators.squeeze(0).parameters}')
    print(f'parameter descriptions: {op12.parameters}')
    print(f'parameter descriptions: {(so7 + so8).parameters}')
    print(
        f'parameter descriptions: {(operators.squeeze(0) * operators.displace(0)).parameters}'
    )
    print(f'parameter descriptions: {so9.parameters}')
    so7.generator = lambda new_parameter: 1.0
    print(f'parameter descriptions: {so9.parameters}')
    so9.generator = lambda reset: reset
    print(f'parameter descriptions: {so9.parameters}')

    def all_zero(sure, args):
        """Some args documentation.
        Args:

        sure (:obj:`int`, optional): my docs for sure
        args: Description of `args`. Multiple
                lines are supported.
        Returns:
        Something that for sure is correct.
        """
        if sure:
            return 0
        else:
            return 1

    print(f'parameter descriptions: {(ScalarOperator(all_zero)).parameters}')

    scop = operators.const(2)
    elop = operators.identity(1)
    print(f"arithmetics: {scop.to_matrix(dims)}")
    print(f"arithmetics: {elop.to_matrix(dims)}")
    print(f"arithmetics: {(elop / scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop * elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop * scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop + elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop + scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop - elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop - scop).to_matrix(dims)}")
    print(f"arithmetics: {((scop * elop) / scop).to_matrix(dims)}")
    print(f"arithmetics: {((elop / scop) * elop).to_matrix(dims)}")
    print(f"arithmetics: {((elop / scop) + elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop * (elop / scop)).to_matrix(dims)}")
    print(f"arithmetics: {(elop + (elop / scop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop + elop) / scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop + (elop / scop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop * elop) / scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop * (elop / scop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop * elop) * scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop * (scop * elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop * elop) * elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop * (scop * elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop * elop) + scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop + (scop * elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop * elop) + elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop + (scop * elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop * elop) - scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop - (scop * elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop * elop) - elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop - (scop * elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop + elop) * scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop * (scop + elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop + elop) * elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop * (scop + elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop - elop) * scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop * (scop - elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop - elop) * elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop * (scop - elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop + elop) + scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop + (scop + elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop + elop) + elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop + (scop + elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop - elop) - scop).to_matrix(dims)}")
    print(f"arithmetics: {(scop - (scop - elop)).to_matrix(dims)}")
    print(f"arithmetics: {((scop - elop) - elop).to_matrix(dims)}")
    print(f"arithmetics: {(elop - (scop - elop)).to_matrix(dims)}")

    opprod = operators.create(0) * operators.annihilate(0)
    opsum = operators.create(0) + operators.annihilate(0)
    for arith in [
            operator.add, operator.sub, operator.mul, operator.truediv,
            operator.pow
    ]:
        print(f"testing {arith} for ScalarOperator")
        print(f"arithmetics: {arith(scop, 2).to_matrix(dims)}")
        print(f"arithmetics: {arith(scop, 2.5).to_matrix(dims)}")
        print(f"arithmetics: {arith(scop, 2j).to_matrix(dims)}")
        print(f"arithmetics: {arith(2, scop).to_matrix(dims)}")
        print(f"arithmetics: {arith(2.5, scop).to_matrix(dims)}")
        print(f"arithmetics: {arith(2j, scop).to_matrix(dims)}")

    for op in [elop, opprod, opsum]:
        for arith in [operator.add, operator.sub, operator.mul]:
            print(f"testing {arith} for {type(op)}")
            print(f"arithmetics: {arith(op, 2).to_matrix(dims)}")
            print(f"arithmetics: {arith(op, 2.5).to_matrix(dims)}")
            print(f"arithmetics: {arith(op, 2j).to_matrix(dims)}")
            print(f"arithmetics: {arith(2, op).to_matrix(dims)}")
            print(f"arithmetics: {arith(2.5, op).to_matrix(dims)}")
            print(f"arithmetics: {arith(2j, op).to_matrix(dims)}")
        print(f"testing {operator.truediv} for {type(op)}")
        print(f"arithmetics: {(op / 2).to_matrix(dims)}")
        print(f"arithmetics: {(op / 2.5).to_matrix(dims)}")
        print(f"arithmetics: {(op / 2j).to_matrix(dims)}")

    print(operators.const(2))
    print(ScalarOperator(lambda alpha: 2 * alpha))
    print(ScalarOperator(all_zero))
    print(spin.x(0))
    print(2 * spin.x(0))
    print(spin.x(0) + 2)
    print(operators.squeeze(0))
    print(operators.squeeze(0) * operators.displace(1))
    print(operators.squeeze(0) + operators.displace(1) * 5)
    print(spin.x(0) - 2)
    print(spin.x(0) - spin.y(1))
    print(spin.x(0).degrees)
    print((spin.x(2) * spin.y(0)).degrees)
    print((spin.x(2) + spin.y(0)).degrees)

    print(ScalarOperator.const(5) == ScalarOperator.const(5))
    print(ScalarOperator.const(5) == ScalarOperator.const(5 + 0j))
    print(ScalarOperator.const(5) == ScalarOperator.const(5j))
    print(ScalarOperator(lambda: 5) == ScalarOperator.const(5))
    print(ScalarOperator(lambda: 5) == ScalarOperator(lambda: 5))
    gen = lambda: 5
    so10 = ScalarOperator(gen)
    so11 = ScalarOperator(lambda: 5)
    print(so10 == so11)
    print(so10 == ScalarOperator(gen))
    so11.generator = gen
    print(so10 == so11)
    print(
        ElementaryOperator.identity(1) *
        so10 == ElementaryOperator.identity(1) * so11)
    print(
        ElementaryOperator.identity(1) +
        so10 == ElementaryOperator.identity(1) + so11)
    print(spin.x(1) + spin.y(1) == spin.y(1) + spin.x(1))
    print(spin.x(1) * spin.y(1) == spin.y(1) * spin.x(1))
    print(spin.x(0) + spin.y(1) == spin.y(1) + spin.x(0))
    print(spin.x(0) * spin.y(1) == spin.y(1) * spin.x(0))
    print(opprod == opprod)
    print(opprod * so10 == so10 * opprod)
    print(opprod + so10 == so10 + opprod)
    print(
        ScalarOperator.const(10) * opprod == opprod *
        ScalarOperator.const(10.0))
    print(
        ScalarOperator.const(10) + opprod == opprod +
        ScalarOperator.const(10.0))
    paulizy = lambda i, j: spin.z(i) * spin.y(j)
    paulixy = lambda i, j: spin.x(i) * spin.y(j)
    print(paulixy(0, 0) + paulizy(0, 0) == paulizy(0, 0) + paulixy(0, 0))
    print(paulixy(0, 0) * paulizy(0, 0) == paulizy(0, 0) * paulixy(0, 0))
    print(
        paulixy(1, 1) * paulizy(0, 0) == paulizy(0, 0) * paulixy(1, 1)
    )  # We have multiple terms acting on the same degree of freedom, so we don't try to reorder here.
    print(paulixy(1, 2) * paulizy(3, 4) == paulizy(3, 4) * paulixy(1, 2))
    print(
        ScalarOperator.const(5) +
        ScalarOperator.const(3) == ScalarOperator.const(4) +
        ScalarOperator.const(4))
    print(
        ScalarOperator.const(6) *
        ScalarOperator.const(2) == ScalarOperator.const(4) *
        ScalarOperator.const(3))
    print((ScalarOperator.const(5) + ScalarOperator.const(3)) * elop == elop *
          (ScalarOperator.const(4) + ScalarOperator.const(4)))
    print(
        ScalarOperator.const(6) * ScalarOperator.const(2) + elop == elop +
        ScalarOperator.const(4) * ScalarOperator.const(3))
    print(
        ScalarOperator.const(6) * ScalarOperator.const(2) * elop == elop *
        ScalarOperator.const(4) * ScalarOperator.const(3))
    print(ScalarOperator.const(5) + 3 == 4 + ScalarOperator.const(4))
    print(ScalarOperator.const(6) * 2 == 4 * ScalarOperator.const(3))
    print((ScalarOperator.const(5) + 3) * elop == elop *
          (4 + ScalarOperator.const(4)))
    print(
        ScalarOperator.const(6) * 2 + elop == elop +
        4 * ScalarOperator.const(3))
    print(
        ScalarOperator.const(6) * 2.0 * elop == elop * 4.0 *
        ScalarOperator.const(3))
    print(ScalarOperator.const(6) / 2 == ScalarOperator.const(3))

    def tranverse_field(num_qubits: int,
                        field_strength: ScalarOperator) -> Operator:
        operator = OperatorSum()
        for i in range(num_qubits):
            operator += field_strength * spin.x(i)
        return operator

    def ising_chain(num_qubits: int,
                    coupling_strength: ScalarOperator) -> Operator:
        operator = OperatorSum()
        for i in range(1, num_qubits):
            operator += coupling_strength * spin.z(i - 1) * spin.z(i)
        return operator

    num_qubits = 5
    start_time, end_time = 0, 1
    field_coeff, coupling_coeff = 1.0, 1.0
    num_steps = 10

    time = ScalarOperator(lambda time: time)
    field_strength = field_coeff * (1 - time)
    coupling_strength = coupling_coeff * time

    hamiltonian = ising_chain(num_qubits, field_strength) + tranverse_field(
        num_qubits, coupling_strength)
    dimensions = dict([(i, 2) for i in hamiltonian.degrees])
    energy = hamiltonian
    magnetization = tranverse_field(num_qubits, operators.const(1)) / num_qubits
    cost_function = ising_chain(num_qubits, operators.const(coupling_coeff))

    uniform_superposition = cudaq.State.from_data(
        numpy.ones(2**num_qubits, dtype=numpy.complex128) /
        numpy.sqrt(2**num_qubits))
    all_zero_state = cudaq.State.from_data(
        numpy.array([1] + (2**num_qubits - 1) * [0], dtype=numpy.complex128))

    steps = numpy.linspace(start_time, end_time, num_steps)
    schedule = Schedule(steps, ["time"])
    #evolution_result = evolve(hamiltonian, dimensions, schedule, uniform_superposition,
    #                          observables = [energy, magnetization, cost_function],
    #                          store_intermediate_results = True)

    print("Evolve on default simulator:")
    schedule.reset()
    evolution_result = evolve(hamiltonian, dimensions, schedule,
                              uniform_superposition)
    evolution_result.final_state().dump()

    print("Evolve asynchronous on default simulator:")
    schedule.reset()
    async_evolution_result = evolve_async(hamiltonian, dimensions, schedule,
                                          uniform_superposition)
    evolution_result = async_evolution_result.get()
    evolution_result.final_state().dump()

    print("Evolve + observe on default simulator:")
    schedule.reset()
    evolution_result = evolve(hamiltonian,
                              dimensions,
                              schedule,
                              uniform_superposition,
                              observables=[cost_function])
    evolution_result.final_state().dump()
    print(
        f"final expectation values: {[res.expectation() for res in evolution_result.final_expectation_values()]}"
    )

    print("Evolve + observe asynchronous on default simulator:")
    schedule.reset()
    async_evolution_result = evolve_async(hamiltonian,
                                          dimensions,
                                          schedule,
                                          uniform_superposition,
                                          observables=[cost_function])
    evolution_result = async_evolution_result.get()
    evolution_result.final_state().dump()
    print(
        f"final expectation values: {[res.expectation() for res in evolution_result.final_expectation_values()]}"
    )

    print("Evolve with intermediate states on default simulator:")
    schedule.reset()
    evolution_result = evolve(hamiltonian,
                              dimensions,
                              schedule,
                              uniform_superposition,
                              store_intermediate_results=True)
    for state in evolution_result.intermediate_states():
        state.dump()
    evolution_result.final_state().dump()

    print("Evolve asynchronous with intermediate states on default simulator:")
    schedule.reset()
    async_evolution_result = evolve_async(hamiltonian,
                                          dimensions,
                                          schedule,
                                          uniform_superposition,
                                          store_intermediate_results=True)
    evolution_result = async_evolution_result.get()
    for state in evolution_result.intermediate_states():
        state.dump()
    evolution_result.final_state().dump()

    print("Evolve + observe with intermediate results on default simulator:")
    schedule.reset()
    evolution_result = evolve(hamiltonian,
                              dimensions,
                              schedule,
                              uniform_superposition,
                              observables=[cost_function],
                              store_intermediate_results=True)
    for state in evolution_result.intermediate_states():
        state.dump()
    for expectations in evolution_result.expectation_values():
        print(
            f"expectation values: {[res.expectation() for res in expectations]}"
        )
    print(
        f"final expectation values: {[res.expectation() for res in evolution_result.final_expectation_values()]}"
    )

    print(
        "Evolve + observe asynchronous with intermediate results on default simulator:"
    )
    schedule.reset()
    async_evolution_result = evolve_async(hamiltonian,
                                          dimensions,
                                          schedule,
                                          uniform_superposition,
                                          observables=[cost_function],
                                          store_intermediate_results=True)
    evolution_result = async_evolution_result.get()
    for state in evolution_result.intermediate_states():
        state.dump()
    for expectations in evolution_result.expectation_values():
        print(
            f"expectation values: {[res.expectation() for res in expectations]}"
        )
    print(
        f"final expectation values: {[res.expectation() for res in evolution_result.final_expectation_values()]}"
    )


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
