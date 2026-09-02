# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math
import sys

import cudaq

cudaq.set_target("quake_fake")

# The fake server recognizes the `syntax_check_` prefix and only validates the
# submitted IR for these kernels. It does not lower or execute them, so this
# file tests frontend syntax coverage rather than runtime results.


@cudaq.kernel
def syntax_check_quantum_control() -> bool:
    q = cudaq.qvector(1)
    total = 0
    for i in range(4):
        if i < 2:
            total = total + 1
            x(q[0])
    if total > 1:
        x(q[0])
    return mz(q[0])


@cudaq.kernel
def syntax_check_real_state(values: list[float]) -> list[int]:
    q = cudaq.qvector(values)
    return mz(q)


@cudaq.kernel
def syntax_check_complex_state() -> list[int]:
    q = cudaq.qvector([0. + 0j, 1. + 0j])
    return mz(q)


@cudaq.kernel
def syntax_check_integer_operators(value: int) -> int:
    result = value * 2
    result = (result << 2) >> 1
    result = (result & 15) | 1
    result = result ^ 3
    result += 5
    result -= 2
    return result


@cudaq.kernel
def syntax_check_float_operators(a: float, b: float) -> float:
    result = a * b
    result += math.pi
    return result / 2


@cudaq.kernel
def syntax_check_float_vector(values: list[float], index: int) -> float:
    return values[index]


@cudaq.kernel
def syntax_check_float_literal(index: int) -> float:
    return [1., 2.][index]


@cudaq.kernel
def syntax_check_bits_to_integer(values: list[bool]) -> int:
    result = 0
    for index, value in enumerate(values):
        result = result | (value << index)
    return result


@cudaq.kernel
def syntax_check_integer_sequence(values: list[int]) -> list[int]:
    for index, value in enumerate(values):
        values[index] = value * value
    return values.copy()


@cudaq.kernel
def syntax_check_bool_sequence(values: list[bool]) -> list[bool]:
    return values.copy()


@cudaq.kernel
def syntax_check_float_sequence(values: list[float]) -> list[float]:
    return values.copy()


@cudaq.kernel
def syntax_check_bool_to_int(value: bool) -> int:
    return value


@cudaq.kernel
def syntax_check_int_to_bool(value: int) -> bool:
    return value


@cudaq.kernel
def syntax_check_bool_to_float(value: bool) -> float:
    return value


@cudaq.kernel
def syntax_check_float_to_bool(value: float) -> bool:
    return value


@cudaq.kernel
def syntax_check_int_to_float(value: int) -> float:
    return value


@cudaq.kernel
def syntax_check_float_to_int(value: float) -> int:
    return value


@cudaq.kernel
def syntax_check_for_search(target: int) -> int:
    found = -1
    for i in range(6):
        if i == target:
            found = i
    return found


@cudaq.kernel
def syntax_check_while_return(target: int) -> int:
    i = 0
    while i < 6:
        if i == target:
            return i
        i += 1
    return -1


@cudaq.kernel
def syntax_check_while_comparisons(value: int) -> int:
    while value >= 10:
        value -= 20
    while value <= -10:
        value += 20
    return value


def check_translation(kernel, *args):
    try:
        cudaq.run(kernel, *args, shots_count=1)
    except RuntimeError as error:
        if str(error) == "Invalid size value":
            return
        raise


def syntax_check():
    check_translation(syntax_check_quantum_control)
    check_translation(syntax_check_real_state, [0., 1.])
    check_translation(syntax_check_complex_state)

    check_translation(syntax_check_integer_operators, 3)
    check_translation(syntax_check_float_operators, 2., 3.)
    check_translation(syntax_check_float_vector, [3.073, 1.719], 1)
    check_translation(syntax_check_float_literal, 0)

    check_translation(syntax_check_bits_to_integer, [True, False, True])
    check_translation(syntax_check_integer_sequence, [1, 2, 3])
    check_translation(syntax_check_bool_sequence, [True, False])
    check_translation(syntax_check_float_sequence, [2.547, 1.32])

    check_translation(syntax_check_bool_to_int, True)
    check_translation(syntax_check_int_to_bool, -1)
    check_translation(syntax_check_bool_to_float, True)
    check_translation(syntax_check_float_to_bool, 1.2)
    check_translation(syntax_check_int_to_float, -2)
    check_translation(syntax_check_float_to_int, -1.2)

    check_translation(syntax_check_for_search, 4)
    check_translation(syntax_check_while_return, 4)
    check_translation(syntax_check_while_comparisons, 25)


try:
    syntax_check()
except Exception as error:
    print(error)
    sys.exit(1)
