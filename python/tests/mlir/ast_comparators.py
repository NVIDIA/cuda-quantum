# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_comparison_operators_for_integers():

    @cudaq.kernel
    def test_integer_equal_to(v1: int, v2: int) -> bool:
        return v1 == v2

    print(cudaq.synthesize(test_integer_equal_to, 5, 5))
    print(cudaq.synthesize(test_integer_equal_to, 3, -3))
    # CHECK-LABEL: test_integer_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_integer_not_equal_to(v1: int, v2: int) -> bool:
        return v1 != v2

    print(cudaq.synthesize(test_integer_not_equal_to, 3, 5))
    print(cudaq.synthesize(test_integer_not_equal_to, -5, -5))
    # CHECK-LABEL: test_integer_not_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_integer_less_than(v1: int, v2: int) -> bool:
        return v1 < v2

    print(cudaq.synthesize(test_integer_less_than, 3, 5))
    print(cudaq.synthesize(test_integer_less_than, 5, -3))
    # CHECK-LABEL: test_integer_less_than
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_integer_greater_than(v1: int, v2: int) -> bool:
        return v1 > v2

    print(cudaq.synthesize(test_integer_greater_than, 5, -3))
    print(cudaq.synthesize(test_integer_greater_than, 3, 5))
    # CHECK-LABEL: test_integer_greater_than
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_integer_less_than_or_equal_to(v1: int, v2: int) -> bool:
        return v1 <= v2

    print(cudaq.synthesize(test_integer_less_than_or_equal_to, -5, -3))
    print(cudaq.synthesize(test_integer_less_than_or_equal_to, 5, 3))
    print(cudaq.synthesize(test_integer_less_than_or_equal_to, 3, 3))
    # CHECK-LABEL: test_integer_less_than_or_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true

    @cudaq.kernel
    def test_integer_greater_than_or_equal_to(v1: int, v2: int) -> bool:
        return v1 >= v2

    print(cudaq.synthesize(test_integer_greater_than_or_equal_to, 5, 3))
    print(cudaq.synthesize(test_integer_greater_than_or_equal_to, -5, -3))
    print(cudaq.synthesize(test_integer_greater_than_or_equal_to, 3, 3))
    # CHECK-LABEL: test_integer_greater_than_or_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true


def test_comparison_operators_for_floats():

    @cudaq.kernel
    def test_float_equal_to(v1: float, v2: float) -> bool:
        return v1 == v2

    print(cudaq.synthesize(test_float_equal_to, 5.2, 5.2))
    print(cudaq.synthesize(test_float_equal_to, 5.1, -5.1))
    # CHECK-LABEL: test_float_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_float_not_equal_to(v1: float, v2: float) -> bool:
        return v1 != v2

    print(cudaq.synthesize(test_float_not_equal_to, 5.1, 5.2))
    print(cudaq.synthesize(test_float_not_equal_to, -5.2, -5.2))
    # CHECK-LABEL: test_float_not_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_float_less_than(v1: float, v2: float) -> bool:
        return v1 < v2

    print(cudaq.synthesize(test_float_less_than, -5.2, -5.1))
    print(cudaq.synthesize(test_float_less_than, 5.2, 5.1))
    # CHECK-LABEL: test_float_less_than
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_float_greater_than(v1: float, v2: float) -> bool:
        return v1 > v2

    print(cudaq.synthesize(test_float_greater_than, 5.2, 5.1))
    print(cudaq.synthesize(test_float_greater_than, -5.8, 5.2))
    # CHECK-LABEL: test_float_greater_than
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_float_less_than_or_equal_to(v1: float, v2: float) -> bool:
        return v1 <= v2

    print(cudaq.synthesize(test_float_less_than_or_equal_to, -5.1, 5.0))
    print(cudaq.synthesize(test_float_less_than_or_equal_to, 5.2, 5.1))
    print(cudaq.synthesize(test_float_less_than_or_equal_to, 5.1, 5.1))
    # CHECK-LABEL: test_float_less_than_or_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true

    @cudaq.kernel
    def test_float_greater_than_or_equal_to(v1: float, v2: float) -> bool:
        return v1 >= v2

    print(cudaq.synthesize(test_float_greater_than_or_equal_to, 5.0, -5.1))
    print(cudaq.synthesize(test_float_greater_than_or_equal_to, 5.1, 5.2))
    print(cudaq.synthesize(test_float_greater_than_or_equal_to, 5.1, 5.1))
    # CHECK-LABEL: test_float_greater_than_or_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true


def test_comparison_operators_for_complex():

    @cudaq.kernel
    def test_complex_equal_to(v1: complex, v2: complex) -> bool:
        return v1 == v2

    print(
        cudaq.synthesize(test_complex_equal_to, complex(1., 0.5),
                         complex(1., 0.5)))
    print(
        cudaq.synthesize(test_complex_equal_to, complex(1., 2.),
                         complex(1., 2.5)))
    print(
        cudaq.synthesize(test_complex_equal_to, complex(1., 0.5),
                         complex(4., -0.5)))
    print(
        cudaq.synthesize(test_complex_equal_to, complex(-1., 0.5),
                         complex(1., 0.5)))
    # CHECK-LABEL: test_complex_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %false = arith.constant false
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_complex_not_equal_to(v1: complex, v2: complex) -> bool:
        return v1 != v2

    print(
        cudaq.synthesize(test_complex_not_equal_to, complex(1., 2.),
                         complex(1., 2.5)))
    print(
        cudaq.synthesize(test_complex_not_equal_to, complex(1., 0.5),
                         complex(1., 0.5)))
    print(
        cudaq.synthesize(test_complex_not_equal_to, complex(1., -0.5),
                         complex(4., 0.5)))
    print(
        cudaq.synthesize(test_complex_not_equal_to, complex(1., 0.5),
                         complex(-1., 0.5)))
    # CHECK-LABEL: test_complex_not_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true
    # CHECK:    %true = arith.constant true


def test_comparison_operators_for_mixed_types():

    @cudaq.kernel
    def test_int_float_equal_to(v1: int, v2: float) -> bool:
        return v1 == v2

    print(cudaq.synthesize(test_int_float_equal_to, 5, 5.0))
    print(cudaq.synthesize(test_int_float_equal_to, 4, 4.5))
    # CHECK-LABEL: test_int_float_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_float_int_equal_to(v1: float, v2: int) -> bool:
        return v1 == v2

    print(cudaq.synthesize(test_float_int_equal_to, 5.0, 5))
    print(cudaq.synthesize(test_float_int_equal_to, 4.5, 4))
    # CHECK-LABEL: test_float_int_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_int_float_not_equal_to(v1: int, v2: float) -> bool:
        return v1 != v2

    print(cudaq.synthesize(test_int_float_not_equal_to, 4, 4.5))
    print(cudaq.synthesize(test_int_float_not_equal_to, 5, 5.0))
    # CHECK-LABEL: test_int_float_not_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_float_int_not_equal_to(v1: float, v2: int) -> bool:
        return v1 != v2

    print(cudaq.synthesize(test_float_int_not_equal_to, 4.5, 4))
    print(cudaq.synthesize(test_float_int_not_equal_to, 5.0, 5))
    # CHECK-LABEL: test_float_int_not_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_int_float_less_than(v1: int, v2: float) -> bool:
        return v1 < v2

    print(cudaq.synthesize(test_int_float_less_than, 5, 5.2))
    print(cudaq.synthesize(test_int_float_less_than, 5, 5.0))
    # CHECK-LABEL: test_int_float_less_than
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_float_int_less_than(v1: float, v2: int) -> bool:
        return v1 < v2

    print(cudaq.synthesize(test_float_int_less_than, 5.8, 6))
    print(cudaq.synthesize(test_float_int_less_than, 5.0, 5))
    # CHECK-LABEL: test_float_int_less_than
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_int_float_greater_than(v1: int, v2: float) -> bool:
        return v1 > v2

    print(cudaq.synthesize(test_int_float_greater_than, 5, 4.8))
    print(cudaq.synthesize(test_int_float_greater_than, 5, 5.0))
    # CHECK-LABEL: test_int_float_greater_than
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_float_int_greater_than(v1: float, v2: int) -> bool:
        return v1 > v2

    print(cudaq.synthesize(test_float_int_greater_than, 5.2, 5))
    print(cudaq.synthesize(test_float_int_greater_than, 5.0, 5))
    # CHECK-LABEL: test_float_int_greater_than
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false

    @cudaq.kernel
    def test_int_float_less_than_or_equal_to(v1: int, v2: float) -> bool:
        return v1 <= v2

    print(cudaq.synthesize(test_int_float_less_than_or_equal_to, 5, 5.2))
    print(cudaq.synthesize(test_int_float_less_than_or_equal_to, 5, 4.8))
    print(cudaq.synthesize(test_int_float_less_than_or_equal_to, 5, 5.0))
    # CHECK-LABEL: test_int_float_less_than_or_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true

    @cudaq.kernel
    def test_float_int_less_than_or_equal_to(v1: float, v2: int) -> bool:
        return v1 <= v2

    print(cudaq.synthesize(test_float_int_less_than_or_equal_to, 5.1, 6))
    print(cudaq.synthesize(test_float_int_less_than_or_equal_to, 5.2, 5))
    print(cudaq.synthesize(test_float_int_less_than_or_equal_to, 5.0, 5))
    # CHECK-LABEL: test_float_int_less_than_or_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true

    @cudaq.kernel
    def test_int_float_greater_than_or_equal_to(v1: int, v2: float) -> bool:
        return v1 >= v2

    print(cudaq.synthesize(test_int_float_greater_than_or_equal_to, 5, 4.8))
    print(cudaq.synthesize(test_int_float_greater_than_or_equal_to, 5, 5.1))
    print(cudaq.synthesize(test_int_float_greater_than_or_equal_to, 5, 5.0))
    # CHECK-LABEL: test_int_float_greater_than_or_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true

    @cudaq.kernel
    def test_float_int_greater_than_or_equal_to(v1: float, v2: int) -> bool:
        return v1 >= v2

    print(cudaq.synthesize(test_float_int_greater_than_or_equal_to, 5.1, 5))
    print(cudaq.synthesize(test_float_int_greater_than_or_equal_to, 5.1, 6))
    print(cudaq.synthesize(test_float_int_greater_than_or_equal_to, 5.0, 5))
    # CHECK-LABEL: test_float_int_greater_than_or_equal_to
    # CHECK:    %true = arith.constant true
    # CHECK:    %false = arith.constant false
    # CHECK:    %true = arith.constant true


def test_comparison_in():

    @cudaq.kernel
    def test_integer_in_int_list(v: int) -> bool:
        return v in [1, 2, 3]

    print(test_integer_in_int_list)
    print(cudaq.run(test_integer_in_int_list, 1, shots_count=1))
    print(cudaq.run(test_integer_in_int_list, 5, shots_count=1))
    print(cudaq.run(test_integer_in_int_list, -1, shots_count=1))
    # CHECK-LABEL: test_integer_in_int_list
    # CHECK:    [True]
    # CHECK:    [False]
    # CHECK:    [False]

    @cudaq.kernel
    def test_integer_not_in_int_list(v: int) -> bool:
        return v not in [1, 2, 3]

    print(test_integer_not_in_int_list)
    print(cudaq.run(test_integer_not_in_int_list, 5, shots_count=1))
    print(cudaq.run(test_integer_not_in_int_list, -1, shots_count=1))
    print(cudaq.run(test_integer_not_in_int_list, 1, shots_count=1))
    # CHECK-LABEL: test_integer_not_in_int_list
    # CHECK:    [True]
    # CHECK:    [True]
    # CHECK:    [False]

    @cudaq.kernel
    def test_float_in_float_list(v: float) -> bool:
        return v in [1.5, 2.5, 3.5]

    print(test_float_in_float_list)
    print(cudaq.run(test_float_in_float_list, 1.5, shots_count=1))
    print(cudaq.run(test_float_in_float_list, 2., shots_count=1))
    print(cudaq.run(test_float_in_float_list, -1.5, shots_count=1))
    # CHECK-LABEL: test_float_in_float_list
    # CHECK:    [True]
    # CHECK:    [False]
    # CHECK:    [False]

    @cudaq.kernel
    def test_float_not_in_float_list(v: float) -> bool:
        return v not in [1.5, 2.5, 3.5]

    print(test_float_not_in_float_list)
    print(cudaq.run(test_float_not_in_float_list, -1.5, shots_count=1))
    print(cudaq.run(test_float_not_in_float_list, 2.5, shots_count=1))
    print(cudaq.run(test_float_not_in_float_list, -2.5, shots_count=1))
    # CHECK-LABEL: test_float_not_in_float_list
    # CHECK:    [True]
    # CHECK:    [False]
    # CHECK:    [True]

    @cudaq.kernel
    def test_complex_in_float_list(v: complex) -> bool:
        return v in [complex(-1, 0.5), complex(2, 0.5)]

    print(test_complex_in_float_list)
    print(cudaq.run(test_complex_in_float_list, complex(2, 0.5), shots_count=1))
    print(
        cudaq.run(test_complex_in_float_list, complex(-1., 0.5), shots_count=1))
    print(cudaq.run(test_complex_in_float_list, complex(2, 0.), shots_count=1))
    print(cudaq.run(test_complex_in_float_list, complex(0., 0.5),
                    shots_count=1))
    print(cudaq.run(test_complex_in_float_list, complex(2, -0.5),
                    shots_count=1))
    print(cudaq.run(test_complex_in_float_list, complex(-2, 0.5),
                    shots_count=1))
    # CHECK-LABEL: test_complex_in_float_list
    # CHECK:    [True]
    # CHECK:    [True]
    # CHECK:    [False]
    # CHECK:    [False]
    # CHECK:    [False]
    # CHECK:    [False]

    @cudaq.kernel
    def test_complex_not_in_float_list(v: complex) -> bool:
        return v not in [complex(1, -0.5), complex(2, 0.5)]

    print(test_complex_not_in_float_list)
    print(
        cudaq.run(test_complex_not_in_float_list, complex(2, 0.),
                  shots_count=1))
    print(
        cudaq.run(test_complex_not_in_float_list,
                  complex(0., 0.5),
                  shots_count=1))
    print(
        cudaq.run(test_complex_not_in_float_list,
                  complex(2, 0.5),
                  shots_count=1))
    print(
        cudaq.run(test_complex_not_in_float_list,
                  complex(1, -0.5),
                  shots_count=1))
    print(
        cudaq.run(test_complex_not_in_float_list,
                  complex(2, -0.5),
                  shots_count=1))
    print(
        cudaq.run(test_complex_not_in_float_list,
                  complex(-2, 0.5),
                  shots_count=1))
    # CHECK-LABEL: test_complex_not_in_float_list
    # CHECK:    [True]
    # CHECK:    [True]
    # CHECK:    [False]
    # CHECK:    [False]
    # CHECK:    [True]
    # CHECK:    [True]


def test_comparison_failures():

    # complex
    try:

        @cudaq.kernel
        def test_complex_less_than(v1: complex, v2: complex) -> bool:
            return v1 < v2

        print(
            cudaq.synthesize(test_complex_less_than, complex(1, 0),
                             complex(1, 0)))
    except Exception as e:
        print("Failure test_complex_less_than:")
        print(e)

    # CHECK-LABEL:    Failure test_complex_less_than:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 < v2)

    try:

        @cudaq.kernel
        def test_float_complex_less_than(v1: float, v2: complex) -> bool:
            return v1 < v2

        print(cudaq.synthesize(test_float_complex_less_than, 1., complex(1, 0)))
    except Exception as e:
        print("Failure test_float_complex_less_than:")
        print(e)

    # CHECK-LABEL:    Failure test_float_complex_less_than:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 < v2)

    try:

        @cudaq.kernel
        def test_complex_float_less_than(v1: complex, v2: float) -> bool:
            return v1 < v2

        print(cudaq.synthesize(test_complex_float_less_than, complex(1, 0), 1.))
    except Exception as e:
        print("Failure test_complex_float_less_than:")
        print(e)

    # CHECK-LABEL:    Failure test_complex_float_less_than:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 < v2)

    try:

        @cudaq.kernel
        def test_complex_greater_than(v1: complex, v2: complex) -> bool:
            return v1 > v2

        print(
            cudaq.synthesize(test_complex_greater_than, complex(1, 0),
                             complex(1, 0)))
    except Exception as e:
        print("Failure test_complex_greater_than:")
        print(e)

    # CHECK-LABEL:    Failure test_complex_greater_than:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 > v2)

    try:

        @cudaq.kernel
        def test_float_complex_greater_than(v1: float, v2: complex) -> bool:
            return v1 > v2

        print(
            cudaq.synthesize(test_float_complex_greater_than, 1., complex(1,
                                                                          0)))
    except Exception as e:
        print("Failure test_float_complex_greater_than:")
        print(e)

    # CHECK-LABEL:    Failure test_float_complex_greater_than:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 > v2)

    try:

        @cudaq.kernel
        def test_complex_float_greater_than(v1: complex, v2: float) -> bool:
            return v1 > v2

        print(
            cudaq.synthesize(test_complex_float_greater_than, complex(1, 0),
                             1.))
    except Exception as e:
        print("Failure test_complex_float_greater_than:")
        print(e)

    # CHECK-LABEL:    Failure test_complex_float_greater_than:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 > v2)

    try:

        @cudaq.kernel
        def test_complex_less_than_or_equal_to(v1: complex,
                                               v2: complex) -> bool:
            return v1 <= v2

        print(
            cudaq.synthesize(test_complex_less_than_or_equal_to, complex(1, 0),
                             complex(1, 0)))
    except Exception as e:
        print("Failure test_complex_less_than_or_equal_to:")
        print(e)

    # CHECK-LABEL:    Failure test_complex_less_than_or_equal_to:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 <= v2)

    try:

        @cudaq.kernel
        def test_float_complex_less_than_or_equal_to(v1: float,
                                                     v2: complex) -> bool:
            return v1 <= v2

        print(
            cudaq.synthesize(test_float_complex_less_than_or_equal_to, 1.,
                             complex(1, 0)))
    except Exception as e:
        print("Failure test_float_complex_less_than_or_equal_to:")
        print(e)

    # CHECK-LABEL:    Failure test_float_complex_less_than_or_equal_to:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 <= v2)

    try:

        @cudaq.kernel
        def test_complex_float_less_than_or_equal_to(v1: complex,
                                                     v2: float) -> bool:
            return v1 <= v2

        print(
            cudaq.synthesize(test_complex_float_less_than_or_equal_to,
                             complex(1, 0), 1.))
    except Exception as e:
        print("Failure test_complex_float_less_than_or_equal_to:")
        print(e)

    # CHECK-LABEL:    Failure test_complex_float_less_than_or_equal_to:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 <= v2)

    try:

        @cudaq.kernel
        def test_complex_greater_than_or_equal_to(v1: complex,
                                                  v2: complex) -> bool:
            return v1 >= v2

        print(
            cudaq.synthesize(test_complex_greater_than_or_equal_to,
                             complex(1, 0), complex(1, 0)))
    except Exception as e:
        print("Failure test_complex_greater_than_or_equal_to:")
        print(e)

    # CHECK-LABEL:    Failure test_complex_greater_than_or_equal_to:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 >= v2)

    try:

        @cudaq.kernel
        def test_float_complex_greater_than_or_equal_to(v1: float,
                                                        v2: complex) -> bool:
            return v1 >= v2

        print(
            cudaq.synthesize(test_float_complex_greater_than_or_equal_to, 1.,
                             complex(1, 0)))
    except Exception as e:
        print("Failure test_float_complex_greater_than_or_equal_to:")
        print(e)

    # CHECK-LABEL:    Failure test_float_complex_greater_than_or_equal_to:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 >= v2)

    try:

        @cudaq.kernel
        def test_complex_float_greater_than_or_equal_to(v1: complex,
                                                        v2: float) -> bool:
            return v1 >= v2

        print(
            cudaq.synthesize(test_complex_float_greater_than_or_equal_to,
                             complex(1, 0), 1.))
    except Exception as e:
        print("Failure test_complex_float_greater_than_or_equal_to:")
        print(e)

    # CHECK-LABEL:    Failure test_complex_float_greater_than_or_equal_to:
    # CHECK:          invalid type 'Complex' in comparison
    # CHECK-NEXT:     (offending source -> v1 >= v2)

    # Not yet supported - left as todos...

    try:

        @cudaq.kernel
        def test_list_in_list(v: list[int]) -> bool:
            return v in [[1], [2], [3]]

        print(cudaq.run(test_list_in_list, [1], shots_count=1))
    except Exception as e:
        print("Failure test_list_in_list:")
        print(e)

    # CHECK-LABEL:    Failure test_list_in_list:
    # CHECK:          invalid type in comparison
    # CHECK-NEXT:     (offending source -> v in {{.*}}1], [2], [3{{.*}})

    try:

        @cudaq.kernel
        def test_int_tuple_equal_to(v: tuple[int, int]) -> bool:
            return v == (1, 3)

        print(cudaq.run(test_int_tuple_equal_to, (1, 3), shots_count=1))
    except Exception as e:
        print("Failure test_int_tuple_equal_to:")
        print(e)

    # CHECK-LABEL:    Failure test_int_tuple_equal_to:
    # CHECK:          invalid type in comparison
    # CHECK-NEXT:     (offending source -> v == (1, 3))

    try:

        @cudaq.kernel
        def test_composition() -> bool:
            return 2. == 1.5 | 2. == 2.5

        print(cudaq.run(test_composition, shots_count=1))
    except Exception as e:
        print("Failure test_composition:")
        print(e)

    # CHECK-LABEL:    Failure test_composition:
    # CHECK:          only single comparators are supported
    # CHECK-NEXT:     (offending source -> 2.0 == 1.5 | 2.0 == 2.5)
