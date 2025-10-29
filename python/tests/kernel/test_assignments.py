# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, pytest
import cudaq
from dataclasses import dataclass


def test_list_update():

    @cudaq.kernel
    def sum(l : list[int]) -> int:
        total = 0
        for item in l:
            total += item
        return total
    
    @cudaq.kernel
    def to_integer(ms : list[bool]) -> int:
        res = 0
        for idx, v in enumerate(ms):
            res = res | (v << idx)
        return res

    @cudaq.kernel
    def test1(arg: list[int]) -> tuple[int, int]:
        qs = cudaq.qvector(len(arg) + 1)
        for i in arg:
            i += 1
            x(qs[i])
        return sum(arg), to_integer(mz(qs))

    results = cudaq.run(test1, [0, 1, 2], shots_count=1)
    # to_integer(0111) = 2 + 4 + 8 = 14
    assert len(results) == 1 and results[0] == (3, 14)

    @cudaq.kernel
    def double_entries(arg: list[int]):
        for i, v in enumerate(arg):
            arg[i] = 2 * v

    @cudaq.kernel
    def test2(arg: list[int]) -> int:
        double_entries(arg)
        return sum(arg)

    arg = [4, 5, 6]
    results = cudaq.run(test2, arg, shots_count=1)
    assert len(results) == 1 and results[0] == 30  # 2 * (4 + 5 + 6) = 30
    # NOTE: kernel invocations create a copy of their arguments!
    assert arg == [4, 5, 6]

    @cudaq.kernel
    def test3(arg: list[int]) -> tuple[int, int]:
        alias = arg
        double_entries(alias)
        return sum(alias), sum(arg)

    results = cudaq.run(test3, [0, 1, 2], shots_count=1)
    assert len(results) == 1 and results[0] == (6, 6)

    @cudaq.kernel
    def test4(arg: list[int]) -> tuple[int, int]:
        alias = arg
        double_entries(arg)
        return sum(alias), sum(arg)

    results = cudaq.run(test4, [0, 1, 2], shots_count=1)
    assert len(results) == 1 and results[0] == (6, 6)


    @cudaq.kernel
    def test4(arg: list[int]) -> tuple[int, int]:
        alias = arg
        double_entries(arg)
        return sum(alias), sum(arg)

    results = cudaq.run(test4, [0, 1, 2], shots_count=1)
    assert len(results) == 1 and results[0] == (6, 6)

    @cudaq.kernel
    def modify_and_return(arg: list[int]) -> list[int]:
        for i, v in enumerate(arg):
            arg[i] = v * v
        return arg

    @cudaq.kernel
    def test5(arg: list[int]) -> tuple[int, int]:
        alias = modify_and_return(arg)
        alias[0] = 5
        return sum(alias), sum(arg)

    results = cudaq.run(test5, [0, 1, 2], shots_count=1)
    assert len(results) == 1 and results[0] == (10, 10)

    @cudaq.kernel
    def get_list() -> list[int]:
        return [0, 1, 2]
    assert get_list() == [0, 1, 2]

    @cudaq.kernel
    def test6() -> tuple[int, int]:
        local = get_list()
        alias = modify_and_return(local)
        alias[0] = 5
        return sum(alias), sum(local)

    results = cudaq.run(test6, shots_count=1)
    assert len(results) == 1 and results[0] == (10, 10)

    @dataclass(slots=True)
    class MyTuple:
        l1: list[int]
        l2: list[int]

    @cudaq.kernel
    def get_MyTuple(l1 : list[int]) -> MyTuple:
        return MyTuple(l1, [1,1])
    # Not currently supported
    # assert get_MyTuple([0, 0]) == MyTuple([0,0], [1,1])

    @cudaq.kernel
    def test7() -> tuple[int, int, int]:
        arg = [2, 2]
        t = get_MyTuple(arg).copy()
        arg[0] = 3
        return sum(arg), sum(t.l1), sum(t.l2)

    results = cudaq.run(test7, shots_count=1)
    assert len(results) == 1 and results[0] == (5, 5, 2)

    # FIXME: something is going wrong here - 
    # this should not be needed...
    cudaq.__clearKernelRegistries()

    @cudaq.kernel
    def test8() -> tuple[int, int, int]:
        arg = [2, 2]
        t = get_MyTuple(arg).copy()
        t.l1[0] = 4
        t.l2[1] = 2
        return sum(arg), sum(t.l1), sum(t.l2)

    results = cudaq.run(test8, shots_count=1)
    assert len(results) == 1 and results[0] == (6, 6, 3)

    # TODO: test list of list (outer new list, inner ref to same)

def test_list_update_failures():

    @dataclass(slots=True)
    class MyTuple:
        l1: list[int]
        l2: list[int]

    @cudaq.kernel
    def get_MyTuple(l1 : list[int]) -> MyTuple:
        return MyTuple(l1, [1,1])

    with pytest.raises(RuntimeError) as e:
        get_MyTuple([0, 0])
    assert 'Unsupported element type in struct type' in str(e.value)
    # FIXME: cudaq.run(get_MyTuple) currently results in a cryptic/incorrect
    # error 'Tuple size mismatch in value and label'

    @cudaq.kernel
    def test1() -> tuple[int, int]:
        arg = [2, 2]
        t = get_MyTuple(arg) # see test below for why we don't support this
        arg[0] = 3
        return sum(t.l1), sum(t.l2)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(test1)
    assert 'cannot create reference to dataclass passed to or returned from function' in str(e.value)
    assert 'use `.copy()` to create a new value that can be assigned' in str(e.value)
    assert '(offending source -> t = get_MyTuple(arg))' in str(e.value)


def test_dataclass_update():

    @dataclass(slots=True)
    class MyTuple:
        angle: float
        idx: int

    @cudaq.kernel
    def update_tuple1(arg : MyTuple):
        t = arg.copy()
        t.angle = 5.

    @cudaq.kernel
    def test1() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple1(t)
        return t
    
    out = cudaq.run(test1, shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(0., 0)
    print("result test1: " + str(out[0]))

    @cudaq.kernel
    def update_tuple2(arg : MyTuple) -> MyTuple:
        t = arg.copy()
        t.angle = 5.
        return t

    @cudaq.kernel
    def test2() -> MyTuple:
        t = MyTuple(0., 0)
        return update_tuple2(t)
    
    out = cudaq.run(test2, shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(5., 0)
    print("result test2: " + str(out[0]))


def test_dataclass_update_failures():

    @dataclass(slots=True)
    class MyQTuple:
        controls: cudaq.qview
        target: cudaq.qubit

    # We do not currently allow any kind of updates to
    # quantum structs.
    @cudaq.kernel
    def test1(t : MyQTuple, controls: cudaq.qview):
        t.controls = controls

    with pytest.raises(RuntimeError) as e:
        print(test1)
    assert 'quantum data type cannot be updated' in str(e.value)
    assert '(offending source -> t.controls = controls)' in str(e.value)

    @cudaq.kernel
    def test2(arg : MyQTuple, controls: cudaq.qview):
        t = arg.copy()
        t.controls = controls

    with pytest.raises(RuntimeError) as e:
        print(test2)
    assert 'unsupported function copy' in str(e.value)
    assert '(offending source -> arg.copy())' in str(e.value)

    @dataclass(slots=True)
    class MyTuple:
        angle: float
        idx: int

    @cudaq.kernel
    def update_tuple1(t : MyTuple):
        t.angle = 5.

    @cudaq.kernel
    def test3() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple1(t)
        return t

    with pytest.raises(RuntimeError) as e:
        print(test3)
    assert 'value cannot be modified - use `.copy()` to create a new value that can be modified' in str(e.value)
    assert '(offending source -> t.angle)' in str(e.value)

    @cudaq.kernel
    def update_tuple2(t : MyTuple):
        t.angle += 5.

    @cudaq.kernel
    def test4() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple2(t)
        return t

    with pytest.raises(RuntimeError) as e:
        print(test4)
    assert 'augment-assign target variable is not defined or cannot be assigned to' in str(e.value)
    assert '(offending source -> t.angle += 5.0)' in str(e.value)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
