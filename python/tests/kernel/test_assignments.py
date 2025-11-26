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
from typing import Callable


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_list_update():

    @cudaq.kernel
    def sum(l: list[int]) -> int:
        total = 0
        for item in l:
            total += item
        return total

    @cudaq.kernel
    def to_integer(ms: list[bool]) -> int:
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
    # TODO: we generally create a copy when passing values
    # from host to kernel (with the exception of State).
    # Changes hence won't currently be reflected in the
    # host code.
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
        return arg.copy()

    @cudaq.kernel
    def test5(arg: list[int]) -> tuple[int, int]:
        alias = modify_and_return(arg)
        alias[0] = 5
        return sum(alias), sum(arg)

    results = cudaq.run(test5, [0, 1, 2], shots_count=1)
    assert len(results) == 1 and results[0] == (10, 5)

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
    assert len(results) == 1 and results[0] == (10, 5)

    @dataclass(slots=True)
    class MyTuple:
        l1: list[int]
        l2: list[int]

    @cudaq.kernel
    def get_MyTuple(arg: list[int]) -> MyTuple:
        return MyTuple(arg.copy(), [1, 1])

    @cudaq.kernel
    def test7() -> tuple[int, int, int]:
        arg = [2, 2]
        t = get_MyTuple(arg)
        arg[0] = 3
        return sum(arg), sum(t.l1), sum(t.l2)

    results = cudaq.run(test7, shots_count=1)
    assert len(results) == 1 and results[0] == (5, 4, 2)

    @cudaq.kernel
    def test8() -> tuple[int, int, int]:
        arg = [2, 2]
        t = get_MyTuple(arg)
        t.l1[0] = 4
        t.l2[1] = 2
        return sum(arg), sum(t.l1), sum(t.l2)

    results = cudaq.run(test8, shots_count=1)
    assert len(results) == 1 and results[0] == (4, 6, 3)

    @cudaq.kernel
    def create_list_list_int(val: int, size: tuple[int,
                                                   int]) -> list[list[int]]:
        inner_list = [val for _ in range(size[1])]
        return [inner_list.copy() for _ in range(size[0])]

    @cudaq.kernel
    def test9() -> int:
        ls = create_list_list_int(1, (3, 4))
        tot = 0
        ls[1] = [5]
        ls[2][3] = 2
        inner = ls[2]
        inner[1] = 2
        for l in ls:
            tot += sum(l)
        return tot

    assert test9() == 15


def test_list_update_failures():

    @dataclass(slots=True)
    class MyTuple:
        l1: list[int]
        l2: list[int]

    @cudaq.kernel
    def kernel1(l1: list[int]) -> MyTuple:
        return MyTuple(l1, [1, 1])

    with pytest.raises(RuntimeError) as e:
        cudaq.run(kernel1, [1, 2])
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)
    assert '(offending source -> MyTuple(l1, [1, 1]))' in str(e.value)

    @cudaq.kernel
    def get_MyTuple(l1: list[int]) -> MyTuple:
        return MyTuple(l1.copy(), [1, 1])

    with pytest.raises(RuntimeError) as e:
        get_MyTuple([0, 0])
    assert 'return values with dynamically sized element types are not yet supported' in str(
        e.value)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(get_MyTuple, [0, 0])
    assert 'return values with dynamically sized element types are not yet supported' in str(
        e.value)

    @cudaq.kernel
    def sum(l: list[int]) -> int:
        total = 0
        for item in l:
            total += item
        return total

    @cudaq.kernel
    def modify_and_return(arg: list[int]) -> list[int]:
        for i, v in enumerate(arg):
            arg[i] = v * v
        # If we allowed this, then the correct output of
        # kernel2 below would be 10, 10
        return arg

    @cudaq.kernel
    def call_modifier(mod: Callable[[list[int]], list[int]],
                      arg: list[int]) -> list[int]:
        return mod(arg)

    with pytest.raises(RuntimeError) as e:
        print(call_modifier)
    assert 'passing kernels as arguments that return a value is not currently supported' in str(
        e.value)

    @cudaq.kernel
    def call_multiply(arg: list[int]) -> list[int]:
        return modify_and_return(arg)

    @cudaq.kernel
    def kernel2(arg: list[int]) -> tuple[int, int]:
        alias = call_multiply(arg)
        alias[0] = 5
        return sum(alias), sum(arg)

    with pytest.raises(RuntimeError) as e:
        kernel2([0, 1, 2])
    assert 'return value must not contain a list that is a function argument or an item in a function argument' in str(
        e.value)
    assert '(offending source -> return arg)' in str(e.value)


def test_dataclass_update():

    @dataclass(slots=True)
    class MyTuple:
        angle: float
        idx: int

    @cudaq.kernel
    def update_tuple1(arg: MyTuple) -> MyTuple:
        t = arg.copy()
        t.angle = 5.
        return arg

    @cudaq.kernel
    def update1() -> MyTuple:
        t = MyTuple(0., 0)
        return update_tuple1(t)

    out = cudaq.run(update1, shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(0., 0)
    print("result update1: " + str(out[0]))

    @cudaq.kernel
    def update_tuple2(arg: MyTuple) -> MyTuple:
        t = arg.copy()
        t.angle = 5.
        return t

    @cudaq.kernel
    def update2() -> MyTuple:
        return update_tuple2(MyTuple(0., 0))

    out = cudaq.run(update2, shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(5., 0)
    print("result update2: " + str(out[0]))

    @cudaq.kernel
    def update3(arg: MyTuple) -> MyTuple:
        t = arg.copy()
        t.angle += 5.
        return t

    arg = MyTuple(1, 1)
    out = cudaq.run(update3, MyTuple(1, 1), shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(6., 1)
    assert arg == MyTuple(1, 1)
    print("result update3: " + str(out[0]))

    @cudaq.kernel
    def serialize(t1: MyTuple, t2: MyTuple, t3: MyTuple) -> list[float]:
        return [t1.angle, t1.idx, t2.angle, t2.idx, t3.angle, t3.idx]

    @cudaq.kernel
    def update4() -> list[float]:
        t1 = MyTuple(1, 1)
        t2 = t1
        t3 = MyTuple(2, 2)
        t1 = t3
        t3.angle = 5
        return serialize(t1, t2, t3)

    assert update4() == [5.0, 2.0, 1.0, 1.0, 5.0, 2.0]

    @cudaq.kernel
    def update5(cond: bool) -> list[float]:
        t1 = MyTuple(1, 1)
        t2 = t1
        if cond:
            t1.angle = 5
        return [t1.angle, t1.idx, t2.angle, t2.idx]

    assert update5(True) == [5.0, 1.0, 5.0, 1.0]
    assert update5(False) == [1.0, 1.0, 1.0, 1.0]


def test_dataclass_update_failures():

    @dataclass(slots=True)
    class MyQTuple:
        controls: cudaq.qview
        target: cudaq.qubit

    # We do not currently allow any kind of updates to
    # quantum structs.
    @cudaq.kernel
    def test1(t: MyQTuple, controls: cudaq.qview):
        t.controls = controls

    with pytest.raises(RuntimeError) as e:
        print(test1)
    assert 'accessing attribute of quantum tuple or dataclass does not produce a modifiable value' in str(
        e.value)
    assert '(offending source -> t.controls)' in str(e.value)

    @cudaq.kernel
    def test2(arg: MyQTuple, controls: cudaq.qview):
        t = arg.copy()
        t.controls = controls

    with pytest.raises(RuntimeError) as e:
        print(test2)
    assert 'copy is not supported' in str(e.value)
    assert '(offending source -> arg.copy())' in str(e.value)

    @dataclass(slots=True)
    class MyTuple:
        angle: float
        idx: int

    @cudaq.kernel
    def update_tuple1(t: MyTuple):
        t.angle = 5.

    @cudaq.kernel
    def test3() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple1(t)
        return t

    with pytest.raises(RuntimeError) as e:
        print(test3)
    assert 'value cannot be modified - use `.copy(deep)` to create a new value that can be modified' in str(
        e.value)
    assert '(offending source -> t.angle)' in str(e.value)

    @cudaq.kernel
    def update_tuple2(t: MyTuple):
        t.angle += 5.

    @cudaq.kernel
    def test4() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple2(t)
        return t

    with pytest.raises(RuntimeError) as e:
        print(test4)
    assert 'value cannot be modified - use `.copy(deep)` to create a new value that can be modified' in str(
        e.value)
    assert '(offending source -> t.angle)' in str(e.value)

    @cudaq.kernel
    def update_tuple3(arg: MyTuple):
        t = arg
        t.angle = 5.

    @cudaq.kernel
    def test5() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple3(t)
        return t

    with pytest.raises(RuntimeError) as e:
        print(test5())
    assert 'cannot assign dataclass passed as function argument to a local variable' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new value that can be assigned' in str(
        e.value)
    assert '(offending source -> t = arg)' in str(e.value)

    @dataclass(slots=True)
    class NumberedMyTuple:
        val: MyTuple
        num: int

    @cudaq.kernel
    def test6() -> NumberedMyTuple:
        t = MyTuple(0.5, 1)
        return NumberedMyTuple(t, 0)

    with pytest.raises(RuntimeError) as e:
        test6()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new MyTuple' in str(e.value)

    @cudaq.kernel
    def test7(cond: bool) -> tuple[MyTuple, MyTuple]:
        t1 = MyTuple(1, 1)
        t2 = t1
        if cond:
            t3 = MyTuple(2, 2)
            t1 = t3
            t3.angle = 5
        return (t1, t2)

    with pytest.raises(RuntimeError) as e:
        test7(True)
    assert 'only literals can be assigned to variables defined in parent scope' in str(
        e.value)
    assert '(offending source -> t1 = t3)' in str(e.value)

    @cudaq.kernel
    def test8(cond: bool) -> MyTuple:
        t1 = [MyTuple(1, 1)]
        if cond:
            t3 = MyTuple(2, 2)
            t1[0] = t3
            t3.angle = 5
        return t1

    with pytest.raises(RuntimeError) as e:
        test8(True)
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new MyTuple' in str(e.value)
    assert '(offending source -> t1[0] = t3)' in str(e.value)


def test_list_of_tuple_updates():

    @cudaq.kernel
    def fill_back(l: list[tuple[int, int]], t: tuple[int, int], n: int):
        for idx in range(len(l) - n, len(l)):
            l[idx] = t

    @cudaq.kernel
    def test10() -> list[int]:
        l = [(1, 1) for _ in range(3)]
        fill_back(l, (2, 2), 2)
        res = [0 for _ in range(6)]
        for i in range(3):
            res[2 * i] = l[i][0]
            res[2 * i + 1] = l[i][1]
        return res

    assert test10() == [1, 1, 2, 2, 2, 2]

    @cudaq.kernel
    def get_list_of_int_tuple(t: tuple[int, int],
                              size: int) -> list[tuple[int, int]]:
        l = [t for _ in range(size + 1)]
        l[0] = (3, 3)
        return l

    @cudaq.kernel
    def test11() -> list[int]:
        t = (1, 2)
        l = get_list_of_int_tuple(t, 2)
        l[1] = (4, 4)
        res = [0 for _ in range(6)]
        for idx in range(3):
            res[2 * idx] = l[idx][0]
            res[2 * idx + 1] = l[idx][1]
        return res

    assert test11() == [3, 3, 4, 4, 1, 2]

    @cudaq.kernel
    def get_list_of_int_tuple2(arg: tuple[int, int],
                               size: int) -> list[tuple[int, int]]:
        t = arg.copy()
        l = [t for _ in range(size + 1)]
        l[0] = (3, 3)
        return l

    @cudaq.kernel
    def test12() -> list[int]:
        t = (1, 2)
        l = get_list_of_int_tuple2(t, 2)
        l[1] = (4, 4)
        res = [0 for _ in range(6)]
        for idx in range(3):
            res[2 * idx] = l[idx][0]
            res[2 * idx + 1] = l[idx][1]
        return res

    assert test12() == [3, 3, 4, 4, 1, 2]

    @cudaq.kernel
    def modify_first_item(ls: list[tuple[list[int], list[int]]], idx: int,
                          val: int):
        ls[0][0][idx] = val

    @cudaq.kernel
    def test13() -> list[int]:
        l1 = [0, 0]
        tlist = [(l1, l1)]
        modify_first_item(tlist, 0, 2)
        l1[1] = 3
        t = tlist[0]
        return [t[0][0], t[0][1], t[1][0], t[1][1], l1[0], l1[1]]

    assert test13() == [2, 3, 2, 3, 2, 3]

    @dataclass(slots=True)
    class NumberedTuple:
        idx: int
        vals: tuple[int, list[int]]

    @cudaq.kernel
    def test7() -> list[int]:
        l = [1]
        t = NumberedTuple(0, (0, [0]))
        t.vals = (1, l)
        t.vals[1][0] = 2
        return [t.idx, t.vals[0], t.vals[1][0], l[0]]

    assert test7() == [0, 1, 2, 2]


def test_list_of_tuple_update_failures():

    @cudaq.kernel
    def get_list_of_int_tuple(t: tuple[int, int],
                              size: int) -> list[tuple[int, int]]:
        l = [t for _ in range(size + 1)]
        l[0] = (3, 3)
        return l

    with pytest.raises(RuntimeError) as e:
        get_list_of_int_tuple((1, 2), 2)
    assert 'Expected a complex, floating, or integral type' in str(e.value)

    @cudaq.kernel
    def test2() -> list[int]:
        t = (1, 2)
        l = get_list_of_int_tuple(t, 2)
        l[1][0] = 4
        res = [0 for _ in range(6)]
        for idx in range(3):
            res[2 * idx] = l[idx][0]
            res[2 * idx + 1] = l[idx][1]
        return res

    with pytest.raises(RuntimeError) as e:
        print(test2)
    assert 'tuple value cannot be modified' in str(e.value)

    @cudaq.kernel
    def assign_and_return_list_tuple(
            value: tuple[list[int], list[int]]) -> tuple[list[int], list[int]]:
        local = ([1], [1])
        local = value
        return local

    @cudaq.kernel
    def test3() -> list[int]:
        l1 = [1]
        t1 = (l1, l1)
        t2 = assign_and_return_list_tuple(t1)
        l1[0] = 2
        return [l1[0], t1[0][0], t1[1][0], t2[0][0], t2[1][0]]

    with pytest.raises(RuntimeError) as e:
        test3()  # should output [2,2,2,2,2]
    assert 'cannot assign tuple or dataclass passed as function argument to a local variable if it contains a list' in str(
        e.value)

    @cudaq.kernel
    def get_item(ls: list[tuple[list[int], list[int]]],
                 idx: int) -> tuple[list[int], list[int]]:
        return ls[idx]

    @cudaq.kernel
    def test4() -> list[int]:
        l1 = [0, 0]
        tlist = [(l1, l1)]
        t = get_item(tlist, 0)
        l1[1] = 3
        # If we allowed the return in modify_and_return_item,
        # the correct output would be [0, 3, 0, 3, 0, 3]
        return [t[0][0], t[0][1], t[1][0], t[1][1], l1[0], l1[1]]

    with pytest.raises(RuntimeError) as e:
        test4()
    assert 'return value must not contain a list that is a function argument or an item in a function argument' in str(
        e.value)
    assert '(offending source -> return ls[idx])' in str(e.value)

    @cudaq.kernel
    def test5():
        l = [(0, 1) for _ in range(3)]
        l[0][1] = 2

    with pytest.raises(RuntimeError) as e:
        test5()
    assert 'tuple value cannot be modified' in str(e.value)
    assert '(offending source -> l[0][1])' in str(e.value)

    @cudaq.kernel
    def test6():
        l = [(0, [(1, 1)]) for _ in range(3)]
        l[-1][1][0] = (2, 2)
        l[2][1][0][0] = 3

    with pytest.raises(RuntimeError) as e:
        test6()
    assert 'tuple value cannot be modified' in str(e.value)
    assert '(offending source -> l[2][1][0][0])' in str(e.value)

    @dataclass(slots=True)
    class NumberedTuple:
        idx: int
        vals: tuple[int, list[int]]

    @cudaq.kernel
    def test7():
        t = NumberedTuple(0, (0, [0]))
        t.vals = (1, [1])
        t.vals[1] = [2]

    with pytest.raises(RuntimeError) as e:
        test7()
    assert 'tuple value cannot be modified' in str(e.value)
    assert '(offending source -> t.vals[1])' in str(e.value)


def test_list_of_dataclass_updates():

    @dataclass(slots=True)
    class MyTuple:
        l1: list[int]
        l2: list[int]

    @cudaq.kernel
    def serialize(tlist: list[MyTuple]) -> list[int]:
        tot_size = 2 * len(tlist)
        for t in tlist:
            tot_size += len(t.l1) + len(t.l2)
        res = [0 for _ in range(tot_size)]
        idx = 0
        for t in tlist:
            res[idx] = len(t.l1)
            idx += 1
            for i, v in enumerate(t.l1):
                res[idx + i] = v
            idx += len(t.l1)
            res[idx] = len(t.l2)
            idx += 1
            for i, v in enumerate(t.l2):
                res[idx + i] = v
            idx += len(t.l2)
        return res

    @cudaq.kernel
    def populate_MyTuple_list(t: MyTuple, size: int) -> list[MyTuple]:
        return [t.copy(deep=True) for _ in range(size)]

    @cudaq.kernel
    def test1() -> list[int]:
        l = populate_MyTuple_list(MyTuple([1], [1]), 2)
        return serialize(l)

    assert test1() == [1, 1, 1, 1, 1, 1, 1, 1]

    @cudaq.kernel
    def test2() -> list[int]:
        l = populate_MyTuple_list(MyTuple([1, 1], [1, 1]), 2)
        l[0].l1 = [2]
        return serialize(l)

    assert test2() == [1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1]

    @cudaq.kernel
    def test3() -> list[int]:
        l = populate_MyTuple_list(MyTuple([1, 1], [1, 1]), 2)
        l[1].l2[0] = 3
        return serialize(l)

    assert test3() == [2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 1]

    @cudaq.kernel
    def flatten(ls: list[list[int]]) -> list[int]:
        size = 0
        for l in ls:
            size += len(l)
        res = [0 for _ in range(size)]
        idx = 0
        for l in ls:
            for i in l:
                res[idx] = i
                idx += 1
        return res

    @cudaq.kernel
    def test4() -> list[int]:
        l1 = [1, 1]
        t = MyTuple(l1, l1)
        l3 = [2, 2]
        t.l1 = l3
        l3[0] = 5
        return flatten([t.l1, t.l2, l1, l3])

    assert test4() == [5, 2, 1, 1, 1, 1, 5, 2]

    @cudaq.kernel
    def test5(cond: bool) -> list[int]:
        l1 = [1, 1]
        t = MyTuple(l1, l1)
        if cond:
            t.l1 = [2, 2]
        t.l1[0] = 5
        return flatten([t.l1, t.l2, l1])

    assert test5(True) == [5, 2, 1, 1, 1, 1]
    assert test5(False) == [5, 1, 5, 1, 5, 1]

    @cudaq.kernel
    def update_list(old: list[int], new: list[int]):
        old = new

    @cudaq.kernel
    def test6(cond: bool) -> list[int]:
        l1 = [1, 1]
        t = MyTuple(l1, l1)
        if cond:
            update_list(t.l1, [2, 2])
        t.l1[0] = 5
        return flatten([t.l1, t.l2, l1])

    assert test6(True) == [5, 1, 5, 1, 5, 1]
    assert test6(False) == [5, 1, 5, 1, 5, 1]

    @cudaq.kernel
    def update_list2(old: list[int], new: list[int]):
        for idx, v in enumerate(new):
            old[idx] = v

    @cudaq.kernel
    def test7(cond: bool) -> list[int]:
        l1 = [1, 1]
        t = MyTuple(l1, l1)
        if cond:
            update_list2(t.l1, [2, 2])
        t.l1[0] = 5
        return flatten([t.l1, t.l2, l1])

    assert test7(True) == [5, 2, 5, 2, 5, 2]
    assert test7(False) == [5, 1, 5, 1, 5, 1]

    @cudaq.kernel
    def modify_MyTuple(ls: list[MyTuple], idx: int, val: list[int]):
        ls[idx].l1 = val.copy()
        ls[idx].l2 = val

    @cudaq.kernel
    def test8() -> list[int]:
        default = [0]
        vals = [1, 1]
        tlist = [MyTuple(default, default)]
        modify_MyTuple(tlist, 0, vals)
        tlist[0].l1[0] = 2
        return flatten([default, vals, tlist[0].l1, tlist[0].l2])

    assert test8() == [0, 1, 1, 2, 1, 1, 1]

    @cudaq.kernel
    def test9() -> list[int]:
        default = [0]
        vals = [1, 1]
        tlist = [MyTuple(default, default)]
        modify_MyTuple(tlist, 0, vals)
        vals[0] = 2
        return flatten([default, vals, tlist[0].l1, tlist[0].l2])

    assert test9() == [0, 2, 1, 1, 1, 2, 1]

    @cudaq.kernel
    def test10() -> list[int]:
        default = [0]
        vals = [1, 1]
        tlist = [MyTuple(default, default)]
        modify_MyTuple(tlist, 0, vals)
        tlist[0].l2[0] = 3
        return flatten([default, vals, tlist[0].l1, tlist[0].l2])

    assert test10() == [0, 3, 1, 1, 1, 3, 1]


def test_list_of_dataclass_update_failures():

    @dataclass(slots=True)
    class MyTuple:
        l1: list[int]
        l2: list[int]

    @cudaq.kernel
    def get_MyTuple_list(t: MyTuple) -> list[MyTuple]:
        return [t]

    with pytest.raises(RuntimeError) as e:
        print(get_MyTuple_list)
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new MyTuple' in str(e.value)

    @cudaq.kernel
    def populate_MyTuple_list(t: MyTuple, size: int) -> list[MyTuple]:
        # If we allowed this, then the following scenario would lead to
        # incorrect behavior due to the copy of inner lists during return:
        # Caller allocates l1, creates MyTuple using l1 as its first item,
        # calls `populate_MyTuple_list`, modifies an item in l1.
        # In this case, the correct behavior would be that the change to l1
        # is reflected in the list returned by `populate_MyTuple_list`.
        return [MyTuple(t.l1, t.l2) for _ in range(size)]

    with pytest.raises(RuntimeError) as e:
        print(populate_MyTuple_list)
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new list' in str(e.value)

    @cudaq.kernel
    def get_MyTuple_list(size: int) -> list[MyTuple]:
        return [MyTuple([1], [1]) for _ in range(size)]

    with pytest.raises(RuntimeError) as e:
        print(get_MyTuple_list(2))
    assert 'Expected a complex, floating, or integral type' in str(e.value)

    @cudaq.kernel
    def test1(t: MyTuple, size: int) -> list[int]:
        l = [t.copy(deep=True) for _ in range(size)]
        res = [0 for _ in range(4 * len(l))]
        for idx, item in enumerate(l):
            res[4 * idx] = len(item.l1)
            res[4 * idx + 1] = item.l1[0]
            res[4 * idx + 2] = len(item.l2)
            res[4 * idx + 3] = item.l2[0]
        return res

    # TODO: support.
    # The argument conversion from host to device is not correct currently.
    with pytest.raises(RuntimeError) as e:
        test1(MyTuple([1], [1]), 2)
    assert 'dynamically sized element types for function arguments are not yet supported' in str(
        e.value)

    @cudaq.kernel
    def populate_MyTuple_list2(t: MyTuple, size: int) -> list[MyTuple]:
        return [t.copy(deep=True) for _ in range(size)]

    @cudaq.kernel
    def test2() -> MyTuple:
        l = populate_MyTuple_list2(MyTuple([1, 1], [1, 1]), 2)
        l[0].l1 = [2]
        return l[0]

    # TODO: support.
    with pytest.raises(RuntimeError) as e:
        test2()
    assert 'return values with dynamically sized element types are not yet supported' in str(
        e.value)

    @cudaq.kernel
    def test3() -> list[MyTuple]:
        t1 = MyTuple([1, 1], [1, 1])
        t2 = MyTuple([2, 2], [2, 2])
        l = [t1, t2]
        return l

    with pytest.raises(RuntimeError) as e:
        test3()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new MyTuple' in str(e.value)

    @cudaq.kernel
    def test4() -> list[MyTuple]:
        t = MyTuple([2, 2], [2, 2])
        l = [MyTuple([1, 1], [1, 1]) for _ in range(3)]
        l[0] = t
        return l

    with pytest.raises(RuntimeError) as e:
        test4()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new MyTuple' in str(e.value)

    @cudaq.kernel
    def test5() -> tuple[MyTuple, MyTuple]:
        t1 = MyTuple([1, 1], [1, 1])
        t2 = MyTuple([2, 2], [2, 2])
        return (t1, t2)

    with pytest.raises(RuntimeError) as e:
        test5()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new MyTuple' in str(e.value)

    @cudaq.kernel
    def test6() -> tuple[MyTuple, MyTuple]:
        l = [MyTuple([1], [1])]
        t = MyTuple([2], [2])
        l[0] = t
        t.first = [3]
        l[0].second = 4
        # If we allowed this, then
        # t should be MyTuple(first=3, second=4) and
        # l should be [MyTuple(first=3, second=4)]
        return (l[0], t)

    with pytest.raises(RuntimeError) as e:
        test6()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)
    assert 'use `.copy(deep)` to create a new MyTuple' in str(e.value)

    @cudaq.kernel
    def update_list(old: MyTuple, new: list[int]):
        for idx, v in enumerate(new):
            old.l1[idx] = v

    @cudaq.kernel
    def test7(cond: bool) -> list[int]:
        l1 = [1, 1]
        t = MyTuple(l1, l1)
        if cond:
            update_list(t, [2, 2])
        t.l1[0] = 5
        return [t.l1[0], t.l1[1], t.l2[0], t.l2[1], l1[0], l1[1]]

    with pytest.raises(RuntimeError) as e:
        test7()
    assert 'value cannot be modified - use `.copy(deep)` to create a new value that can be modified' in str(
        e.value)
    assert '(offending source -> old.l1)' in str(e.value)

    @cudaq.kernel
    def modify_and_return_item(ls: list[MyTuple], idx: int) -> MyTuple:
        ls[idx].l1[0] = 2
        return ls[idx]

    @cudaq.kernel
    def test8() -> list[int]:
        l1 = [0, 0]
        tlist = [MyTuple(l1, l1)]
        t = modify_and_return_item(tlist, 0)
        t.l1[1] = 3
        # If we allowed the return in modify_and_return_item,
        # the correct output would be [2, 3, 2, 3, 2, 3]
        return [t.l1[0], t.l1[1], t.l2[0], t.l2[1], l1[0], l1[1]]

    with pytest.raises(RuntimeError) as e:
        test8()
    assert 'return value must not contain a list that is a function argument or an item in a function argument' in str(
        e.value)
    assert '(offending source -> return ls[idx])' in str(e.value)


def test_list_of_list_updates():

    @cudaq.kernel
    def flatten(ls: list[list[int]]) -> list[int]:
        size = 0
        for l in ls:
            size += len(l)
        res = [0 for _ in range(size)]
        idx = 0
        for l in ls:
            for i in l:
                res[idx] = i
                idx += 1
        return res

    @cudaq.kernel
    def test1() -> list[int]:
        l1 = [1, 1]
        l2 = l1
        l3 = [2, 2]
        l1 = l3
        l3[0] = 5
        return flatten([l1, l2, l3])

    assert test1() == [5, 2, 1, 1, 5, 2]

    @cudaq.kernel
    def test2(cond: bool) -> list[int]:
        element = [1, 1]
        ls = [element, element]
        if cond:
            update = [2, 2]
            ls[0] = update
            update[0] = 5
        return flatten([ls[0], ls[1], element])

    assert test2(True) == [5, 2, 1, 1, 1, 1]
    assert test2(False) == [1, 1, 1, 1, 1, 1]

    @cudaq.kernel
    def test3(cond: bool) -> list[int]:
        element = [1, 1]
        ls = [element, element]
        if cond:
            update = [2, 2]
            ls[0] = update
            ls[0][0] = 5
            return flatten([ls[0], ls[1], update])
        return flatten([ls[0], ls[1], element])

    assert test3(True) == [5, 2, 1, 1, 5, 2]
    assert test3(False) == [1, 1, 1, 1, 1, 1]

    @cudaq.kernel
    def test4(cond: bool) -> list[int]:
        element = [1, 1]
        ls = [element, element]
        if cond:
            ls[0][0] = 5
        return flatten([ls[0], ls[1], element])

    assert test4(True) == [5, 1, 5, 1, 5, 1]
    assert test4(False) == [1, 1, 1, 1, 1, 1]

    @cudaq.kernel
    def test5(cond: bool) -> list[int]:
        element = [1, 1]
        ls = [element]
        copy = ls[0]
        if cond:
            ls[0][0] = 5
        return flatten([ls[0], copy, element])

    assert test5(True) == [5, 1, 5, 1, 5, 1]
    assert test5(False) == [1, 1, 1, 1, 1, 1]


def test_list_of_list_update_failures():

    @cudaq.kernel
    def flatten(ls: list[list[int]]) -> list[int]:
        size = 0
        for l in ls:
            size += len(l)
        res = [0 for _ in range(size)]
        idx = 0
        for l in ls:
            for i in l:
                res[idx] = i
                idx += 1
        return res

    @cudaq.kernel
    def test1(cond: bool) -> list[int]:
        l1 = [1, 1]
        l2 = l1
        if cond:
            l3 = [2, 2]
            l1 = l3
            l3[0] = 5
            return flatten([l1, l2, l3])
        return flatten([l1, l2])

    with pytest.raises(RuntimeError) as e:
        test1(True)
    assert 'variable defined in parent scope cannot be modified' in str(e.value)
    assert '(offending source -> l1 = l3)' in str(e.value)


def test_disallow_update_capture():

    n = 3
    ls = [1, 2, 3]

    @cudaq.kernel
    def kernel1() -> int:
        # Shadow n, no error
        n = 4
        return n

    res = kernel1()
    assert res == 4

    @cudaq.kernel
    def kernel2() -> int:
        if True:
            # Shadow n, no error
            n = 4
        # n is not defined in this scope, error
        return n

    with pytest.raises(RuntimeError) as e:
        kernel2()
    assert "'n' is not defined" in repr(e)

    @cudaq.kernel
    def kernel3() -> int:
        if True:
            # causes the variable to be added to the symbol table
            cudaq.dbg.ast.print_i64(n)
            # Change n, emits an error
            n += 4
        return n

    with pytest.raises(RuntimeError) as e:
        kernel3()
    assert "CUDA-Q does not allow assignments to variables captured from parent scope" in str(
        e.value)
    assert "(offending source -> n)" in str(e.value)

    @cudaq.kernel
    def kernel4() -> list[int]:
        vals = ls
        vals[0] = 5
        return vals

    assert kernel4() == [5, 2, 3] and ls == [1, 2, 3]

    @cudaq.kernel
    def kernel5():
        ls[0] = 5

    with pytest.raises(RuntimeError) as e:
        kernel5()
    assert "CUDA-Q does not allow assignments to variables captured from parent scope" in str(
        e.value)
    assert "(offending source -> ls)" in str(e.value)

    tp = (1, 5)

    @cudaq.kernel
    def kernel6() -> tuple[int, int]:
        # Capturing tuples is not currently supported.
        # If support is enabled, add test to check that it
        # cannot be modified inside the kernel.
        return tp

    with pytest.raises(RuntimeError) as e:
        kernel6()
    assert "Invalid type for variable (tp) captured from parent scope" in str(
        e.value)
    assert "(offending source -> tp)" in str(e.value)


def test_disallow_value_updates():

    @cudaq.kernel
    def test1() -> list[bool]:
        qs = cudaq.qvector(4)
        c = qs[0]
        if True:
            c = qs[1]
        x(c)
        return mz(qs)

    with pytest.raises(RuntimeError) as e:
        test1()
    assert 'variable defined in parent scope cannot be modified' in str(e.value)
    assert '(offending source -> c = qs[1])' in str(e.value)

    @cudaq.kernel
    def test2() -> bool:
        qs = cudaq.qvector(2)
        res = mz(qs[0])
        if True:
            x(qs[1])
            res = mz(qs[1])
        return res

    # TODO: The reason we cannot currently support this is
    # because we store measurement results as values in the
    # symbol table. This should be changed and supported when
    # we do the change to properly distinguish measurement
    # types from booleans.
    with pytest.raises(RuntimeError) as e:
        test2()
    assert 'variable defined in parent scope cannot be modified' in str(e.value)
    assert '(offending source -> res = mz(qs[1]))' in str(e.value)


def test_function_arguments():

    @dataclass(slots=True)
    class BasicTuple:
        first: int
        second: float

    @dataclass(slots=True)
    class ListTuple:
        first: list[int]
        second: list[float]

    # Case 1: value is function arg
    # Case 2: value is item in function arg
    # Case a: value is a list
    # Case b: value is a tuple that does not contain a list
    # Case c: value is a tuple that contains a list
    # Case d: value is a dataclass that does not contain a list
    # Case e: value is a dataclass that contains a list

    # Assignment to the same scope

    @cudaq.kernel
    def test1a(value: list[int]) -> list[int]:
        local = [1., 1.]
        local = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1a.compile()
    assert 'return value must not contain a list that is a function argument or an item in a function argument' in str(
        e.value)

    @cudaq.kernel
    def test1b(value: tuple[int, int]) -> list[tuple[int, int]]:
        local = (1., 1.)
        local = value
        return [local]

    test1b.compile()

    @cudaq.kernel
    def test1c(
            value: tuple[list[int], list[int]]) -> tuple[list[int], list[int]]:
        local = ([1], [1])
        local = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1c.compile()
    assert 'cannot assign tuple or dataclass passed as function argument to a local variable if it contains a list' in str(
        e.value)

    @cudaq.kernel
    def test1d(value: BasicTuple) -> BasicTuple:
        local = BasicTuple(1, 5)
        local = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1d.compile()
    assert 'cannot assign dataclass passed as function argument to a local variable' in str(
        e.value)

    @cudaq.kernel
    def test1e(value: ListTuple) -> ListTuple:
        local = ListTuple([1], [1])
        local = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1e.compile()
    assert 'cannot assign dataclass passed as function argument to a local variable' in str(
        e.value)

    @cudaq.kernel
    def test2a(value: list[list[int]]) -> list[int]:
        local = [1., 1.]
        local = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2a.compile()
    assert 'lists passed as or contained in function arguments cannot be assigned to to a local variable' in str(
        e.value)

    @cudaq.kernel
    def test2b(value: list[tuple[int, int]]) -> list[tuple[int, int]]:
        local = (1., 1.)
        local = value[0]
        return [local]

    test2b.compile()

    @cudaq.kernel
    def test2c(
        value: list[tuple[list[int],
                          list[int]]]) -> tuple[list[int], list[int]]:
        local = ([1.], [1.])
        local = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2c.compile()
    assert 'cannot assign tuple or dataclass passed as function argument to a local variable if it contains a list' in str(
        e.value)

    @cudaq.kernel
    def test2d(value: tuple[BasicTuple, BasicTuple]) -> BasicTuple:
        local = BasicTuple(1, 1)
        local = value[0]
        return local

    test2d.compile()

    @cudaq.kernel
    def test2e(value: tuple[ListTuple, ListTuple]) -> ListTuple:
        local = ListTuple([1], [1])
        local = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2e.compile()
    assert 'cannot assign tuple or dataclass passed as function argument to a local variable if it contains a list' in str(
        e.value)

    # Assignment to a parent scope

    @cudaq.kernel
    def test1a(cond: bool, value: list[int]) -> list[int]:
        local = [1., 1.]
        if cond:
            local = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1a.compile()
    assert 'lists passed as or contained in function arguments cannot be assigned to variables in the parent scope' in str(
        e.value)

    @cudaq.kernel
    def test1b(cond: bool, value: tuple[int, int]) -> list[tuple[int, int]]:
        local = (1., 1.)
        if cond:
            local = value
        return [local]

    test1b.compile()

    @cudaq.kernel
    def test1c(
            cond: bool, value: tuple[list[int],
                                     list[int]]) -> tuple[list[int], list[int]]:
        local = ([1], [1])
        if cond:
            local = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1c.compile()
    assert 'cannot assign tuple or dataclass passed as function argument to a local variable if it contains a list' in str(
        e.value)

    @cudaq.kernel
    def test1d(cond: bool, value: BasicTuple) -> BasicTuple:
        local = BasicTuple(1, 5)
        if cond:
            local = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1d.compile()
    assert 'cannot assign dataclass passed as function argument to a local variable' in str(
        e.value)

    @cudaq.kernel
    def test1e(cond: bool, value: ListTuple) -> ListTuple:
        local = ListTuple([1], [1])
        if cond:
            local = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1e.compile()
    assert 'cannot assign dataclass passed as function argument to a local variable' in str(
        e.value)

    @cudaq.kernel
    def test2a(cond: bool, value: tuple[list[int], list[int]]) -> list[int]:
        local = [1., 1.]
        if cond:
            local = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2a.compile()
    assert 'lists passed as or contained in function arguments cannot be assigned to to a local variable' in str(
        e.value)

    @cudaq.kernel
    def test2b(
        cond: bool, value: tuple[tuple[int, int],
                                 tuple[int, int]]) -> list[tuple[int, int]]:
        local = (1., 1.)
        if cond:
            local = value[0]
        return [local]

    test2b.compile()

    @cudaq.kernel
    def test2c(
        cond: bool,
        value: list[tuple[list[int],
                          list[int]]]) -> tuple[list[int], list[int]]:
        local = ([1.], [1.])
        if cond:
            local = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2c.compile()
    assert 'cannot assign tuple or dataclass passed as function argument to a local variable if it contains a list' in str(
        e.value)

    @cudaq.kernel
    def test2d(cond: bool, value: list[BasicTuple]) -> BasicTuple:
        local = BasicTuple(1, 1)
        if cond:
            local = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2d.compile()
    assert 'only literals can be assigned to variables defined in parent scope' in str(
        e.value)

    @cudaq.kernel
    def test2e(cond: bool, value: list[ListTuple]) -> ListTuple:
        local = ListTuple([1], [1])
        if cond:
            local = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2e.compile()
    assert 'cannot assign tuple or dataclass passed as function argument to a local variable if it contains a list' in str(
        e.value)

    # Item assignment to a container in the same scope

    @cudaq.kernel
    def test1a(value: list[int]) -> list[list[int]]:
        local = [[1., 1.]]
        local[0] = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1a.compile()
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test1b(value: tuple[int, int]) -> list[tuple[int, int]]:
        local = [(1., 1.)]
        local[0] = value
        return local

    test1b.compile()

    @cudaq.kernel
    def test1c(
        value: tuple[list[int],
                     list[int]]) -> list[tuple[list[int], list[int]]]:
        local = [([1], [1])]
        local[0] = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1c.compile()
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test1d(value: BasicTuple) -> list[BasicTuple]:
        local = [BasicTuple(1, 5)]
        local[0] = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1d.compile()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test1e(value: ListTuple) -> list[ListTuple]:
        local = [ListTuple([1], [1])]
        local[0] = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1e.compile()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test2a(value: list[list[int]]) -> list[list[int]]:
        local = [[1., 1.]]
        local[0] = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2a.compile()
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test2b(value: list[tuple[int, int]]) -> list[tuple[int, int]]:
        local = [(1., 1.)]
        local[0] = value[0]
        return local

    test2b.compile()

    @cudaq.kernel
    def test2c(
        value: list[tuple[list[int], list[int]]]
    ) -> list[tuple[list[int], list[int]]]:
        local = [([1.], [1.])]
        local[0] = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2c.compile()
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test2d(value: tuple[BasicTuple, BasicTuple]) -> list[BasicTuple]:
        local = [BasicTuple(1, 1)]
        local[0] = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2d.compile()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test2e(value: tuple[ListTuple, ListTuple]) -> list[ListTuple]:
        local = [ListTuple([1], [1])]
        local[0] = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2e.compile()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)

    # Item assignment to a container in a parent scope

    @cudaq.kernel
    def test1a(cond: bool, value: list[int]) -> list[list[int]]:
        local = [[1., 1.]]
        if cond:
            local[0] = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1a.compile()
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test1b(cond: bool, value: tuple[int, int]) -> list[tuple[int, int]]:
        local = [(1., 1.)]
        if cond:
            local[0] = value
        return local

    test1b.compile()

    @cudaq.kernel
    def test1c(
        cond: bool,
        value: tuple[list[int],
                     list[int]]) -> list[tuple[list[int], list[int]]]:
        local = [([1], [1])]
        if cond:
            local[0] = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1c.compile()
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test1d(cond: bool, value: BasicTuple) -> list[BasicTuple]:
        local = [BasicTuple(1, 5)]
        if cond:
            local[0] = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1d.compile()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test1e(cond: bool, value: ListTuple) -> list[ListTuple]:
        local = [ListTuple([1], [1])]
        if cond:
            local[0] = value
        return local

    with pytest.raises(RuntimeError) as e:
        test1e.compile()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test2a(cond: bool, value: list[list[int]]) -> list[list[int]]:
        local = [[1., 1.]]
        if cond:
            local[0] = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2a.compile()
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test2b(cond: bool, value: list[tuple[int,
                                             int]]) -> list[tuple[int, int]]:
        local = [(1., 1.)]
        if cond:
            local[0] = value[0]
        return local

    test2b.compile()

    @cudaq.kernel
    def test2c(
        cond: bool, value: list[tuple[list[int], list[int]]]
    ) -> list[tuple[list[int], list[int]]]:
        local = [([1.], [1.])]
        if cond:
            local[0] = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2c.compile()
    assert 'lists passed as or contained in function arguments cannot be inner items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test2d(cond: bool, value: tuple[BasicTuple,
                                        BasicTuple]) -> list[BasicTuple]:
        local = [BasicTuple(1, 1)]
        if cond:
            local[0] = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2d.compile()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)

    @cudaq.kernel
    def test2e(cond: bool, value: tuple[ListTuple,
                                        ListTuple]) -> list[ListTuple]:
        local = [ListTuple([1], [1])]
        if cond:
            local[0] = value[0]
        return local

    with pytest.raises(RuntimeError) as e:
        test2e.compile()
    assert 'only dataclass literals may be used as items in other container values' in str(
        e.value)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
