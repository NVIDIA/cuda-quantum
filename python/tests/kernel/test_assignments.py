# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
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
def run_and_clear_registries():
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

    # Assigning/embedding a function-argument list as a struct item is now
    # allowed (the item aliases the argument's storage, matching Python
    # semantics). Returning a struct containing a list from an entry-point
    # kernel still hits a separate, pre-existing runtime marshaling
    # limitation (see `get_MyTuple` below).
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel1(l1: list[int]) -> MyTuple:
            return MyTuple(l1, [1, 1])

        cudaq.run(kernel1, [1, 2])
    assert 'Unsupported element type in struct type' in str(e.value)

    @cudaq.kernel
    def get_MyTuple(l1: list[int]) -> MyTuple:
        return MyTuple(l1.copy(), [1, 1])

    with pytest.raises(RuntimeError) as e:
        get_MyTuple([0, 0])
    assert 'Unsupported element type in struct type' in str(e.value)

    # FIXME: this should have a better error message.
    # Error message in main for both this case and the above case is:
    # return values with dynamically sized element types are not yet supported
    with pytest.raises(RuntimeError) as e:
        cudaq.run(get_MyTuple, [0, 0])
    assert 'Unsupported element type in struct type' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def call_modifier(mod: Callable[[list[int]], list[int]],
                          arg: list[int]) -> list[int]:
            return mod(arg)

        print(call_modifier)
    assert ('passing kernels as arguments that return a value is not '
            'currently supported' in str(e.value))

    # Returning a list rooted in a function argument is now allowed: the
    # return value is always copied (matching every other returned list,
    # regardless of provenance), so it is no longer the *same* storage as
    # the original argument once it crosses a `return` boundary - mutating
    # `alias` below does not affect `arg`.
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
            return arg

    @cudaq.kernel
    def call_multiply(arg: list[int]) -> list[int]:
        return modify_and_return(arg)

    @cudaq.kernel
    def kernel2(arg: list[int]) -> tuple[int, int]:
        alias = call_multiply(arg)
        alias[0] = 5
        return sum(alias), sum(arg)

    assert kernel2([0, 1, 2]) == (8, 3)


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
    print("result update1:", str(out[0]))

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
    print("result update2:", str(out[0]))

    @cudaq.kernel
    def update3(arg: MyTuple) -> MyTuple:
        t = arg.copy()
        t.angle += 5.
        return t

    arg = MyTuple(1, 1)
    out = cudaq.run(update3, MyTuple(1, 1), shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(6., 1)
    assert arg == MyTuple(1, 1)
    print("result update3:", str(out[0]))

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

    with pytest.raises(RuntimeError) as e:

        # We do not currently allow any kind of updates to
        # quantum structs.
        @cudaq.kernel
        def test1(t: MyQTuple, controls: cudaq.qview):
            t.controls = controls

        print(test1)
    assert 'accessing attribute of quantum tuple or dataclass does not produce a modifiable value' in str(
        e.value)
    assert '(offending source -> t.controls)' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test2(arg: MyQTuple, controls: cudaq.qview):
            t = arg.copy()
            t.controls = controls

        print(test2)
    assert 'copy is not supported' in str(e.value)
    assert '(offending source -> arg.copy())' in str(e.value)

    @dataclass(slots=True)
    class MyTuple:
        angle: float
        idx: int

    # Mutating a dataclass function argument's field is now allowed. Struct
    # arguments are passed by value at the kernel-to-kernel call boundary
    # (confirmed via the codegen/launcher research backing this fix), so the
    # mutation is confined to the callee's own local copy and never visible
    # to the caller - matches ordinary Python argument-passing semantics.
    @cudaq.kernel
    def update_tuple1(t: MyTuple):
        t.angle = 5.

    @cudaq.kernel
    def test3() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple1(t)
        return t

    assert test3() == MyTuple(0., 0)

    @cudaq.kernel
    def update_tuple2(t: MyTuple):
        t.angle += 5.

    @cudaq.kernel
    def test4() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple2(t)
        return t

    assert test4() == MyTuple(0., 0)

    # Assigning a dataclass function argument to a local variable now
    # aliases it (`t = arg`), but that aliasing is still confined to the
    # callee's own frame - `arg` itself is already a by-value copy of the
    # caller's argument, so mutating through the alias never propagates
    # back to the caller either.
    @cudaq.kernel
    def update_tuple3(arg: MyTuple):
        t = arg
        t.angle = 5.

    @cudaq.kernel
    def test5() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple3(t)
        return t

    assert test5() == MyTuple(0., 0)

    @dataclass(slots=True)
    class NumberedMyTuple:
        val: MyTuple
        num: int

    # Embedding a local dataclass in another dataclass literal is now
    # allowed (the item is stored as a value copy, matching how list items
    # already work) - the compiler-level escape-analysis restriction is
    # gone. Returning a struct containing another struct from an
    # entry-point kernel still hits a separate, pre-existing runtime
    # marshaling limitation, reproducible even with no local variables or
    # aliasing involved at all (a bare literal `NumberedMyTuple(MyTuple(0.5,
    # 1), 0)` hits the same error).
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test6() -> NumberedMyTuple:
            t = MyTuple(0.5, 1)
            return NumberedMyTuple(t, 0)

        test6()
    assert 'Unsupported element type in struct type' in str(e.value)

    # Cross-scope dataclass reassignment now aliases correctly: `t1 = t3`
    # rebinds `t1` to alias `t3`'s storage, so `t3.angle = 5` afterward is
    # visible through `t1` too, while `t2` (aliasing `t1`'s *original*
    # storage from before the reassignment) is unaffected - exactly
    # matching real Python object-identity semantics. `tuple[MyTuple,
    # MyTuple]` as an entry-point return type hits the same pre-existing
    # struct-return marshaling limitation as above, so serialize through a
    # helper kernel to observe the actual field values.
    @cudaq.kernel
    def serialize_pair(t1: MyTuple, t2: MyTuple) -> list[float]:
        return [t1.angle, t1.idx, t2.angle, t2.idx]

    @cudaq.kernel
    def test7(cond: bool) -> list[float]:
        t1 = MyTuple(1, 1)
        t2 = t1
        if cond:
            t3 = MyTuple(2, 2)
            t1 = t3
            t3.angle = 5
        return serialize_pair(t1, t2)

    assert test7(True) == [5.0, 2.0, 1.0, 1.0]

    # A dataclass stored as a list item is a value copy, not an alias (list
    # items must not be references, same rule as before): `t1[0] = t3`
    # copies t3's fields at that point in time, so the later `t3.angle = 5`
    # does not affect `t1[0]`.
    @cudaq.kernel
    def test8(cond: bool) -> list[float]:
        t1 = [MyTuple(1, 1)]
        if cond:
            t3 = MyTuple(2, 2)
            t1[0] = t3
            t3.angle = 5
        return [t1[0].angle, t1[0].idx]

    assert test8(True) == [2.0, 2.0]


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

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def get_list_of_int_tuple(t: tuple[int, int],
                                  size: int) -> list[tuple[int, int]]:
            l = [t for _ in range(size + 1)]
            l[0] = (3, 3)
            return l

        get_list_of_int_tuple((1, 2), 2)
    assert 'Expected a complex, floating, or integral type' in str(e.value)

    with pytest.raises(RuntimeError) as e:

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

        print(test2)
    assert 'tuple value cannot be modified' in str(e.value)

    # Assigning a function-argument tuple (even one containing lists) to a
    # local variable is now allowed. The tuple itself is a value copy (no
    # aliasing - tuples are immutable), but its list-typed fields still
    # alias their underlying array storage where that storage genuinely
    # survives (`t1`, constructed directly from `l1` with no intervening
    # return boundary). Returning a list (including one nested in a tuple)
    # always copies it now, regardless of provenance - so `t2`, obtained
    # via a `return`, is an independent snapshot taken at return time and
    # does not see the later `l1[0] = 2` mutation.
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

    assert test3() == [2, 2, 2, 1, 1]

    # Same reasoning for a list of tuples of lists: `get_item` returns a
    # snapshot of `tlist[0]` taken at return time, unaffected by the later
    # `l1[1] = 3`.
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
        return [t[0][0], t[0][1], t[1][0], t[1][1], l1[0], l1[1]]

    assert test4() == [0, 0, 0, 0, 0, 3]

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test5():
            l = [(0, 1) for _ in range(3)]
            l[0][1] = 2

        test5()
    assert 'tuple value cannot be modified' in str(e.value)
    assert '(offending source -> l[0][1])' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test6():
            l = [(0, [(1, 1)]) for _ in range(3)]
            l[-1][1][0] = (2, 2)
            l[2][1][0][0] = 3

        test6()
    assert 'tuple value cannot be modified' in str(e.value)
    assert '(offending source -> l[2][1][0][0])' in str(e.value)

    @dataclass(slots=True)
    class NumberedTuple:
        idx: int
        vals: tuple[int, list[int]]

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test7():
            t = NumberedTuple(0, (0, [0]))
            t.vals = (1, [1])
            t.vals[1] = [2]

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
        for l1 in ls:
            size += len(l1)
        res = [0 for _ in range(size)]
        idx = 0
        for l2 in ls:
            for i in l2:
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

    # NOTE: This test is flaky on py3.11 + arm64 + cuda12.6.
    # The struct argument marshaling is now correct (see PR #3879), but the
    # kernel-internal deep copy and list mutation still produce wrong results
    # intermittently on that configuration. Disabling until root cause is
    # identified.
    # See https://github.com/NVIDIA/cuda-quantum/issues/3923
    @cudaq.kernel
    def test11(t: MyTuple, size: int) -> list[int]:
        l = [t.copy(deep=True) for _ in range(size)]
        l[0].l1 = [2]
        l[1].l2[0] = 3
        res = [0 for _ in range(4 * len(l))]
        for idx, item in enumerate(l):
            res[4 * idx] = len(item.l1)
            res[4 * idx + 1] = item.l1[0]
            res[4 * idx + 2] = len(item.l2)
            res[4 * idx + 3] = item.l2[0]
        return res

    result = test11(MyTuple([1], [1]), 2)
    assert (result == [1, 2, 1, 1, 1, 1, 1, 3])

    # Embedding a dataclass (even one containing lists, even one rooted in
    # a function argument) as a list/tuple item is now allowed - it's a
    # value copy, exactly like a plain list item already was. These cases
    # now compile successfully; the errors that remain are unrelated,
    # pre-existing runtime marshaling limitations - `list[MyTuple]` isn't
    # currently a marshalable return type at all, independent of anything
    # about aliasing.
    @cudaq.kernel
    def get_MyTuple_list(t: MyTuple) -> list[MyTuple]:
        return [t]

    with pytest.raises(RuntimeError) as e:
        get_MyTuple_list(MyTuple([1], [1]))
    assert 'Expected a complex, floating, or integral type' in str(e.value)

    @cudaq.kernel
    def populate_MyTuple_list(t: MyTuple, size: int) -> list[MyTuple]:
        return [MyTuple(t.l1, t.l2) for _ in range(size)]

    with pytest.raises(RuntimeError) as e:
        populate_MyTuple_list(MyTuple([1], [1]), 2)
    assert 'Expected a complex, floating, or integral type' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def get_MyTuple_list2(size: int) -> list[MyTuple]:
            return [MyTuple([1], [1]) for _ in range(size)]

        print(get_MyTuple_list2(2))
    assert 'Expected a complex, floating, or integral type' in str(e.value)

    @cudaq.kernel
    def populate_MyTuple_list2(t: MyTuple, size: int) -> list[MyTuple]:
        return [t.copy(deep=True) for _ in range(size)]

    # Returning a struct containing a list from an entry-point kernel hits
    # a separate, pre-existing runtime marshaling limitation (see
    # `test_dataclass_update_failures::test6`).
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test2() -> MyTuple:
            l = populate_MyTuple_list2(MyTuple([1, 1], [1, 1]), 2)
            l[0].l1 = [2]
            return l[0]

        test2()
    assert 'Unsupported element type in struct type' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test3() -> list[MyTuple]:
            t1 = MyTuple([1, 1], [1, 1])
            t2 = MyTuple([2, 2], [2, 2])
            l = [t1, t2]
            return l

        test3()
    assert 'Expected a complex, floating, or integral type' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test4() -> list[MyTuple]:
            t = MyTuple([2, 2], [2, 2])
            l = [MyTuple([1, 1], [1, 1]) for _ in range(3)]
            l[0] = t
            return l

        test4()
    assert 'Expected a complex, floating, or integral type' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test5() -> tuple[MyTuple, MyTuple]:
            t1 = MyTuple([1, 1], [1, 1])
            t2 = MyTuple([2, 2], [2, 2])
            return (t1, t2)

        test5()
    assert 'Unsupported element type in struct type' in str(e.value)

    # A dataclass stored as a list item is a value copy, not an alias:
    # `l[0] = t` copies `t`'s fields at that point in time, so later
    # mutations through `t` (whether a whole-field reassignment like
    # `t.l1 = [3]`, or an element mutation) do not affect `l[0]`.
    @cudaq.kernel
    def test6() -> list[int]:
        l = [MyTuple([1], [1])]
        t = MyTuple([2], [2])
        l[0] = t
        t.l1 = [3]
        l[0].l2 = [4]
        return [l[0].l1[0], l[0].l2[0], t.l1[0], t.l2[0]]

    assert test6() == [2, 4, 3, 2]

    # Mutating a list-typed field of a dataclass function argument now
    # works. The struct itself is passed by value (`old`'s own fields are
    # an independent copy), but a list field's span still aliases the same
    # underlying array as the caller's, since only the span descriptor -
    # not the array - gets copied along with the struct.
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

    assert test7(True) == [5, 2, 5, 2, 5, 2]
    assert test7(False) == [5, 1, 5, 1, 5, 1]

    # Returning a function-argument-rooted dataclass is now allowed - like
    # any other returned value containing a list, it's unconditionally
    # copied at the return boundary, so `t`'s list fields are independent
    # snapshots taken at return time (already reflecting the earlier
    # `ls[idx].l1[0] = 2` mutation, made through `ls`, a by-reference list
    # argument, before the return happened) and no longer alias `l1`.
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
        return [t.l1[0], t.l1[1], t.l2[0], t.l2[1], l1[0], l1[1]]

    assert test8() == [2, 3, 2, 0, 2, 0]


def test_list_of_list_updates():

    @cudaq.kernel
    def flatten(ls: list[list[int]]) -> list[int]:
        size = 0
        for l1 in ls:
            size += len(l1)
        res = [0 for _ in range(size)]
        idx = 0
        for l2 in ls:
            for i in l2:
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
        for l1 in ls:
            size += len(l1)
        res = [0 for _ in range(size)]
        idx = 0
        for l2 in ls:
            for i in l2:
                res[idx] = i
                idx += 1
        return res

    # Cross-scope list reassignment now aliases correctly: `l1 = l3` rebinds
    # `l1` to alias `l3`'s storage, so `l3[0] = 5` afterward is visible
    # through `l1` too, while `l2` (aliasing `l1`'s *original* storage from
    # before the reassignment) is unaffected - matches real Python list
    # aliasing semantics.
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

    assert test1(True) == [5, 2, 1, 1, 5, 2]


def test_disallow_value_updates():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test1() -> list[bool]:
            qs = cudaq.qvector(4)
            c = qs[0]
            if True:
                c = qs[1]
            x(c)
            return mz(qs)

        test1()
    assert 'variable defined in parent scope cannot be modified' in str(e.value)
    assert '(offending source -> c = qs[1])' in str(e.value)

    # Reassigning a `measure_handle`-typed variable across scopes is
    # supported now that `mz` returns `cudaq.measure_handle` instead of
    # `bool`: the symbol-table slot has handle type, the inner-scope
    # store binds a fresh handle, and the bool-coercion at `return res`
    # discriminates exactly once. Previously this case was disallowed
    # because measurement results were stored as raw `i1` values in the
    # symbol table.
    @cudaq.kernel
    def test2() -> bool:
        qs = cudaq.qvector(2)
        res = mz(qs[0])
        if True:
            x(qs[1])
            res = mz(qs[1])
        return res

    # `qs[1]` is flipped to |1> and re-measured, then bound to `res`; the
    # bool-coercion at the return discriminates exactly once and yields True.
    assert cudaq.run(test2, shots_count=1)[0] == True


def test_var_scopes():

    @cudaq.kernel
    def test1(cond: bool) -> int:
        if cond:
            val = 3
        else:
            val = 4
        return val

    assert test1(True) == 3
    assert test1(False) == 4

    @cudaq.kernel
    def test2(cond: bool) -> int:
        if cond:
            val = 3
        return val

    assert test2(True) == 3
    # NOTE: test2(False) does not fail but will return an
    # uninitialized value (i.e. garbage).

    @cudaq.kernel
    def test3(val: int) -> int:
        qs = cudaq.qvector(val)
        for idx in range(val):
            x(qs[idx])
        return idx

    assert test3(3) == 2
    assert test3(5) == 4

    @cudaq.kernel
    def test4(cond: bool) -> list[int]:
        if cond:
            ls = [1, 2, 3]
        return ls

    assert test4(True) == [1, 2, 3]
    # NOTE: test4(False) does not fail but will return an
    # uninitialized value (i.e. garbage), just like test2 above.


def test_var_capture():

    @cudaq.kernel
    def test1() -> list[bool]:
        q = cudaq.qvector(3)
        if captured_bool:
            x(q)
        return mz(q)

    # `captured_bool` is not defined yet, so compilation will fail
    with pytest.raises(RuntimeError) as e:
        test1.compile()
    assert "Invalid variable name requested" in str(e.value)

    captured_bool = False

    # now succeeds
    test1.compile()

    out = cudaq.run(test1, shots_count=10)
    assert all(res == [False, False, False] for res in out)

    # Captured variables are evaluated at kernel
    # invocation time.
    captured_bool = True
    out = cudaq.run(test1, shots_count=10)
    assert all(res == [True, True, True] for res in out)

    # The type of a captured variable must not change
    # between kernel definition time and kernel
    # invocation time.
    captured_bool = 1
    with pytest.raises(RuntimeError) as e:
        out = cudaq.run(test1, shots_count=10)
    # TODO: error message could be clearer
    assert "Invalid runtime argument type" in str(e.value)


def test_var_capture_updates():

    n = 3

    @cudaq.kernel
    def kernel1() -> int:
        # Shadow n, no error
        n = 4
        return n

    assert kernel1() == 4

    @cudaq.kernel
    def kernel2a() -> int:
        if True:
            n = 5
        # Returning local variable n
        return n

    assert kernel2a() == 5

    @cudaq.kernel
    def kernel2b(cond: bool) -> int:
        if cond:
            n = 6
        # Returning local variable n
        return n

    assert kernel2b(True) == 6
    # NOTE:
    # kernel2b(False) will still return the local variable,
    # which is uninitialized if the cond is False

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(defer_compilation=False)
        def kernel3() -> int:
            if True:
                # causes the variable to be added to the symbol table
                cudaq.dbg.ast.print_i64(n)
                # Change n, emits an error
                n += 4
            return n

    assert "augment-assign target variable is not defined or cannot be assigned to" in str(
        e.value)
    assert "(offending source -> n += 4)" in str(e.value)

    ls = [1, 2, 3]

    # `ls` (a captured list, treated like any other function argument) is
    # already a copy by the time it's inside the kernel - entry-point
    # arguments are always packed by full byte-copy at the host boundary -
    # so aliasing/mutating it locally, or returning it directly, is now
    # allowed and never visible to the host-side `ls`. `.copy()` remains
    # legal but is no longer required.
    @cudaq.kernel
    def kernel4() -> list[int]:
        vals = ls
        vals[0] = 5
        return vals

    assert kernel4() == [5, 2, 3] and ls == [1, 2, 3]

    @cudaq.kernel
    def kernel5() -> list[int]:
        # `ls` is treated like any other function argument
        ls[0] = 5
        return ls

    assert kernel5() == [5, 2, 3] and ls == [1, 2, 3]

    tp = (1, 5)

    @cudaq.kernel
    def kernel6() -> tuple[int, int]:
        return tp

    assert kernel6() == (1, 5)

    @dataclass(slots=True)
    class MyTuple:
        first: int
        second: int

    mtp = MyTuple(1, 5)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(defer_compilation=False)
        def kernel7():
            mtp.first = 2

    assert "value cannot be modified" in str(e.value)
    assert "(offending source -> mtp.first)" in str(e.value)

    @cudaq.kernel
    def kernel7() -> MyTuple:
        res = mtp.copy()
        res.first = 2
        return res

    assert kernel7() == MyTuple(2, 5) and mtp == MyTuple(1, 5)


def test_inner_functions():

    @cudaq.kernel
    def test1(first: bool, second: bool) -> tuple[bool, bool]:
        i = first
        q = cudaq.qubit()

        def fct():
            i = second
            if i:
                x(q)

        fct()
        return i, mz(q)

    out = cudaq.run(test1, True, False, shots_count=10)
    assert all(res == (True, False) for res in out)
    out = cudaq.run(test1, False, True, shots_count=10)
    assert all(res == (False, True) for res in out)

    @cudaq.kernel
    def test2(cond: bool) -> list[bool]:
        q = cudaq.qvector(2)

        def fct():
            if cond:
                x(q)

        fct()
        return mz(q)

    out = cudaq.run(test2, True, shots_count=10)
    assert all(res == [True, True] for res in out)
    out = cudaq.run(test2, False, shots_count=10)
    assert all(res == [False, False] for res in out)

    captured_bool = False

    @cudaq.kernel
    def test3() -> list[bool]:
        q = cudaq.qvector(3)

        def fct():
            if captured_bool:
                x(q)

        fct()
        return mz(q)

    out = cudaq.run(test3, shots_count=10)
    assert all(res == [False, False, False] for res in out)
    captured_bool = True
    out = cudaq.run(test3, shots_count=10)
    assert all(res == [True, True, True] for res in out)

    @cudaq.kernel
    def test4a():
        q = cudaq.qubit()
        angle = numpy.pi

        def apply_ry():
            ry(angle, q)

        apply_ry()

    out = cudaq.sample(test4a)
    assert len(out) == 1 and '1' in out

    # Python allows this but we don't support it
    # see also test_var_capture.
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(defer_compilation=False)
        def test4b():

            def apply_ry():
                ry(angle, q)

            q = cudaq.qubit()
            angle = numpy.pi
            apply_ry()

    assert "Invalid variable name requested" in str(e.value)
    assert "(offending source -> angle)" in str(e.value)

    @cudaq.kernel
    def test5() -> int:
        ls = [0]

        def fct():
            ls[0] = 5

        fct()
        return ls[0]

    assert test5() == 5

    @cudaq.kernel
    def test6() -> int:
        ls = [0]

        def fct():
            ls[0] += 6

        fct()
        return ls[0]

    assert test6() == 6

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(defer_compilation=False)
        def test7() -> int:
            i = 0

            def fct():
                i += 1

            fct()
            return i

    assert "augment-assign target variable is not defined or cannot be assigned to" in str(
        e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(defer_compilation=False)
        def test8() -> list[int]:
            ls = [0]

            def fct():
                ls += [1]

            fct()
            return ls

    assert "augment-assign target variable is not defined or cannot be assigned to" in str(
        e.value)

    @dataclass(slots=True)
    class BasicTuple:
        first: int
        second: float

    @cudaq.kernel
    def test9(cond: bool) -> BasicTuple:
        t = BasicTuple(1, 0.5)

        def fct():
            t.first = 2

        if cond:
            fct()
        return t

    assert test9(False) == BasicTuple(1, 0.5)
    assert test9(True) == BasicTuple(2, 0.5)

    @cudaq.kernel
    def test10() -> BasicTuple:
        t = BasicTuple(1, 0.5)

        def fct():
            t.second += 2

        fct()
        return t

    assert test10() == BasicTuple(1, 2.5)

    @dataclass(slots=True)
    class ListTuple:
        first: list[int]
        second: list[float]

    @cudaq.kernel
    def test11() -> tuple[int, int]:
        ls = [0]
        t = ListTuple(ls, [0.])

        def fct():
            ls[0] = 4

        fct()
        return ls[0], t.first[0]

    assert test11() == (4, 4)

    @cudaq.kernel
    def test12() -> tuple[float, float]:
        ls = [0.]
        t = ListTuple([0], ls)

        def fct():
            t.second[0] = 4.

        fct()
        return ls[0], t.second[0]

    assert test12() == (4., 4.)


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

    test1a.compile()

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

    test1c.compile()

    @cudaq.kernel
    def test1d(value: BasicTuple) -> BasicTuple:
        local = BasicTuple(1, 5)
        local = value
        return local

    test1d.compile()

    @cudaq.kernel
    def test1e(value: ListTuple) -> ListTuple:
        local = ListTuple([1], [1])
        local = value
        return local

    test1e.compile()

    @cudaq.kernel
    def test2a(value: list[list[int]]) -> list[int]:
        local = [1., 1.]
        local = value[0]
        return local

    test2a.compile()

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

    test2c.compile()

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

    test2e.compile()

    # Assignment to a parent scope

    @cudaq.kernel
    def test1a(cond: bool, value: list[int]) -> list[int]:
        local = [1., 1.]
        if cond:
            local = value
        return local

    test1a.compile()

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

    test1c.compile()

    @cudaq.kernel
    def test1d(cond: bool, value: BasicTuple) -> BasicTuple:
        local = BasicTuple(1, 5)
        if cond:
            local = value
        return local

    test1d.compile()

    @cudaq.kernel
    def test1e(cond: bool, value: ListTuple) -> ListTuple:
        local = ListTuple([1], [1])
        if cond:
            local = value
        return local

    test1e.compile()

    @cudaq.kernel
    def test2a(cond: bool, value: tuple[list[int], list[int]]) -> list[int]:
        local = [1., 1.]
        if cond:
            local = value[0]
        return local

    test2a.compile()

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

    test2c.compile()

    @cudaq.kernel
    def test2d(cond: bool, value: list[BasicTuple]) -> BasicTuple:
        local = BasicTuple(1, 1)
        if cond:
            local = value[0]
        return local

    test2d.compile()

    @cudaq.kernel
    def test2e(cond: bool, value: list[ListTuple]) -> ListTuple:
        local = ListTuple([1], [1])
        if cond:
            local = value[0]
        return local

    test2e.compile()

    # Item assignment to a container in the same scope

    @cudaq.kernel
    def test1a(value: list[int]) -> list[list[int]]:
        local = [[1., 1.]]
        local[0] = value
        return local

    test1a.compile()

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

    test1c.compile()

    @cudaq.kernel
    def test1d(value: BasicTuple) -> list[BasicTuple]:
        local = [BasicTuple(1, 5)]
        local[0] = value
        return local

    test1d.compile()

    @cudaq.kernel
    def test1e(value: ListTuple) -> list[ListTuple]:
        local = [ListTuple([1], [1])]
        local[0] = value
        return local

    test1e.compile()

    @cudaq.kernel
    def test2a(value: list[list[int]]) -> list[list[int]]:
        local = [[1., 1.]]
        local[0] = value[0]
        return local

    test2a.compile()

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

    test2c.compile()

    @cudaq.kernel
    def test2d(value: tuple[BasicTuple, BasicTuple]) -> list[BasicTuple]:
        local = [BasicTuple(1, 1)]
        local[0] = value[0]
        return local

    test2d.compile()

    @cudaq.kernel
    def test2e(value: tuple[ListTuple, ListTuple]) -> list[ListTuple]:
        local = [ListTuple([1], [1])]
        local[0] = value[0]
        return local

    test2e.compile()

    # Item assignment to a container in a parent scope

    @cudaq.kernel
    def test1a(cond: bool, value: list[int]) -> list[list[int]]:
        local = [[1., 1.]]
        if cond:
            local[0] = value
        return local

    test1a.compile()

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

    test1c.compile()

    @cudaq.kernel
    def test1d(cond: bool, value: BasicTuple) -> list[BasicTuple]:
        local = [BasicTuple(1, 5)]
        if cond:
            local[0] = value
        return local

    test1d.compile()

    @cudaq.kernel
    def test1e(cond: bool, value: ListTuple) -> list[ListTuple]:
        local = [ListTuple([1], [1])]
        if cond:
            local[0] = value
        return local

    test1e.compile()

    @cudaq.kernel
    def test2a(cond: bool, value: list[list[int]]) -> list[list[int]]:
        local = [[1., 1.]]
        if cond:
            local[0] = value[0]
        return local

    test2a.compile()

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

    test2c.compile()

    @cudaq.kernel
    def test2d(cond: bool, value: tuple[BasicTuple,
                                        BasicTuple]) -> list[BasicTuple]:
        local = [BasicTuple(1, 1)]
        if cond:
            local[0] = value[0]
        return local

    test2d.compile()

    @cudaq.kernel
    def test2e(cond: bool, value: tuple[ListTuple,
                                        ListTuple]) -> list[ListTuple]:
        local = [ListTuple([1], [1])]
        if cond:
            local[0] = value[0]
        return local

    test2e.compile()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
