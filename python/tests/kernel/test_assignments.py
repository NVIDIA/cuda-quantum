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


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


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
    # FIXME: NEED TO ERROR FOR MyTuple(l1, [1,1])
    # SINCE WE COPY THE LIST ON RETURN IN CASE IT WAS STACK ALLOCATED
    #assert len(results) == 1 and results[0] == (5, 5, 2)

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
    # FIXME: SAME HERE
    #assert len(results) == 1 and results[0] == (6, 6, 3)

    # TODO: test list of list (outer new list, inner ref to same)


def test_list_of_tuple_updates():

    # FIXME: MAYBE WE SHOULD MAKE SURE ARRAYS DO NOT STORE REFS
    # AND STRUCTS DO NOT STORE REFS; I.E. FORBID NESTED STRUCT (ALSO) WHEN CREATING THEM INSIDE KERNEL

    @cudaq.kernel
    def fill_back(l: list[tuple[int, int]], t: tuple[int, int], n: int):
        for idx in range(len(l) - n, len(l)):
            l[idx] = t

    @cudaq.kernel
    def test10() -> list[int]:
        l = [(1, 1) for _ in range(3)]
        # FIXME: this necessarily creates a copy of l due to the args conversion...
        # -> FAIL THE CONVERSION IF LIST CONTAINS STRUCTS?
        # -> STRUCTS THEMSELVES SHOULD BE HANDLED OK SINCE WE REQUIRE COPY WHEN YOU UPDATE
        # -> make structs be loaded by default and only return pointer for assignment
        # -> list comp type must be updated; 
        #    how does list comp of variables vs list comp of const work?
        #    allow list of ptr or not??? 
        #    list element is loaded when iter over list?? 
        #    if so, should be able to create new list by copy
        #    may be confusing too, however, to get an implicit copy on list comp...
        # -> maybe the "easy" solution is to not allow lists of structs on kernel signatures...
        # SOLUTION:
        # -> or don't allow to update lists passed as arguments if they contain tuples (list[tuple], list[list[tuple]] etc)!
        # -> (maybe covered automatically if we just fix the todos in the assign subscript... but error would be cryptic...)
        # -> in that case, return doesn't need to worry about avoiding the heap copy for structs in lists
        # -> but we very much still need to worry about the copy for lists in structs
        # FIXME: WE ALSO NEED TO PREVENT ASSIGNING A LIST WE GOT AS ARG TO A STRUCT, IN CASE IT IS RETURNED...
        #fill_back(l, (2, 2), 2)
        #res = [0 for _ in range(6)]
        #for i in range(3):
        #    res[2 * i] = l[i][0]
        #    res[2 * i + 1] = l[i][1]
        #return res
        return [1]
    
    #print(test10)
    #print(test10())

    @cudaq.kernel
    def get_list_of_int_tuple(t: tuple[int, int], size: int) -> list[tuple[int, int]]:
        #t = arg.copy() # FIXME Make separate test
        l = [t for _ in range(size + 1)]
        l[0] = (3, 3)
        return l
    #print(get_list_of_int_tuple)
    # FIXME: FAILS WITH 
    # Expected a complex, floating, or integral type
    #assert get_list_of_int_tuple((1, 2), 2) == [(3, 3), (1, 2), (1, 2)]

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

    print(test11())
    assert test11() == [3, 3, 4, 4, 1, 2]

    @cudaq.kernel
    def test2() -> list[int]:
        t = (1, 2)
        l = get_list_of_int_tuple(t, 2)
        # Error: indexing into tuple or dataclass does not produce a modifiable value
        #l[1][0] = 4
        res = [0 for _ in range(6)]
        for idx in range(3):
            res[2 * idx] = l[idx][0]
            res[2 * idx + 1] = l[idx][1]
        return res

    print(test2())
    #assert 'Unsupported element type in struct type' in str(e.value)

    # FIXME: LIST OF DATA CLASSES DON'T STORE THEM AS REFERENCES...
    '''
    @cudaq.kernel
    def get_MyTuple_list(t : MyTuple) -> list[MyTuple]:
        return [t]

    # FIXME: segfaults
    # Something should complain about empty lists.
    # Better, we could  properly handle them when we know
    # the type; we can support empty lists if/when there
    # are type annotations for it.
    # get_MyTuple_list(MyTuple([], []))
    # FIXME: segfaults as well....
    #print(get_MyTuple_list(MyTuple([1], [1])))
    print(get_MyTuple_list)
    '''


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
    def kernel1() -> MyTuple:
        t = MyTuple(0., 0)
        update_tuple1(t)
        return t
    
    out = cudaq.run(kernel1, shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(0., 0)
    print("result kernel1: " + str(out[0]))

    @cudaq.kernel
    def update_tuple2(arg : MyTuple) -> MyTuple:
        t = arg.copy()
        t.angle = 5.
        return t

    @cudaq.kernel
    def kernel2() -> MyTuple:
        t = MyTuple(0., 0)
        return update_tuple2(t)
    
    out = cudaq.run(kernel2, shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(5., 0)
    print("result kernel2: " + str(out[0]))

    @cudaq.kernel
    def kernel3(arg : MyTuple) -> MyTuple:
        t = arg.copy()
        t.angle += 5.
        return t

    out = cudaq.run(kernel3, MyTuple(1, 1), shots_count=1)
    assert len(out) == 1 and out[0] == MyTuple(6., 1)
    print("result kernel3: " + str(out[0]))


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
    assert 'accessing attribute of quantum tuple or dataclass does not produce a modifiable value' in str(e.value)
    assert '(offending source -> t.controls)' in str(e.value)

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
    assert 'value cannot be modified - use `.copy()` to create a new value that can be modified' in str(e.value)
    assert '(offending source -> t.angle)' in str(e.value)


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
    assert "CUDA-Q does not allow assignments to variables captured from parent scope" in str(e.value)
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
    assert "CUDA-Q does not allow assignments to variables captured from parent scope" in str(e.value)
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
    assert "Invalid type for variable (tp) captured from parent scope" in str(e.value)
    assert "(offending source -> tp)" in str(e.value)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
