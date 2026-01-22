# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest

cudaq_test_cpp_algo = pytest.importorskip('cudaq_test_cpp_algo')


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_mergeExternal():

    @cudaq.kernel
    def kernel(i: int):
        q = cudaq.qvector(i)
        h(q[0])

    kernel.compile()
    kernel(10)

    otherMod = '''module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__test = "__nvqpp__mlirgen__test_PyKernelEntryPointRewrite"}} {
  func.func @__nvqpp__mlirgen__test() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.alloca !quake.veq<2>
    %1 = quake.extract_ref %0[0] : (!quake.veq<2>) -> !quake.ref
    quake.h %1 : (!quake.ref) -> ()
    return
  }
}'''
    newMod = kernel.merge_quake_source(otherMod)
    print(newMod)
    assert '__nvqpp__mlirgen__test' in str(
        newMod) and '__nvqpp__mlirgen__kernel' in str(newMod)


def testSynthTwoArgs():

    from typing import Callable

    @cudaq.kernel
    def kernel22(k: Callable[[cudaq.qview], None], j: Callable[[cudaq.qview],
                                                               None]):
        q = cudaq.qvector(2)
        k(q)
        j(q)

    @cudaq.kernel
    def callee0(q: cudaq.qview):
        x(q)

    @cudaq.kernel
    def callee1(q: cudaq.qview):
        x(q)

    ka = kernel22.merge_kernel(callee1)
    print(ka)
    kb = ka.merge_kernel(kernel22)
    print(kb)

    counts = cudaq.sample(kb, callee0, callee1)
    counts.dump()
    assert '00' in counts and len(counts) == 1


def test_cpp_kernel_from_python_0():

    from cudaq_test_cpp_algo import qstd

    @cudaq.kernel
    def callQftAndAnother():
        q = cudaq.qvector(4)
        qstd.qft(q)
        h(q)
        qstd.another(q, 2)

    callQftAndAnother()

    counts = cudaq.sample(callQftAndAnother)
    counts.dump()
    assert len(counts) == 1 and '0010' in counts


def test_cpp_kernel_from_python_1():

    @cudaq.kernel
    def callQftAndAnother():
        q = cudaq.qvector(4)
        cudaq_test_cpp_algo.qstd.qft(q)
        h(q)
        cudaq_test_cpp_algo.qstd.another(q, 2)

    callQftAndAnother()

    counts = cudaq.sample(callQftAndAnother)
    counts.dump()
    assert len(counts) == 1 and '0010' in counts


def test_cpp_kernel_from_python_2():

    @cudaq.kernel
    def callUCCSD():
        q = cudaq.qvector(4)
        cudaq_test_cpp_algo.qstd.uccsd(q, 2)

    callUCCSD()


def test_capture():

    @cudaq.kernel
    def takesCapture(s: int):
        pass

    spin = 0

    @cudaq.kernel(verbose=True)
    def entry():
        takesCapture(spin)

    entry.compile()
