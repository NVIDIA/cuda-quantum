# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest

cudaq_test_cpp_algo = pytest.importorskip(
    'cudaq_test_cpp_algo', reason="cannot find the cudaq_test_cpp_algo.so")


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_mergeExternal():

    @cudaq.kernel
    def kernel(i: int):
        q = cudaq.qvector(i)
        h(q[0])

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
    s = str(newMod)
    assert '__nvqpp__mlirgen__test' in s and '__nvqpp__mlirgen__kernel' in s


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

    # Merge callee1 into kernel22 and then kernel22 into that result. The second
    # merge must be a NOP.
    ka = kernel22.merge_kernel(callee1)
    kb = ka.merge_kernel(kernel22)

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


def test_callbacks():

    @cudaq.kernel
    def entry(qnum: int):
        qs = cudaq.qvector(qnum)
        h(qs)
        x(qs)

    cudaq_test_cpp_algo.run0(entry, 4)


@pytest.mark.skip(reason="temporarily disabled")
def test_callbacks_b():

    @cudaq.kernel
    def entry(qnum: int):
        qs = cudaq.qvector(qnum)
        h(qs)
        z(qs)

    cudaq_test_cpp_algo.run0b(entry, 4)


def test_callback_with_capture():

    @cudaq.kernel
    def captured_qernel(s: int):
        qs = cudaq.qvector(s)
        h(qs)
        y(qs)
        h(qs)

    egb_spin = 6

    @cudaq.kernel
    def entry():
        captured_qernel(egb_spin)

    cudaq_test_cpp_algo.run1(entry)


def test_callback_with_capture_quantum():

    @cudaq.kernel
    def entry(qs: cudaq.qview):
        h(qs)
        y(qs)
        h(qs)

    cudaq_test_cpp_algo.run2(entry)


def test_callback_with_capture_quantum_and_classical():

    @cudaq.kernel
    def entry(qs: cudaq.qview, i: int):
        h(qs)
        x(qs[i])
        y(qs)
        h(qs)

    cudaq_test_cpp_algo.run3(entry)
