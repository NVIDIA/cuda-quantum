# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest


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
    pytest.importorskip('cudaq_test_cpp_algo')

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

    # TODO: currently not supported;
    # support and test this instead
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def callQftAndAnother(withAdj: bool):
            q = cudaq.qvector(4)
            qstd.qft(q)
            h(q)
            qstd.another(q, 2)
            if withAdj:
                cudaq.adjoint(qstd.another, q, 2)
                h(q)
                cudaq.adjoint(qstd.qft, q)

        counts = cudaq.sample(callQftAndAnother, True)
        assert len(counts) == 1 and '0000' in counts

    assert "calling cudaq.control or cudaq.adjoint on a kernel defined in C++ is not currently supported" in str(
        e.value)


def test_cpp_kernel_from_python_1():
    pytest.importorskip('cudaq_test_cpp_algo')

    import cudaq_test_cpp_algo

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

    # TODO: currently not supported;
    # support and test this instead
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def callQftAndAnother(withAdj: bool):
            q = cudaq.qvector(4)
            cudaq_test_cpp_algo.qstd.qft(q)
            h(q)
            cudaq_test_cpp_algo.qstd.another(q, 2)
            if withAdj:
                cudaq.adjoint(cudaq_test_cpp_algo.qstd.another, q, 2)
                h(q)
                cudaq.adjoint(cudaq_test_cpp_algo.qstd.qft, q)

        counts = cudaq.sample(callQftAndAnother, True)
        assert len(counts) == 1 and '0000' in counts

    assert "calling cudaq.control or cudaq.adjoint on a kernel defined in C++ is not currently supported" in str(
        e.value)


def test_cpp_kernel_from_python_2():
    pytest.importorskip('cudaq_test_cpp_algo')

    import cudaq_test_cpp_algo

    @cudaq.kernel
    def callUCCSD():
        q = cudaq.qvector(4)
        cudaq_test_cpp_algo.qstd.uccsd(q, 2)

    callUCCSD()

    # TODO: currently not supported;
    # support and enable test
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def callUCCSD(setControl: bool):
            c, q = cudaq.qubit(), cudaq.qvector(4)
            if setControl:
                x(c)
            cudaq.control(cudaq_test_cpp_algo.qstd.uccsd, c, q, 2)

        counts = cudaq.sample(callUCCSD, False)
        assert len(counts) == 1 and '0000' in counts
        counts = cudaq.sample(callUCCSD, True)
        assert len(counts) > 1

    assert "calling cudaq.control or cudaq.adjoint on a kernel defined in C++ is not currently supported" in str(
        e.value)


def test_callbacks():
    pytest.importorskip('cudaq_test_cpp_algo')

    import cudaq_test_cpp_algo

    @cudaq.kernel
    def entry(qnum: int):
        qs = cudaq.qvector(qnum)
        h(qs)
        x(qs)

    cudaq_test_cpp_algo.run0(entry, 4)


@pytest.mark.skip(reason="temporarily disabled")
def test_callbacks_b():
    pytest.importorskip('cudaq_test_cpp_algo')

    import cudaq_test_cpp_algo

    @cudaq.kernel
    def entry(qnum: int):
        qs = cudaq.qvector(qnum)
        h(qs)
        z(qs)

    cudaq_test_cpp_algo.run0b(entry, 4)


def test_callback_with_capture():
    pytest.importorskip('cudaq_test_cpp_algo')

    import cudaq_test_cpp_algo

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
    pytest.importorskip('cudaq_test_cpp_algo')

    import cudaq_test_cpp_algo

    @cudaq.kernel
    def entry(qs: cudaq.qview):
        h(qs)
        y(qs)
        h(qs)

    cudaq_test_cpp_algo.run2(entry)


def test_callback_with_capture_quantum_and_classical():
    pytest.importorskip('cudaq_test_cpp_algo')

    import cudaq_test_cpp_algo

    @cudaq.kernel
    def entry(qs: cudaq.qview, i: int):
        h(qs)
        x(qs[i])
        y(qs)
        h(qs)

    cudaq_test_cpp_algo.run3(entry)
