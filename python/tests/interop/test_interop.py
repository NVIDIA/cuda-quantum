# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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


def test_call_python_from_cpp():

    @cudaq.kernel
    def statePrep(q: cudaq.qvector):
        x(q)

    # The test cpp qalgo just takes the user statePrep and
    # applies it to a 2 qubit register, so we should expect 11 in the output.
    counts = cudaq_test_cpp_algo.test_cpp_qalgo(statePrep)
    counts.dump()
    assert len(counts) == 1 and '11' in counts


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
    newMod = kernel.merge_kernel(otherMod)
    print(newMod)
    assert '__nvqpp__mlirgen__test' in str(
        newMod) and '__nvqpp__mlirgen__kernel' in str(newMod)


def test_synthCallable():

    @cudaq.kernel
    def callee(q: cudaq.qview):
        x(q[0])
        x(q[1])

    callee.compile()

    otherMod = '''module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__caller = "__nvqpp__mlirgen__caller_PyKernelEntryPointRewrite"}} {
  func.func @__nvqpp__mlirgen__caller(%arg0: !cc.callable<(!quake.veq<?>) -> ()>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.alloca !quake.veq<2>
    %1 = quake.relax_size %0 : (!quake.veq<2>) -> !quake.veq<?>
    %2 = cc.callable_func %arg0 : (!cc.callable<(!quake.veq<?>) -> ()>) -> ((!quake.veq<?>) -> ())
    call_indirect %2(%1) : (!quake.veq<?>) -> ()
    return
  }
}'''

    # Merge the external code with the current pure device kernel
    newKernel = callee.merge_kernel(otherMod)
    print(newKernel.name, newKernel)
    # Synthesize away the callable arg with the pure device kernel
    newKernel.synthesize_callable_arguments(['callee'])
    print(newKernel)

    counts = cudaq.sample(newKernel)
    assert len(counts) == 1 and '11' in counts


def test_synthCallableCCCallCallableOp():

    @cudaq.kernel
    def callee(q: cudaq.qview):
        x(q[0])
        x(q[1])

    callee.compile()

    otherMod = '''module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__caller = "__nvqpp__mlirgen__caller_PyKernelEntryPointRewrite"}} {
  func.func @__nvqpp__mlirgen__adapt_caller(%arg0: i64, %arg1: !cc.callable<(!quake.veq<?>) -> ()>, %arg2: !cc.stdvec<f64>, %arg3: !cc.stdvec<f64>, %arg4: !cc.stdvec<!cc.charspan>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = quake.alloca !quake.veq<?>[%arg0 : i64]
    cc.call_callable %arg1, %0 : (!cc.callable<(!quake.veq<?>) -> ()>, !quake.veq<?>) -> ()
    %1 = cc.loop while ((%arg5 = %c0_i64) -> (i64)) {
      %2 = cc.stdvec_size %arg2 : (!cc.stdvec<f64>) -> i64
      %3 = arith.cmpi ult, %arg5, %2 : i64
      cc.condition %3(%arg5 : i64)
    } do {
    ^bb0(%arg5: i64):
      %2 = cc.loop while ((%arg6 = %c0_i64) -> (i64)) {
        %3 = cc.stdvec_size %arg4 : (!cc.stdvec<!cc.charspan>) -> i64
        %4 = arith.cmpi ult, %arg6, %3 : i64
        cc.condition %4(%arg6 : i64)
      } do {
      ^bb0(%arg6: i64):
        %3 = cc.stdvec_data %arg2 : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
        %4 = cc.compute_ptr %3[%arg5] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
        %5 = cc.load %4 : !cc.ptr<f64>
        %6 = cc.stdvec_data %arg3 : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
        %7 = cc.compute_ptr %6[%arg6] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
        %8 = cc.load %7 : !cc.ptr<f64>
        %9 = arith.mulf %5, %8 : f64
        %10 = cc.stdvec_data %arg4 : (!cc.stdvec<!cc.charspan>) -> !cc.ptr<!cc.array<!cc.charspan x ?>>
        %11 = cc.compute_ptr %10[%arg6] : (!cc.ptr<!cc.array<!cc.charspan x ?>>, i64) -> !cc.ptr<!cc.charspan>
        %12 = cc.load %11 : !cc.ptr<!cc.charspan>
        quake.exp_pauli %9, %0, %12 : (f64, !quake.veq<?>, !cc.charspan) -> ()
        cc.continue %arg6 : i64
      } step {
      ^bb0(%arg6: i64):
        %3 = arith.addi %arg6, %c1_i64 : i64
        cc.continue %3 : i64
      }
      cc.continue %arg5 : i64
    } step {
    ^bb0(%arg5: i64):
      %2 = arith.addi %arg5, %c1_i64 : i64
      cc.continue %2 : i64
    }
    return
  }
}'''

    # Merge the external code with the current pure device kernel
    newKernel = callee.merge_kernel(otherMod)
    print(newKernel)
    # Synthesize away the callable arg with the pure device kernel
    newKernel.synthesize_callable_arguments(['callee'])
    print(newKernel)
    assert '!cc.callable' not in str(newKernel)


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

    callees = callee0.merge_kernel(callee1)
    print(callees)
    merged = callees.merge_kernel(kernel22)
    print(merged)

    merged.synthesize_callable_arguments(['callee0', 'callee1'])

    print(merged)
    counts = cudaq.sample(merged)
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
