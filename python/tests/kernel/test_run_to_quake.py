# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""`cudaq.run` on objects that lower to Quake through a `to_quake()` method."""

import re

import cudaq
import numpy as np
import pytest
from cudaq.mlir.ir import Context, Module

requires_gpu = pytest.mark.skipif(cudaq.num_available_gpus() == 0,
                                  reason="the nvidia target needs a GPU")

# Admission adds no target-specific path, so every test runs on every target.
# The fixtures are Clifford, as `stim` requires.
TARGETS = [
    pytest.param(("qpp-cpu", {}), id="qpp-cpu"),
    pytest.param(("density-matrix-cpu", {}), id="density-matrix-cpu"),
    pytest.param(("stim", {}), id="stim"),
    pytest.param(("nvidia", {}), id="nvidia-fp32", marks=requires_gpu),
    pytest.param(("nvidia", dict(option="fp64")),
                 id="nvidia-fp64",
                 marks=requires_gpu),
]


@pytest.fixture(autouse=True, params=TARGETS)
def target(request):
    name, options = request.param
    cudaq.set_target(name, **options)
    try:
        yield
    finally:
        cudaq.reset_target()


class QuakeArtifact:
    """Stands in for a compiler artifact that lowers to Quake. Each
    `to_quake()` call returns a new module in an MLIR context of its own."""

    def __init__(self, quake):
        self.quake = quake
        self.conversions = 0

    def to_quake(self):
        self.conversions += 1
        context = Context()
        context.allow_unregistered_dialects = True
        return Module.parse(self.quake, context=context)


# The fixtures are frontend output, not hand-written IR. Each is the Quake that
# the Python frontend emits for the kernel in the comment above it, captured
# before `aot-prep-pipeline`, with its module attributes removed and its entry
# point renamed. A "wire form" fixture then went through `cudaq-opt
# --pass-pipeline='builtin.module(func.func(unwind-lowering,add-dealloc,
# factor-quantum-alloc,memtoreg{classical=false quantum=true}),canonicalize,
# cse,symbol-dce)'`, which turns qubit references into `!quake.wire` values.

# @cudaq.kernel
# def bool_result() -> bool:
#     q = cudaq.qubit()
#     x(q)
#     return mz(q)
BOOL_RESULT = """
module {
  func.func @bool_result() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.alloca !quake.ref
    quake.x %0 : (!quake.ref) -> ()
    %measOut = quake.mz %0 : (!quake.ref) -> !cc.measure_handle
    %1 = quake.discriminate %measOut : (!cc.measure_handle) -> i1
    return %1 : i1
  }
}
"""

# @cudaq.kernel
# def i8_result(a: int) -> np.int8:
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
#     return a - 1
I8_RESULT = """
module {
  func.func @i8_result(%arg0: i64) -> i8 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.alloca i64
    cc.store %arg0, %0 : !cc.ptr<i64>
    %1 = quake.alloca !quake.ref
    quake.x %1 : (!quake.ref) -> ()
    %measOut = quake.mz %1 : (!quake.ref) -> !cc.measure_handle
    %2 = cc.load %0 : !cc.ptr<i64>
    %c1_i64 = arith.constant 1 : i64
    %3 = arith.subi %2, %c1_i64 : i64
    %4 = cc.cast %3 : (i64) -> i8
    return %4 : i8
  }
}
"""

# @cudaq.kernel
# def i16_result(a: int) -> np.int16:
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
#     return a - 1
I16_RESULT = """
module {
  func.func @i16_result(%arg0: i64) -> i16 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.alloca i64
    cc.store %arg0, %0 : !cc.ptr<i64>
    %1 = quake.alloca !quake.ref
    quake.x %1 : (!quake.ref) -> ()
    %measOut = quake.mz %1 : (!quake.ref) -> !cc.measure_handle
    %2 = cc.load %0 : !cc.ptr<i64>
    %c1_i64 = arith.constant 1 : i64
    %3 = arith.subi %2, %c1_i64 : i64
    %4 = cc.cast %3 : (i64) -> i16
    return %4 : i16
  }
}
"""

# @cudaq.kernel
# def i32_result(a: int) -> np.int32:
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
#     return a - 1
I32_RESULT = """
module {
  func.func @i32_result(%arg0: i64) -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.alloca i64
    cc.store %arg0, %0 : !cc.ptr<i64>
    %1 = quake.alloca !quake.ref
    quake.x %1 : (!quake.ref) -> ()
    %measOut = quake.mz %1 : (!quake.ref) -> !cc.measure_handle
    %2 = cc.load %0 : !cc.ptr<i64>
    %c1_i64 = arith.constant 1 : i64
    %3 = arith.subi %2, %c1_i64 : i64
    %4 = cc.cast %3 : (i64) -> i32
    return %4 : i32
  }
}
"""

# @cudaq.kernel
# def i64_result(a: int) -> int:
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
#     return a - 1
I64_RESULT = """
module {
  func.func @i64_result(%arg0: i64) -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.alloca i64
    cc.store %arg0, %0 : !cc.ptr<i64>
    %1 = quake.alloca !quake.ref
    quake.x %1 : (!quake.ref) -> ()
    %measOut = quake.mz %1 : (!quake.ref) -> !cc.measure_handle
    %2 = cc.load %0 : !cc.ptr<i64>
    %c1_i64 = arith.constant 1 : i64
    %3 = arith.subi %2, %c1_i64 : i64
    return %3 : i64
  }
}
"""

# @cudaq.kernel
# def f32_result(theta: float) -> np.float32:
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
#     return theta - 0.5
F32_RESULT = """
module {
  func.func @f32_result(%arg0: f64) -> f32 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.alloca f64
    cc.store %arg0, %0 : !cc.ptr<f64>
    %1 = quake.alloca !quake.ref
    quake.x %1 : (!quake.ref) -> ()
    %measOut = quake.mz %1 : (!quake.ref) -> !cc.measure_handle
    %2 = cc.load %0 : !cc.ptr<f64>
    %cst = arith.constant 5.000000e-01 : f64
    %3 = arith.subf %2, %cst : f64
    %4 = cc.cast %3 : (f64) -> f32
    return %4 : f32
  }
}
"""

# @cudaq.kernel
# def f64_result(theta: float) -> float:
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
#     return theta - 0.5
F64_RESULT = """
module {
  func.func @f64_result(%arg0: f64) -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.alloca f64
    cc.store %arg0, %0 : !cc.ptr<f64>
    %1 = quake.alloca !quake.ref
    quake.x %1 : (!quake.ref) -> ()
    %measOut = quake.mz %1 : (!quake.ref) -> !cc.measure_handle
    %2 = cc.load %0 : !cc.ptr<f64>
    %cst = arith.constant 5.000000e-01 : f64
    %3 = arith.subf %2, %cst : f64
    return %3 : f64
  }
}
"""

# @cudaq.kernel
# def tuple_result(a: int, f: float) -> tuple[bool, int, float]:
#     q = cudaq.qubit()
#     x(q)
#     return (mz(q), a, f)
# Wire form.
TUPLE_RESULT = """
module {
  func.func @tuple_result(%arg0: i64, %arg1: f64) -> !cc.struct<"tuple" {i1, i64, f64}> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.alloca f64
    %1 = cc.alloca i64
    cc.store %arg0, %1 : !cc.ptr<i64>
    cc.store %arg1, %0 : !cc.ptr<f64>
    %2 = quake.null_wire
    %3 = quake.x %2 : (!quake.wire) -> !quake.wire
    %measOut, %wires = quake.mz %3 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %4 = cc.load %1 : !cc.ptr<i64>
    %5 = cc.load %0 : !cc.ptr<f64>
    %6 = quake.discriminate %measOut : (!cc.measure_handle) -> i1
    %7 = cc.undef !cc.struct<"tuple" {i1, i64, f64}>
    %8 = cc.insert_value %7[0], %6 : (!cc.struct<"tuple" {i1, i64, f64}>, i1) -> !cc.struct<"tuple" {i1, i64, f64}>
    %9 = cc.insert_value %8[1], %4 : (!cc.struct<"tuple" {i1, i64, f64}>, i64) -> !cc.struct<"tuple" {i1, i64, f64}>
    %10 = cc.insert_value %9[2], %5 : (!cc.struct<"tuple" {i1, i64, f64}>, f64) -> !cc.struct<"tuple" {i1, i64, f64}>
    quake.sink %wires : !quake.wire
    return %10 : !cc.struct<"tuple" {i1, i64, f64}>
  }
}
"""

# @cudaq.kernel
# def bool_list_result() -> list[bool]:
#     a = cudaq.qubit()
#     b = cudaq.qubit()
#     c = cudaq.qubit()
#     x(a)
#     x(c)
#     return [mz(a), mz(b), mz(c)]
# Wire form.
BOOL_LIST_RESULT = """
module {
  func.func @bool_list_result() -> !cc.sequence<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c1_i64 = arith.constant 1 : i64
    %c3_i64 = arith.constant 3 : i64
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %3 = quake.x %0 : (!quake.wire) -> !quake.wire
    %4 = quake.x %2 : (!quake.wire) -> !quake.wire
    %measOut, %wires = quake.mz %3 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %measOut_0, %wires_1 = quake.mz %1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %measOut_2, %wires_3 = quake.mz %4 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %5 = cc.alloca !cc.array<!cc.measure_handle x 3>
    %6 = cc.cast %5 : (!cc.ptr<!cc.array<!cc.measure_handle x 3>>) -> !cc.ptr<!cc.array<!cc.measure_handle x ?>>
    %7 = cc.cast %5 : (!cc.ptr<!cc.array<!cc.measure_handle x 3>>) -> !cc.ptr<!cc.measure_handle>
    cc.store %measOut, %7 : !cc.ptr<!cc.measure_handle>
    %8 = cc.compute_ptr %5[1] : (!cc.ptr<!cc.array<!cc.measure_handle x 3>>) -> !cc.ptr<!cc.measure_handle>
    cc.store %measOut_0, %8 : !cc.ptr<!cc.measure_handle>
    %9 = cc.compute_ptr %5[2] : (!cc.ptr<!cc.array<!cc.measure_handle x 3>>) -> !cc.ptr<!cc.measure_handle>
    cc.store %measOut_2, %9 : !cc.ptr<!cc.measure_handle>
    %10 = cc.sequence_init %6, %c3_i64 : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>, i64) -> !cc.sequence<!cc.measure_handle>
    %11 = quake.discriminate %10 : (!cc.sequence<!cc.measure_handle>) -> !cc.sequence<i1>
    %12 = cc.sequence_data %11 : (!cc.sequence<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
    %13 = cc.sequence_size %11 : (!cc.sequence<i1>) -> i64
    %14 = cc.cast %12 : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
    %15 = call @__nvqpp_vectorCopyCtor(%14, %13, %c1_i64) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
    %16 = cc.sequence_init %15, %13 : (!cc.ptr<i8>, i64) -> !cc.sequence<i1>
    quake.sink %wires : !quake.wire
    quake.sink %wires_3 : !quake.wire
    quake.sink %wires_1 : !quake.wire
    return %16 : !cc.sequence<i1>
  }
  func.func private @llvm.memcpy.p0.p0.i64(!cc.ptr<i8>, !cc.ptr<i8>, i64, i1)
  func.func private @malloc(i64) -> !cc.ptr<i8>
  func.func private @__nvqpp_vectorCopyCtor(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: i64) -> !cc.ptr<i8> {
    %false = arith.constant false
    %0 = arith.muli %arg1, %arg2 : i64
    %1 = call @malloc(%0) : (i64) -> !cc.ptr<i8>
    call @llvm.memcpy.p0.p0.i64(%1, %arg0, %0, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    return %1 : !cc.ptr<i8>
  }
}
"""

# @cudaq.kernel
# def float_list_result() -> list[float]:
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
#     return [-2.5, 0.25, 1.0e300]
# Wire form.
FLOAT_LIST_RESULT = """
module {
  func.func @float_list_result() -> !cc.sequence<f64> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant -2.500000e+00 : f64
    %c3_i64 = arith.constant 3 : i64
    %cst_0 = arith.constant 1.000000e+300 : f64
    %cst_1 = arith.constant 2.500000e-01 : f64
    %0 = quake.null_wire
    %1 = quake.x %0 : (!quake.wire) -> !quake.wire
    %measOut, %wires = quake.mz %1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %2 = cc.alloca !cc.array<f64 x 3>
    %3 = cc.cast %2 : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
    cc.store %cst, %3 : !cc.ptr<f64>
    %4 = cc.compute_ptr %2[1] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
    cc.store %cst_1, %4 : !cc.ptr<f64>
    %5 = cc.compute_ptr %2[2] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
    cc.store %cst_0, %5 : !cc.ptr<f64>
    %6 = cc.cast %2 : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<i8>
    %7 = call @__nvqpp_vectorCopyCtor(%6, %c3_i64, %c8_i64) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
    %8 = cc.sequence_init %7, %c3_i64 : (!cc.ptr<i8>, i64) -> !cc.sequence<f64>
    quake.sink %wires : !quake.wire
    return %8 : !cc.sequence<f64>
  }
  func.func private @llvm.memcpy.p0.p0.i64(!cc.ptr<i8>, !cc.ptr<i8>, i64, i1)
  func.func private @malloc(i64) -> !cc.ptr<i8>
  func.func private @__nvqpp_vectorCopyCtor(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: i64) -> !cc.ptr<i8> {
    %false = arith.constant false
    %0 = arith.muli %arg1, %arg2 : i64
    %1 = call @malloc(%0) : (i64) -> !cc.ptr<i8>
    call @llvm.memcpy.p0.p0.i64(%1, %arg0, %0, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    return %1 : !cc.ptr<i8>
  }
}
"""

# @cudaq.kernel
# def list_from_argument(v: list[int]) -> list[int]:
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
#     return v
# Wire form.
LIST_FROM_ARGUMENT = """
module {
  func.func @list_from_argument(%arg0: !cc.sequence<i64>) -> !cc.sequence<i64> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c8_i64 = arith.constant 8 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = cc.alloca !cc.sequence<i64>
    %1 = cc.cast %c0_i64 : (i64) -> !cc.ptr<!cc.array<i64 x ?>>
    %2 = cc.sequence_init %1, %c0_i64 : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.sequence<i64>
    cc.store %2, %0 : !cc.ptr<!cc.sequence<i64>>
    cc.store %arg0, %0 : !cc.ptr<!cc.sequence<i64>>
    %3 = quake.null_wire
    %4 = quake.x %3 : (!quake.wire) -> !quake.wire
    %measOut, %wires = quake.mz %4 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %5 = cc.load %0 : !cc.ptr<!cc.sequence<i64>>
    %6 = cc.sequence_data %5 : (!cc.sequence<i64>) -> !cc.ptr<!cc.array<i8 x ?>>
    %7 = cc.sequence_size %5 : (!cc.sequence<i64>) -> i64
    %8 = cc.cast %6 : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
    %9 = call @__nvqpp_vectorCopyCtor(%8, %7, %c8_i64) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
    %10 = cc.sequence_init %9, %7 : (!cc.ptr<i8>, i64) -> !cc.sequence<i64>
    quake.sink %wires : !quake.wire
    return %10 : !cc.sequence<i64>
  }
  func.func private @llvm.memcpy.p0.p0.i64(!cc.ptr<i8>, !cc.ptr<i8>, i64, i1)
  func.func private @malloc(i64) -> !cc.ptr<i8>
  func.func private @__nvqpp_vectorCopyCtor(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: i64) -> !cc.ptr<i8> {
    %false = arith.constant false
    %0 = arith.muli %arg1, %arg2 : i64
    %1 = call @malloc(%0) : (i64) -> !cc.ptr<i8>
    call @llvm.memcpy.p0.p0.i64(%1, %arg0, %0, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    return %1 : !cc.ptr<i8>
  }
}
"""

# C++ frontend output (`cudaq-quake -std=c++20`) for `list_from_helper`
# below, cut down to that function and the functions it calls, in wire form.
# Each `return` copies its vector to the heap (`copyDynamicValueToHeap`), and
# the call copies the result back to the stack (`copyDynamicValueToStack`).
#
#   __qpu__ std::vector<long> cpp_helper(bool flip) {
#     if (flip)
#       return {-1, 0, 1099511627779};
#     return {5, 5, 5};
#   }
#
#   __qpu__ std::vector<long> list_from_helper() {
#     cudaq::qubit q;
#     x(q);
#     bool b = mz(q);
#     return cpp_helper(b);
#   }
LIST_FROM_HELPER = """
module {
  func.func private @cpp_helper(%arg0: i1) -> !cc.sequence<i64> attributes {"cudaq-kernel", no_this} {
    %c5_i64 = arith.constant 5 : i64
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1099511627779_i64 = arith.constant 1099511627779 : i64
    %0 = cc.alloca i1
    cc.store %arg0, %0 : !cc.ptr<i1>
    %1 = cc.load %0 : !cc.ptr<i1>
    cf.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = cc.alloca !cc.array<i64 x 3>
    %3 = cc.cast %2 : (!cc.ptr<!cc.array<i64 x 3>>) -> !cc.ptr<i64>
    cc.store %c-1_i64, %3 : !cc.ptr<i64>
    %4 = cc.compute_ptr %2[1] : (!cc.ptr<!cc.array<i64 x 3>>) -> !cc.ptr<i64>
    cc.store %c0_i64, %4 : !cc.ptr<i64>
    %5 = cc.compute_ptr %2[2] : (!cc.ptr<!cc.array<i64 x 3>>) -> !cc.ptr<i64>
    cc.store %c1099511627779_i64, %5 : !cc.ptr<i64>
    %6 = cc.cast %2 : (!cc.ptr<!cc.array<i64 x 3>>) -> !cc.ptr<i8>
    %7 = call @__nvqpp_vectorCopyCtor(%6, %c3_i64, %c8_i64) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
    %8 = cc.sequence_init %7, %c3_i64 : (!cc.ptr<i8>, i64) -> !cc.sequence<i64>
    return %8 : !cc.sequence<i64>
  ^bb2:  // pred: ^bb0
    %9 = cc.alloca !cc.array<i64 x 3>
    %10 = cc.cast %9 : (!cc.ptr<!cc.array<i64 x 3>>) -> !cc.ptr<i64>
    cc.store %c5_i64, %10 : !cc.ptr<i64>
    %11 = cc.compute_ptr %9[1] : (!cc.ptr<!cc.array<i64 x 3>>) -> !cc.ptr<i64>
    cc.store %c5_i64, %11 : !cc.ptr<i64>
    %12 = cc.compute_ptr %9[2] : (!cc.ptr<!cc.array<i64 x 3>>) -> !cc.ptr<i64>
    cc.store %c5_i64, %12 : !cc.ptr<i64>
    %13 = cc.cast %9 : (!cc.ptr<!cc.array<i64 x 3>>) -> !cc.ptr<i8>
    %14 = call @__nvqpp_vectorCopyCtor(%13, %c3_i64, %c8_i64) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
    %15 = cc.sequence_init %14, %c3_i64 : (!cc.ptr<i8>, i64) -> !cc.sequence<i64>
    return %15 : !cc.sequence<i64>
  }
  func.func @list_from_helper() -> !cc.sequence<i64> attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %c8_i64 = arith.constant 8 : i64
    %0 = quake.null_wire
    %1 = quake.x %0 : (!quake.wire) -> !quake.wire
    %measOut, %wires = quake.mz %1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %2 = cc.alloca !cc.measure_handle
    cc.store %measOut, %2 : !cc.ptr<!cc.measure_handle>
    %3 = cc.load %2 : !cc.ptr<!cc.measure_handle>
    %4 = quake.discriminate %3 : (!cc.measure_handle) -> i1
    %5 = cc.alloca i1
    cc.store %4, %5 : !cc.ptr<i1>
    %6 = cc.load %5 : !cc.ptr<i1>
    %7 = call @cpp_helper(%6) : (i1) -> !cc.sequence<i64>
    %8 = cc.sequence_data %7 : (!cc.sequence<i64>) -> !cc.ptr<i64>
    %9 = cc.sequence_size %7 : (!cc.sequence<i64>) -> i64
    %10 = arith.muli %9, %c8_i64 : i64
    %11 = cc.alloca i64[%9 : i64]
    %12 = cc.cast %11 : (!cc.ptr<!cc.array<i64 x ?>>) -> !cc.ptr<i8>
    %13 = cc.cast %8 : (!cc.ptr<i64>) -> !cc.ptr<i8>
    call @__nvqpp_vectorCopyToStack(%12, %13, %10) : (!cc.ptr<i8>, !cc.ptr<i8>, i64) -> ()
    %14 = call @__nvqpp_vectorCopyCtor(%12, %9, %c8_i64) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
    %15 = cc.sequence_init %14, %9 : (!cc.ptr<i8>, i64) -> !cc.sequence<i64>
    quake.sink %wires : !quake.wire
    return %15 : !cc.sequence<i64>
  }
  func.func private @llvm.memcpy.p0.p0.i64(!cc.ptr<i8>, !cc.ptr<i8>, i64, i1)
  func.func private @malloc(i64) -> !cc.ptr<i8>
  func.func private @__nvqpp_vectorCopyCtor(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: i64) -> !cc.ptr<i8> {
    %false = arith.constant false
    %0 = arith.muli %arg1, %arg2 : i64
    %1 = call @malloc(%0) : (i64) -> !cc.ptr<i8>
    call @llvm.memcpy.p0.p0.i64(%1, %arg0, %0, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    return %1 : !cc.ptr<i8>
  }
  func.func private @free(!cc.ptr<i8>)
  func.func private @__nvqpp_vectorCopyToStack(%arg0: !cc.ptr<i8>, %arg1: !cc.ptr<i8>, %arg2: i64) {
    %false = arith.constant false
    call @llvm.memcpy.p0.p0.i64(%arg0, %arg1, %arg2, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    call @free(%arg1) : (!cc.ptr<i8>) -> ()
    return
  }
}
"""

# @cudaq.kernel
# def bell() -> list[bool]:
#     a = cudaq.qubit()
#     b = cudaq.qubit()
#     h(a)
#     x.ctrl(a, b)
#     return [mz(a), mz(b)]
# Wire form.
BELL = """
module {
  func.func @bell() -> !cc.sequence<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.h %0 : (!quake.wire) -> !quake.wire
    %3:2 = quake.x [%2] %1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    %measOut, %wires = quake.mz %3#0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %measOut_0, %wires_1 = quake.mz %3#1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %4 = cc.alloca !cc.array<!cc.measure_handle x 2>
    %5 = cc.cast %4 : (!cc.ptr<!cc.array<!cc.measure_handle x 2>>) -> !cc.ptr<!cc.array<!cc.measure_handle x ?>>
    %6 = cc.cast %4 : (!cc.ptr<!cc.array<!cc.measure_handle x 2>>) -> !cc.ptr<!cc.measure_handle>
    cc.store %measOut, %6 : !cc.ptr<!cc.measure_handle>
    %7 = cc.compute_ptr %4[1] : (!cc.ptr<!cc.array<!cc.measure_handle x 2>>) -> !cc.ptr<!cc.measure_handle>
    cc.store %measOut_0, %7 : !cc.ptr<!cc.measure_handle>
    %8 = cc.sequence_init %5, %c2_i64 : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>, i64) -> !cc.sequence<!cc.measure_handle>
    %9 = quake.discriminate %8 : (!cc.sequence<!cc.measure_handle>) -> !cc.sequence<i1>
    %10 = cc.sequence_data %9 : (!cc.sequence<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
    %11 = cc.sequence_size %9 : (!cc.sequence<i1>) -> i64
    %12 = cc.cast %10 : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
    %13 = call @__nvqpp_vectorCopyCtor(%12, %11, %c1_i64) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
    %14 = cc.sequence_init %13, %11 : (!cc.ptr<i8>, i64) -> !cc.sequence<i1>
    quake.sink %wires : !quake.wire
    quake.sink %wires_1 : !quake.wire
    return %14 : !cc.sequence<i1>
  }
  func.func private @llvm.memcpy.p0.p0.i64(!cc.ptr<i8>, !cc.ptr<i8>, i64, i1)
  func.func private @malloc(i64) -> !cc.ptr<i8>
  func.func private @__nvqpp_vectorCopyCtor(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: i64) -> !cc.ptr<i8> {
    %false = arith.constant false
    %0 = arith.muli %arg1, %arg2 : i64
    %1 = call @malloc(%0) : (i64) -> !cc.ptr<i8>
    call @llvm.memcpy.p0.p0.i64(%1, %arg0, %0, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    return %1 : !cc.ptr<i8>
  }
}
"""

# @cudaq.kernel
# def saved_after_reset() -> bool:
#     q = cudaq.qubit()
#     x(q)
#     m = mz(q)
#     reset(q)
#     return m
# Wire form.
SAVED_AFTER_RESET = """
module {
  func.func @saved_after_reset() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.alloca !cc.measure_handle
    %1 = quake.null_wire
    %2 = quake.x %1 : (!quake.wire) -> !quake.wire
    %measOut, %wires = quake.mz %2 name "m" : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    cc.store %measOut, %0 : !cc.ptr<!cc.measure_handle>
    %3 = quake.reset %wires : (!quake.wire) -> !quake.wire
    %4 = cc.load %0 : !cc.ptr<!cc.measure_handle>
    %5 = quake.discriminate %4 : (!cc.measure_handle) -> i1
    quake.sink %3 : !quake.wire
    return %5 : i1
  }
}
"""

# @cudaq.kernel
# def void_entry():
#     q = cudaq.qubit()
#     x(q)
#     mz(q)
# Wire form.
VOID_ENTRY = """
module {
  func.func @void_entry() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.x %0 : (!quake.wire) -> !quake.wire
    %measOut, %wires = quake.mz %1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    quake.sink %wires : !quake.wire
    return
  }
}
"""

# @cudaq.kernel
# def apply_and_measure(f: Callable[[cudaq.qubit], None]) -> bool:
#     q = cudaq.qubit()
#     f(q)
#     return mz(q)
APPLY_AND_MEASURE = """
module {
  func.func @apply_and_measure(%arg0: !cc.callable<(!quake.ref) -> ()>) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.alloca !quake.ref
    cc.call_callable %arg0, %0 : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> () {symbol = "f"}
    %measOut = quake.mz %0 : (!quake.ref) -> !cc.measure_handle
    %1 = quake.discriminate %measOut : (!cc.measure_handle) -> i1
    return %1 : i1
  }
}
"""

# Hand-written, because no frontend emits these forms.
TWO_ENTRY_POINTS = """
module {
  func.func @first() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %true = arith.constant true
    return %true : i1
  }
  func.func @second() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %false = arith.constant false
    return %false : i1
  }
}
"""

UNNAMED_STRUCT_RESULT = """
module {
  func.func @unnamed_struct() -> !cc.struct<{i1, i64}> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.undef !cc.struct<{i1, i64}>
    return %0 : !cc.struct<{i1, i64}>
  }
}
"""

# A frontend fixture with an operation outside the CUDA-Q dialects.
FOREIGN_OPERATION = BOOL_RESULT.replace(
    "    return", '    "foreign.op"() : () -> ()\n    return')


@cudaq.kernel
def flip(q: cudaq.qubit):
    x(q)


def python_types(value):
    if isinstance(value, (list, tuple)):
        return type(value), [type(element) for element in value]
    return type(value)


@pytest.mark.parametrize("quake, args, expected", [
    pytest.param(BOOL_RESULT, (), True, id="i1"),
    pytest.param(I8_RESULT, (-127,), -128, id="i8"),
    pytest.param(I16_RESULT, (-32767,), -32768, id="i16"),
    pytest.param(I32_RESULT, (-2147483647,), -2147483648, id="i32"),
    pytest.param(I64_RESULT, (2**40 + 4,), 2**40 + 3, id="i64"),
    pytest.param(F32_RESULT, (0.75,), 0.25, id="f32"),
    pytest.param(F64_RESULT, (-2.0,), -2.5, id="f64"),
    pytest.param(TUPLE_RESULT, (2**40 + 3, -2.5), (True, 2**40 + 3, -2.5),
                 id="tuple"),
    pytest.param(BOOL_LIST_RESULT, (), [True, False, True], id="bool-list"),
    pytest.param(FLOAT_LIST_RESULT, (), [-2.5, 0.25, 1.0e300], id="float-list"),
    pytest.param(LIST_FROM_ARGUMENT, ([-1, 0, 2**40 + 3],), [-1, 0, 2**40 + 3],
                 id="list-from-argument"),
    pytest.param(LIST_FROM_HELPER, (), [-1, 0, 2**40 + 3],
                 id="list-from-helper"),
    pytest.param(SAVED_AFTER_RESET, (), True, id="saved-before-reset"),
])
def test_results(quake, args, expected):
    results = cudaq.run(QuakeArtifact(quake), *args, shots_count=3)
    assert results == [expected] * 3
    # The same Python types as for an ordinary kernel.
    assert all(python_types(r) == python_types(expected) for r in results)


def test_bell_pair():
    results = cudaq.run(QuakeArtifact(BELL), shots_count=100)
    assert all(len(bits) == 2 and bits[0] == bits[1] for bits in results)
    # Both outcomes occur. All 100 shots agree with probability 2**-99.
    assert {bits[0] for bits in results} == {False, True}


def test_arguments_convert_as_for_any_kernel():
    assert cudaq.run(QuakeArtifact(F64_RESULT), 2, shots_count=2) == [1.5] * 2
    assert cudaq.run(QuakeArtifact(LIST_FROM_ARGUMENT),
                     np.array([3, 4]),
                     shots_count=2) == [[3, 4]] * 2
    assert cudaq.run(QuakeArtifact(APPLY_AND_MEASURE), flip,
                     shots_count=2) == [True] * 2


def test_to_quake_is_called_once_per_run():
    artifact = QuakeArtifact(BOOL_RESULT)
    assert cudaq.run(artifact, shots_count=10) == [True] * 10
    assert cudaq.run(artifact, shots_count=0) == []
    assert artifact.conversions == 2


def test_source_module_is_unchanged():
    module = Module.parse(I64_RESULT, context=Context())
    before = str(module)

    class Holder:

        def to_quake(self):
            return module

    assert cudaq.run(Holder(), 5, shots_count=2) == [4, 4]
    assert str(module) == before


def test_to_quake_errors():

    class Failing:

        def to_quake(self):
            raise ValueError("lowering failed")

    class NotAModule:

        def to_quake(self):
            return BOOL_RESULT

    with pytest.raises(ValueError, match="lowering failed"):
        cudaq.run(Failing())
    with pytest.raises(TypeError, match="must return a .*Module`, not str"):
        cudaq.run(NotAModule())


@pytest.mark.parametrize("quake, message", [
    pytest.param(BOOL_RESULT.replace('"cudaq-entrypoint", ', ""),
                 "found 0$",
                 id="no-entry-point"),
    pytest.param(TWO_ENTRY_POINTS, "found 2 @first @second$", id="two-entry"),
    pytest.param(FOREIGN_OPERATION,
                 "'foreign.op' at .*, which is not in a CUDA-Q dialect",
                 id="foreign-operation"),
    pytest.param(
        UNNAMED_STRUCT_RESULT, "a struct without a name", id="unnamed-struct"),
])
def test_rejected_modules(quake, message):
    # `test_results` runs the same modules without these defects.
    with pytest.raises(RuntimeError, match=message):
        cudaq.run(QuakeArtifact(quake))


def test_run_checks_apply_as_for_any_kernel():
    artifact = QuakeArtifact(I64_RESULT)
    assert cudaq.run(artifact, 1, shots_count=0) == []
    # Arity is checked before the shot count.
    with pytest.raises(RuntimeError,
                       match=re.escape("Invalid number of arguments passed to "
                                       "run. 0 given and 1 expected.")):
        cudaq.run(artifact, shots_count=-1)
    with pytest.raises(RuntimeError, match="Invalid `shots_count`"):
        cudaq.run(artifact, 1, shots_count=-1)
    void = QuakeArtifact(VOID_ENTRY)
    assert cudaq.run(void, shots_count=0) == []
    with pytest.raises(RuntimeError,
                       match="a runnable kernel must return a value"):
        cudaq.run(void, shots_count=1)
    # Runs still succeed after the failed ones.
    assert cudaq.run(artifact, 1, shots_count=2) == [0, 0]


def test_modules_with_the_same_entry_name_run_independently():
    first = QuakeArtifact(I64_RESULT.replace("@i64_result", "@entry"))
    second = QuakeArtifact(F64_RESULT.replace("@f64_result", "@entry"))
    assert cudaq.run(first, 10, shots_count=2) == [9, 9]
    assert cudaq.run(second, 1.0, shots_count=2) == [0.5, 0.5]
    assert cudaq.run(first, 10, shots_count=2) == [9, 9]


@pytest.mark.parametrize(
    "entry",
    [
        # The C++ frontend's convention.
        "__nvqpp__mlirgen__function_bool_result._Z11bool_resultv",
        # The Python frontend's convention.
        "__nvqpp__mlirgen__bool_result..0x7f0012345678",
    ])
def test_entry_point_names_in_frontend_conventions(entry):
    quake = BOOL_RESULT.replace("@bool_result", "@" + entry)
    assert cudaq.run(QuakeArtifact(quake), shots_count=2) == [True] * 2
