/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/CudaqFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MD5.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

// Strings that are longer than this length will be hashed to MD5 names to avoid
// unnecessarily long symbol names. (This is a hidden command line option, so
// that hashing issues can be easily worked around.)
namespace {
struct CLOptions {
  llvm::cl::opt<std::size_t> nameLengthHashSize{
      "length-to-hash-string-literal",
      llvm::cl::desc("string literals that exceed this length will use a hash "
                     "value as their symbol name"),
      llvm::cl::init(32)};
};
} // namespace

static llvm::ManagedStatic<CLOptions> clOptions;

void cudaq::opt::builder::registerCUDAQBuilderCLOptions() {
  // Initialize the command-line options lazily.
  *clOptions;
}

static constexpr std::size_t DefaultPrerequisiteSize = 4;

// Each record in the intrinsics table has this format.
struct IntrinsicCode {
  StringRef name;                             // The name of the intrinsic.
  StringRef preReqs[DefaultPrerequisiteSize]; // Other intrinsics this one
                                              // depends upon.
  StringRef code; // The MLIR code that declares/defines the intrinsic.
};

inline bool operator<(const IntrinsicCode &icode, StringRef name) {
  return icode.name < name;
}

inline bool operator<(const IntrinsicCode &icode, const IntrinsicCode &jcode) {
  return icode.name < jcode.name;
}

/// Table of intrinsics:
/// This table contains CUDA-Q MLIR code for our inlined intrinsics as
/// well as prototypes for LLVM intrinsics and C library calls that are used by
/// the compiler. The table should be kept in sorted order.
static constexpr IntrinsicCode intrinsicTable[] = {
    // These following pauli_word helper functions are only available on the
    // host-side. They ought not be called in kernel code.
    {cudaq::runtime::getPauliWordData, {}, R"#(
  func.func private @_ZNK5cudaq10pauli_word11_nvqpp_dataEv(%pw : !cc.ptr<i8>) -> !cc.ptr<i8>
)#"},
    {cudaq::runtime::getPauliWordSize,
     {cudaq::runtime::getPauliWordData, cudaq::runtime::bindingGetStringData,
      cudaq::runtime::bindingGetStringSize},
     R"#(
  func.func private @_ZNK5cudaq10pauli_word11_nvqpp_sizeEv(%pw : !cc.ptr<i8>) -> i64
)#"},

    {cudaq::runtime::deviceCodeHolderAdd, {}, R"#(
  llvm.func @__cudaq_deviceCodeHolderAdd(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
)#"},

    {cudaq::runtime::getLinkableKernelKey, {}, R"#(
  func.func private @__cudaq_getLinkableKernelKey(!cc.ptr<i8>) -> i64
)#"},

    {cudaq::runtime::registerLinkableKernel, {}, R"#(
  func.func private @__cudaq_registerLinkableKernel(!cc.ptr<i8>, !cc.ptr<i8>, !cc.ptr<i8>) -> ()
)#"},
    {cudaq::runtime::registerRunnableKernel, {}, R"#(
  func.func private @__cudaq_registerRunnableKernel(!cc.ptr<i8>, !cc.ptr<i8>) -> ()
)#"},

    // Initialize a (preallocated) buffer (the first parameter) with i64 values
    // on the semi-open range `[0..n)` where `n` is the second parameter.
    {cudaq::setCudaqRangeVector, {}, R"#(
  func.func private @__nvqpp_CudaqRangeInit(%arg0: !cc.ptr<!cc.array<i64 x ?>>, %arg1: i64) -> !cc.stdvec<i64> {
    %0 = arith.constant 0 : i64
    %1 = cc.loop while ((%i = %0) -> i64) {
      %w1 = arith.cmpi ult, %i, %arg1 : i64
      cc.condition %w1 (%i : i64)
    } do {
      ^bb1(%i: i64):
        %d1 = cc.compute_ptr %arg0[%i] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
        cc.store %i, %d1 : !cc.ptr<i64>
        cc.continue %i : i64
    } step {
      ^bb1(%i: i64):
        %one = arith.constant 1 : i64
        %s1 = arith.addi %i, %one : i64
        cc.continue %s1 : i64
    } {invariant}
    %2 = cc.stdvec_init %arg0, %arg1 : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.stdvec<i64>
    return %2 : !cc.stdvec<i64>
  }
)#"},

    // Compute and initialize a vector from a semi-open triple style notation.
    // The vector returned will contain the ordered set defined by the triple.
    // That set is specifically `{ i, i+s, i+2*s, ... i+(n-1)*s }` where `i` is
    // the initial value `%arg1`, `s` is the step, `%arg3`, and the value
    // `i+(n-1)*s` is strictly in the interval `[%arg1 .. %arg2)` or `(%arg2 ..
    // %arg1]` depending on whether `%arg3` is positive or negative. Invalid
    // triples, such as the step being zero or the lower and upper bounds being
    // transposed will return a vector of length 0 (an empty set). Note that all
    // three parameters are assumed to be signed values, which is required to
    // have a decrementing loop.
    {cudaq::setCudaqRangeVectorTriple, {cudaq::getCudaqSizeFromTriple}, R"#(
  func.func private @__nvqpp_CudaqRangeInitTriple(%arg0: !cc.ptr<!cc.array<i64 x ?>>, %arg1: i64, %arg2: i64, %arg3: i64) -> !cc.stdvec<i64> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = call @__nvqpp_CudaqSizeFromTriple(%arg1, %arg2, %arg3) : (i64, i64, i64) -> i64
    %1:2 = cc.loop while ((%arg4 = %c0_i64, %arg5 = %arg1) -> (i64, i64)) {
      %3 = arith.cmpi ult, %arg4, %0 : i64
      cc.condition %3(%arg4, %arg5 : i64, i64)
    } do {
    ^bb0(%arg4: i64, %arg5: i64):
      %3 = cc.compute_ptr %arg0[%arg4] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
      cc.store %arg5, %3 : !cc.ptr<i64>
      cc.continue %arg4, %arg5 : i64, i64
    } step {
    ^bb0(%arg4: i64, %arg5: i64):
      %3 = arith.addi %arg4, %c1_i64 : i64
      %4 = arith.addi %arg5, %arg3 : i64
      cc.continue %3, %4 : i64, i64
    } {invariant}
    %2 = cc.stdvec_init %arg0, %0 : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.stdvec<i64>
    return %2 : !cc.stdvec<i64>
  }
)#"},

    // Compute the total number of iterations, which is the value `n`, from a
    // semi-open triple style notation. The set defined by the triple is `{ i,
    // i+s, i+2*s, ... i+(n-1)*s }` where `i` is the initial value `%start`, `s`
    // is the step, `%step`, and the value `i+(n-1)*s` is strictly in the
    // interval `[start .. stop)` or `(stop .. start]` depending on whether step
    // is positive or negative. Invalid triples, such as the step being zero or
    // the lower and upper bounds being transposed will return a value of 0.
    // Note that all three parameters are assumed to be signed values, which is
    // required to have a decrementing loop.
    {cudaq::getCudaqSizeFromTriple, {}, R"#(
  func.func private @__nvqpp_CudaqSizeFromTriple(%start: i64, %stop: i64, %step: i64) -> i64 {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %n1 = arith.constant -1 : i64
    %c1 = arith.cmpi eq, %step, %0 : i64
    cf.cond_br %c1, ^b1, ^exit(%0 : i64)
   ^b1:
    %c2 = arith.cmpi sgt, %step, %0 : i64
    %adjust = arith.select %c2, %1, %n1 : i64
    %2 = arith.subi %stop, %adjust : i64
    %3 = arith.subi %2, %start : i64
    %4 = arith.addi %3, %step : i64
    %5 = arith.divsi %4, %step : i64
    %c3 = arith.cmpi sgt, %5, %0 : i64
    cf.cond_br %c3, ^exit(%5 : i64), ^exit(%0 : i64)
   ^exit(%rv : i64):
    return %rv : i64
  }
)#"},

    // __nvqpp__cudaq_em_allocate
    {cudaq::opt::CudaqEMAllocate,
     {},
     "func.func private @__nvqpp__cudaq_em_allocate() -> i64"},

    // __nvqpp__cudaq_em_allocate_veq
    {cudaq::opt::CudaqEMAllocateVeq, {}, R"#(
  func.func private @__nvqpp__cudaq_em_allocate_veq(%span : !cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>, %size : i64) {
    %buffptr = cc.compute_ptr %span[0] : (!cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>) -> !cc.ptr<!cc.ptr<!cc.array<i64 x ?>>>
    %buffer = cc.load %buffptr : !cc.ptr<!cc.ptr<!cc.array<i64 x ?>>>
    %0 = arith.constant 0 : i64
    %1 = cc.loop while ((%arg0 = %0) -> (i64)) {
      %cond = arith.cmpi slt, %arg0, %size : i64
      cc.condition %cond (%arg0 : i64)
    } do {
     ^bb0(%arg0 : i64):
      %2 = func.call @__nvqpp__cudaq_em_allocate() : () -> i64
      %3 = cc.compute_ptr %buffer[%arg0] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
      cc.store %2, %3 : !cc.ptr<i64>
      cc.continue %arg0 : i64
    } step {
     ^bb0(%arg0 : i64):
      %4 = arith.constant 1 : i64
      %5 = arith.addi %arg0, %4 : i64
      cc.continue %5 : i64
    } {invariant}
    return
  }
)#"},
    // __nvqpp__cudaq_em_apply
    {cudaq::opt::CudaqEMApply, {}, R"#(
  func.func private @__nvqpp__cudaq_em_apply(!cc.ptr<i8>, i64, !cc.ptr<!cc.array<f64 x ?>>, !cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>, !cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>, i1)
)#"},
    // __nvqpp__cudaq_em_concatSpan
    {cudaq::opt::CudaqEMConcatSpan, {cudaq::llvmMemCopyIntrinsic}, R"#(
  func.func private @__nvqpp__cudaq_em_concatSpan(%dest : !cc.ptr<i64>, %from : !cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>, %length : i64) {
    %ptrptr = cc.compute_ptr %from[0] : (!cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>) -> !cc.ptr<!cc.ptr<!cc.array<i64 x ?>>>
    %src = cc.load %ptrptr : !cc.ptr<!cc.ptr<!cc.array<i64 x ?>>>
    %eight = arith.constant 8 : i64
    %len = arith.muli %length, %eight : i64
    %false = arith.constant false
    %to0 = cc.cast %dest : (!cc.ptr<i64>) -> !cc.ptr<i8>
    %from0 = cc.cast %src : (!cc.ptr<!cc.array<i64 x ?>>) -> !cc.ptr<i8>
    call @llvm.memcpy.p0i8.p0i8.i64(%to0, %from0, %len, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    return
  }
)#"},
    // __nvqpp__cudaq_em_measure
    {cudaq::opt::CudaqEMMeasure, {}, R"#(
  func.func private @__nvqpp__cudaq_em_measure(!cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>, !cc.ptr<i8>) -> i32
)#"},
    // __nvqpp__cudaq_em_reset
    {cudaq::opt::CudaqEMReset, {}, R"#(
  func.func private @__nvqpp__cudaq_em_reset(!cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>)
)#"},
    // __nvqpp__cudaq_em_return
    {cudaq::opt::CudaqEMReturn, {}, R"#(
  func.func private @__nvqpp__cudaq_em_return(!cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>)
)#"},
    // __nvqpp__cudaq_em_writeToSpan
    {cudaq::opt::CudaqEMWriteToSpan, {}, R"#(
  func.func private @__nvqpp__cudaq_em_writeToSpan(%span : !cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>, %ptr : !cc.ptr<!cc.array<i64 x ?>>, %size : i64) {
    %buffptr = cc.compute_ptr %span[0] : (!cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>) -> !cc.ptr<!cc.ptr<!cc.array<i64 x ?>>>
    cc.store %ptr, %buffptr : !cc.ptr<!cc.ptr<!cc.array<i64 x ?>>>
    %szptr = cc.compute_ptr %span[1] : (!cc.ptr<!cc.struct<".qubit_span" {!cc.ptr<!cc.array<i64 x ?>>, i64}>>) -> !cc.ptr<i64>
    cc.store %size, %szptr : !cc.ptr<i64>
    return
  }
)#"},

    {cudaq::runtime::callDeviceCallback, {}, R"#(
  func.func private @__nvqpp__device_callback_run(i64, !cc.ptr<i8>, !cc.ptr<i8>, !cc.ptr<i8>, i64, i64, i64, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
)#"},

    {cudaq::runtime::extractDevPtr, {}, R"#(
  func.func private @__nvqpp__device_extract_device_ptr(!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>) -> !cc.ptr<i8>
)#"},
    {cudaq::runtime::cleanupArrays, {}, R"#(
  func.func private @__nvqpp_cleanup_arrays() -> ()
)#"},
    {"__nvqpp_createDynamicResult",
     /* arguments:
          arg0: original buffer ptr
          arg1: original buffer size
          arg2: ptr to span of the return data: {ptr, bytes}
          arg3: offset to result slot in buffer */
     {cudaq::llvmMemCopyIntrinsic, "malloc"},
     R"#(
  func.func private @__nvqpp_createDynamicResult(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: !cc.ptr<!cc.struct<{!cc.ptr<i8>, i64}>>, %arg3: i64) -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %0 = cc.compute_ptr %arg2[1] : (!cc.ptr<!cc.struct<{!cc.ptr<i8>, i64}>>) -> !cc.ptr<i64>
    %1 = cc.load %0 : !cc.ptr<i64>
    %2 = arith.addi %arg1, %1 : i64
    %3 = call @malloc(%2) : (i64) -> !cc.ptr<i8>
    %10 = cc.cast %3 : (!cc.ptr<i8>) -> !cc.ptr<!cc.array<i8 x ?>>
    %false = arith.constant false
    call @llvm.memcpy.p0i8.p0i8.i64(%3, %arg0, %arg1, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    %4 = cc.compute_ptr %arg2[0] : (!cc.ptr<!cc.struct<{!cc.ptr<i8>, i64}>>) -> !cc.ptr<!cc.ptr<i8>>
    %5 = cc.load %4 : !cc.ptr<!cc.ptr<i8>>
    %6 = cc.compute_ptr %10[%arg1] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
    call @llvm.memcpy.p0i8.p0i8.i64(%6, %5, %1, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    %7 = cc.undef !cc.struct<{!cc.ptr<i8>, i64}>
    %8 = cc.insert_value %7[0], %3 : (!cc.struct<{!cc.ptr<i8>, i64}>, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %9 = cc.insert_value %8[1], %2 : (!cc.struct<{!cc.ptr<i8>, i64}>, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %11 = cc.compute_ptr %10[%arg3] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
    %12 = cc.cast %11 : (!cc.ptr<i8>) -> !cc.ptr<!cc.ptr<i8>>
    cc.store %6, %12 : !cc.ptr<!cc.ptr<i8>>
    return %9 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
)#"},

    {cudaq::cudaqConvertToInteger, {}, R"#(
  func.func private @__nvqpp_cudaqConvertToInteger(%arg : !cc.stdvec<i1>) -> i64 {
    %size = cc.stdvec_size %arg : (!cc.stdvec<i1>) -> i64
    %data = cc.stdvec_data %arg : (!cc.stdvec<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %res:2 = cc.loop while ((%i = %zero, %v = %zero) -> (i64, i64)) {
      %cmp = arith.cmpi ult, %i, %size : i64
      cc.condition %cmp(%i, %v : i64, i64)
    } do {
      ^bb1(%j : i64, %v : i64):
        %2 = arith.shli %one, %j : i64
        %3 = cc.compute_ptr %data[%j] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
        %4 = cc.load %3 : !cc.ptr<i8>
        %5 = cc.cast unsigned %4 : (i8) -> i64
        %6 = arith.subi %zero, %5 : i64
        %7 = arith.xori %6, %v : i64
        %8 = arith.andi %7, %2 : i64
        %nv = arith.xori %v, %8: i64
        cc.continue %j, %nv : i64, i64
    } step {
      ^bb1(%j : i64, %v : i64):
        %nj = arith.addi %j, %one : i64
        cc.continue %nj, %v : i64, i64
    }
    return %res#1 : i64
  }
)#"},

    {cudaq::createCudaqStateFromDataComplexF32, {}, R"#(
  func.func private @__nvqpp_cudaq_state_createFromData_complex_f32(%p : !cc.ptr<i8>, %s : i64) -> !cc.ptr<!quake.state>
)#"},
    {cudaq::createCudaqStateFromDataComplexF64, {}, R"#(
  func.func private @__nvqpp_cudaq_state_createFromData_complex_f64(%p : !cc.ptr<i8>, %s : i64) -> !cc.ptr<!quake.state>
)#"},
    {cudaq::createCudaqStateFromDataF32, {}, R"#(
  func.func private @__nvqpp_cudaq_state_createFromData_f32(%p : !cc.ptr<i8>, %s : i64) -> !cc.ptr<!quake.state>
)#"},
    {cudaq::createCudaqStateFromDataF64, {}, R"#(
  func.func private @__nvqpp_cudaq_state_createFromData_f64(%p : !cc.ptr<i8>, %s : i64) -> !cc.ptr<!quake.state>
)#"},

    {cudaq::deleteCudaqState, {}, R"#(
  func.func private @__nvqpp_cudaq_state_delete(%p : !cc.ptr<!quake.state>) -> ()
)#"},

    {cudaq::getNumQubitsFromCudaqState, {}, R"#(
  func.func private @__nvqpp_cudaq_state_numberOfQubits(%p : !cc.ptr<!quake.state>) -> i64
)#"},

    {cudaq::runtime::bindingDeconstructString,
     {},
     "func.func private @__nvqpp_deconstructString(!cc.ptr<i8>)"},

    // Note that the malloc* hooks use RVO. The C++ bridge must perform this
    // optimization on its side as well. Specifically, the device_ptr return
    // value *must* be converted to the first argument.
    {"__nvqpp_driver_free", {}, R"#(
  func.func private @__nvqpp_driver_free(!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>)
)#"},
    {"__nvqpp_driver_malloc", {}, R"#(
  func.func private @__nvqpp_driver_malloc(!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>, i64)
)#"},
    {"__nvqpp_driver_malloc_set", {}, R"#(
  func.func private @__nvqpp_driver_malloc_set(!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>, !cc.ptr<i8>, i64, i64)
)#"},
    {"__nvqpp_driver_memcpy_receive", {}, R"#(
  func.func private @__nvqpp_driver_memcpy_receive(!cc.ptr<none>, !cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>)
)#"},
    {"__nvqpp_driver_memcpy_send", {"__nvqpp_driver_memcpy_receive"}, R"#(
  func.func private @__nvqpp_driver_memcpy_send(!cc.ptr<!cc.struct<"device_ptr" {i64, i64, i64}>>, !cc.ptr<none>)
)#"},

    {"__nvqpp_getStateVectorData_fp32", {}, R"#(
  func.func private @__nvqpp_getStateVectorData_fp32(%p : i64, %o : i64) -> !cc.ptr<complex<f32>>
)#"},
    {"__nvqpp_getStateVectorData_fp64", {}, R"#(
  func.func private @__nvqpp_getStateVectorData_fp64(%p : i64, %o : i64) -> !cc.ptr<complex<f64>>
)#"},
    {"__nvqpp_getStateVectorLength_fp32", {}, R"#(
  func.func private @__nvqpp_getStateVectorLength_fp32(%p : i64, %o : i64) -> i64
)#"},
    {"__nvqpp_getStateVectorLength_fp64", {}, R"#(
  func.func private @__nvqpp_getStateVectorLength_fp64(%p : i64, %o : i64) -> i64
)#"},

    // Quasi-portable entry points for use with non-C++ front ends (Python).
    {cudaq::runtime::bindingGetStringData, {}, R"#(
  func.func private @__nvqpp_getStringData(%p: !cc.ptr<i8>) -> !cc.ptr<i8>
)#"},
    {cudaq::runtime::bindingGetStringSize,
     {},
     "func.func private @__nvqpp_getStringSize(%p: !cc.ptr<i8>) -> i64"},

    {cudaq::runtime::bindingInitializeString, {}, R"#(
  func.func private @__nvqpp_initializeStringFromSpan(!cc.ptr<i8>, !cc.ptr<i8>, i64)
)#"},

    // __nvqpp_initializer_list_to_vector_bool
    {cudaq::stdvecBoolCtorFromInitList, {}, R"#(
  func.func private @__nvqpp_initializer_list_to_vector_bool(!cc.ptr<none>, !cc.ptr<none>, i64) -> ())#"},

    // This helper function copies a buffer off the stack to the heap. This is
    // required when the data on the stack is about to go out of scope but is
    // still live.
    {"__nvqpp_vectorCopyCtor", {cudaq::llvmMemCopyIntrinsic, "malloc"}, R"#(
  func.func private @__nvqpp_vectorCopyCtor(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: i64) -> !cc.ptr<i8> {
    %size = arith.muli %arg1, %arg2 : i64
    %0 = call @malloc(%size) : (i64) -> !cc.ptr<i8>
    %false = arith.constant false
    call @llvm.memcpy.p0i8.p0i8.i64(%0, %arg0, %size, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    return %0 : !cc.ptr<i8>
  }
)#"},

    // This helper function copies a buffer that is in the heap to a buffer on
    // the stack. Both buffers must already exist. This helper matches
    // __nvqpp_vectorCopyCtor and eliminates memory leaks.
    {"__nvqpp_vectorCopyToStack", {cudaq::llvmMemCopyIntrinsic, "free"}, R"#(
  func.func private @__nvqpp_vectorCopyToStack(%to: !cc.ptr<i8>, %from: !cc.ptr<i8>, %size: i64) {
    %false = arith.constant false
    call @llvm.memcpy.p0i8.p0i8.i64(%to, %from, %size, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    call @free(%from) : (!cc.ptr<i8>) -> ()
    return
  })#"},

    // __nvqpp_vector_bool_free_temporary_lists
    {cudaq::stdvecBoolFreeTemporaryLists, {}, R"#(
  func.func private @__nvqpp_vector_bool_free_temporary_initlists(!cc.ptr<i8>) -> ()
)#"},

    // __nvqpp_vector_bool_to_initializer_list
    {cudaq::stdvecBoolUnpackToInitList, {}, R"#(
  func.func private @__nvqpp_vector_bool_to_initializer_list(!cc.ptr<!cc.struct<{!cc.ptr<i1>, !cc.ptr<i1>, !cc.ptr<i1>}>>, !cc.ptr<!cc.struct<{!cc.ptr<i1>, !cc.array<i8 x 32>}>>, !cc.ptr<!cc.ptr<i8>>) -> ()
)#"},

    {"__nvqpp_zeroDynamicResult", {}, R"#(
  func.func private @__nvqpp_zeroDynamicResult() -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cc.cast %c0_i64 : (i64) -> !cc.ptr<i8>
    %1 = cc.undef !cc.struct<{!cc.ptr<i8>, i64}>
    %2 = cc.insert_value %1[0], %0 : (!cc.struct<{!cc.ptr<i8>, i64}>, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %3 = cc.insert_value %2[1], %c0_i64 : (!cc.struct<{!cc.ptr<i8>, i64}>, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %3 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
)#"},

    {cudaq::opt::QISTrap, {}, R"#(
  func.func private @__quantum__qis__trap(i64)
)#"},

    // The QIR defined output logging functions.
    {cudaq::opt::QIRArrayRecordOutput,
     // note: uses all DefaultPrerequisiteSize slots
     {cudaq::opt::QIRBoolRecordOutput, cudaq::opt::QIRDoubleRecordOutput,
      cudaq::opt::QIRIntegerRecordOutput, cudaq::opt::QIRTupleRecordOutput},
     R"#(
  func.func private @__quantum__rt__array_record_output(i64, !cc.ptr<i8>)
)#"},
    {cudaq::opt::QIRBoolRecordOutput, {}, R"#(
  func.func private @__quantum__rt__bool_record_output(i1 {llvm.zeroext}, !cc.ptr<i8>)
)#"},
    {cudaq::opt::QIRDoubleRecordOutput, {}, R"#(
  func.func private @__quantum__rt__double_record_output(f64, !cc.ptr<i8>)
)#"},
    {cudaq::opt::QIRIntegerRecordOutput, {}, R"#(
  func.func private @__quantum__rt__int_record_output(i64, !cc.ptr<i8>)
)#"},
    {cudaq::opt::QIRTupleRecordOutput, {}, R"#(
  func.func private @__quantum__rt__tuple_record_output(i64, !cc.ptr<i8>)
)#"},

    // altLaunchKernel(kernelName, thunk, commBuffer, buffSize, resultOffset)
    {cudaq::runtime::launchKernelFuncName, {}, R"#(
  func.func private @altLaunchKernel(!cc.ptr<i8>, !cc.ptr<i8>, !cc.ptr<i8>, i64, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
)#"},

    {cudaq::runtime::CudaqRegisterArgsCreator, {}, R"#(
  func.func private @cudaqRegisterArgsCreator(!cc.ptr<i8>, !cc.ptr<i8>) -> ()
)#"},

    {cudaq::runtime::CudaqRegisterCallbackName, {}, R"#(
  func.func private @cudaqRegisterCallbackName(!cc.ptr<i8>, !cc.ptr<i8>, !cc.ptr<i8>) -> ()
)#"},

    {cudaq::runtime::CudaqRegisterKernelName,
     {cudaq::runtime::CudaqRegisterArgsCreator,
      cudaq::runtime::CudaqRegisterLambdaName,
      cudaq::runtime::registerLinkableKernel,
      cudaq::runtime::getLinkableKernelKey},
     "func.func private @cudaqRegisterKernelName(!cc.ptr<i8>) -> ()"},

    {cudaq::runtime::CudaqRegisterLambdaName, {}, R"#(
  llvm.func @cudaqRegisterLambdaName(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
)#"},

    {"free", {}, "func.func private @free(!cc.ptr<i8>) -> ()"},

    // hybridLaunchKernel(kernelName, thunk, commBuffer, buffSize,
    //                    resultOffset, vectorArgPtrs)
    {cudaq::runtime::launchKernelHybridFuncName, {}, R"#(
  func.func private @hybridLaunchKernel(!cc.ptr<i8>, !cc.ptr<i8>, !cc.ptr<i8>, i64, i64, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
)#"},

    // llvm.memcpy.p0i8.p0i8.i64
    {cudaq::llvmMemCopyIntrinsic, {}, R"#(
  func.func private @llvm.memcpy.p0i8.p0i8.i64(!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
)#"},

    {cudaq::llvmMemSetIntrinsic, // llvm.memset.p0i8.i64
     {},
     R"#(
  func.func private @llvm.memset.p0i8.i64(!cc.ptr<i8>, i8, i64, i1) -> ())#"},

    // NB: load llvmStackSave to get both.
    {cudaq::llvmStackRestore,
     {},
     "func.func private @llvm.stackrestore(!cc.ptr<i8>)"},
    {cudaq::llvmStackSave,
     {cudaq::llvmStackRestore},
     "func.func private @llvm.stacksave() -> !cc.ptr<i8>"},

    {"malloc", {}, "func.func private @malloc(i64) -> !cc.ptr<i8>"},

    // Declarations of QIR functions used by codegen that are common to all
    // subtargets (full, base profle, or adaptive profile).
    // These include qubit allocation and management, control variants of the
    // gates, some one offs, and control form invocation helper routines.
    {"qir_common", {cudaq::opt::QISTrap}, R"#(
  func.func private @__quantum__rt__qubit_allocate() -> !qir_qubit
  func.func private @__quantum__rt__qubit_allocate_array(i64) -> !qir_array
  func.func private @__quantum__rt__qubit_allocate_array_with_state_fp64(i64, !cc.ptr<f64>) -> !qir_array
  func.func private @__quantum__rt__qubit_allocate_array_with_state_fp32(i64, !cc.ptr<f32>) -> !qir_array
  func.func private @__quantum__rt__qubit_allocate_array_with_state_complex64(i64, !cc.ptr<complex<f64>>) -> !qir_array
  func.func private @__quantum__rt__qubit_allocate_array_with_state_complex32(i64, !cc.ptr<complex<f32>>) -> !qir_array
  func.func private @__quantum__rt__qubit_allocate_array_with_state_ptr(!cc.ptr<none>) -> !qir_array
  func.func private @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64, !cc.ptr<!quake.state>) -> !qir_array

  func.func private @__quantum__rt__qubit_release_array(!qir_array)
  func.func private @__quantum__rt__qubit_release(!qir_qubit)

  func.func private @__quantum__rt__array_create_1d(i32, i64) -> !qir_array
  func.func private @__quantum__rt__array_concatenate(!qir_array, !qir_array) -> !qir_array
  func.func private @__quantum__rt__array_get_size_1d(!qir_array) -> i64
  func.func private @__quantum__rt__array_slice(!qir_array, i32, i64, i64, i64) -> !qir_array
  func.func private @__quantum__rt__array_get_element_ptr_1d(!qir_array, i64) -> !cc.ptr<!qir_qubit>

  func.func private @__quantum__qis__h__ctl(!qir_array, !qir_qubit)
  func.func private @__quantum__qis__x__ctl(!qir_array, !qir_qubit)
  func.func private @__quantum__qis__y__ctl(!qir_array, !qir_qubit)
  func.func private @__quantum__qis__z__ctl(!qir_array, !qir_qubit)
  func.func private @__quantum__qis__s__ctl(!qir_array, !qir_qubit)
  func.func private @__quantum__qis__t__ctl(!qir_array, !qir_qubit)
  func.func private @__quantum__qis__sdg__ctl(!qir_array, !qir_qubit)
  func.func private @__quantum__qis__tdg__ctl(!qir_array, !qir_qubit)
  func.func private @__quantum__qis__u3__ctl(f64, f64, f64, !qir_array, !qir_qubit)
  func.func private @__quantum__qis__swap__ctl(!qir_array, !qir_qubit, !qir_qubit)
  func.func private @__quantum__qis__rx__ctl(f64, !qir_array, !qir_qubit)
  func.func private @__quantum__qis__ry__ctl(f64, !qir_array, !qir_qubit)
  func.func private @__quantum__qis__rz__ctl(f64, !qir_array, !qir_qubit)
  func.func private @__quantum__qis__r1__ctl(f64, !qir_array, !qir_qubit)

  func.func private @__quantum__qis__exp_pauli__ctl(f64, !qir_array, !qir_array, !qir_charptr)
  func.func private @__quantum__qis__custom_unitary(!cc.ptr<complex<f64>>, !qir_array, !qir_array, !qir_charptr)
  func.func private @__quantum__qis__custom_unitary__adj(!cc.ptr<complex<f64>>, !qir_array, !qir_array, !qir_charptr)

  func.func private @__quantum__qis__convert_array_to_stdvector(!qir_array) -> !qir_array
  func.func private @__quantum__qis__free_converted_stdvector(!qir_array)

  llvm.func @generalizedInvokeWithRotationsControlsTargets(i64, i64, i64, i64, !qir_llvmptr, ...) attributes {sym_visibility = "private"}
  llvm.func @__quantum__qis__apply_kraus_channel_generalized(i64, i64, i64, i64, i64, ...) attributes {sym_visibility = "private"}
)#"},

    // Declarations for base and adaptive profile QIR functions used by codegen.
    // These include gates, adjoint gates, one offs, and dealing with
    // measurement results.
    {"qir_common_profile", {"qir_common"}, R"#(
  func.func private @__quantum__qis__h__body(!qir_qubit)
  func.func private @__quantum__qis__x__body(!qir_qubit)
  func.func private @__quantum__qis__y__body(!qir_qubit)
  func.func private @__quantum__qis__z__body(!qir_qubit)
  func.func private @__quantum__qis__s__body(!qir_qubit)
  func.func private @__quantum__qis__t__body(!qir_qubit)
  func.func private @__quantum__qis__s__adj(!qir_qubit)
  func.func private @__quantum__qis__t__adj(!qir_qubit)
  func.func private @__quantum__qis__u3__body(f64, f64, f64, !qir_qubit)
  func.func private @__quantum__qis__reset__body(!qir_qubit)
  func.func private @__quantum__qis__mz__body(!qir_qubit, !qir_result) attributes {passthrough = ["irreversible"]} 
  func.func private @__quantum__qis__swap__body(!qir_qubit, !qir_qubit)
  func.func private @__quantum__qis__rx__body(f64, !qir_qubit)
  func.func private @__quantum__qis__phased_rx__body(f64, f64, !qir_qubit)
  func.func private @__quantum__qis__ry__body(f64, !qir_qubit)
  func.func private @__quantum__qis__rz__body(f64, !qir_qubit)
  func.func private @__quantum__qis__r1__body(f64, !qir_qubit)
  func.func private @__quantum__qis__exp_pauli__body(f64, !qir_array, !qir_charptr)

  func.func private @__quantum__rt__result_record_output(!qir_result, !qir_charptr)
  func.func private @__quantum__qis__cnot__body(!qir_qubit, !qir_qubit)
  func.func private @__quantum__qis__cz__body(!qir_qubit, !qir_qubit)
  func.func private @__quantum__rt__read_result(!qir_result) -> i1
  func.func private @__quantum__qis__read_result__body(!qir_result) -> i1
)#"},

    // Declarations of all full QIR functions used by codegen.
    // These include gates (sans the "__body" suffix) and measurements.
    {"qir_full", {"qir_common"}, R"#(
  func.func private @__quantum__qis__h(!qir_qubit)
  func.func private @__quantum__qis__x(!qir_qubit)
  func.func private @__quantum__qis__y(!qir_qubit)
  func.func private @__quantum__qis__z(!qir_qubit)
  func.func private @__quantum__qis__s(!qir_qubit)
  func.func private @__quantum__qis__t(!qir_qubit)
  func.func private @__quantum__qis__s__adj(!qir_qubit)
  func.func private @__quantum__qis__t__adj(!qir_qubit)
  func.func private @__quantum__qis__u3(f64, f64, f64, !qir_qubit)
  func.func private @__quantum__qis__reset(!qir_qubit)
  func.func private @__quantum__qis__mz(!qir_qubit) -> !qir_result
  func.func private @__quantum__qis__mz__to__register(!qir_qubit, !qir_charptr) -> !qir_result
  func.func private @__quantum__qis__swap(!qir_qubit, !qir_qubit)
  func.func private @__quantum__qis__rx(f64, !qir_qubit)
  func.func private @__quantum__qis__phased_rx(f64, f64, !qir_qubit)
  func.func private @__quantum__qis__ry(f64, !qir_qubit)
  func.func private @__quantum__qis__rz(f64, !qir_qubit)
  func.func private @__quantum__qis__r1(f64, !qir_qubit)
  func.func private @__quantum__qis__exp_pauli(f64, !qir_array, !qir_charptr)
)#"},

    // Choose one of the two QIR typing conventions. Opaque pointers are the
    // current LLVM standard. Opaque struct is from an obsolete LLVM version,
    // but used by the QIR specification.

    // Use opaque pointers (LLVM's `ptr` type). The type of the referent is
    // always implicit and unambiguous from its usage. At the moment, this is
    // using i8* instead of ptr, since the latter requires some other changes.
    {"qir_opaque_pointer", {}, R"#(
  !qir_array = !cc.ptr<none>
  !qir_qubit = !cc.ptr<none>
  !qir_result = !cc.ptr<none>
  !qir_charptr = !cc.ptr<none>
  !qir_llvmptr = !llvm.ptr<i8>
)#"},
    // Use the obsolete LLVM opaque struct type.
    {"qir_opaque_struct", {}, R"#(
  !qir_array = !cc.ptr<!llvm.struct<"Array", opaque>>
  !qir_qubit = !cc.ptr<!llvm.struct<"Qubit", opaque>>
  !qir_result = !cc.ptr<!llvm.struct<"Result", opaque>>
  !qir_charptr = !cc.ptr<i8>
  !qir_llvmptr = !llvm.ptr<i8>
)#"},

    // streamlinedLaunchKernel(kernelName, vectorArgPtrs)
    {cudaq::runtime::launchKernelStreamlinedFuncName, {}, R"#(
  func.func private @streamlinedLaunchKernel(!cc.ptr<i8>, !cc.ptr<i8>) -> ()
)#"}};

static constexpr std::size_t intrinsicTableSize =
    sizeof(intrinsicTable) / sizeof(IntrinsicCode);

inline bool intrinsicTableIsSorted() {
  for (std::size_t i = 0; i < intrinsicTableSize - 1; i++)
    if (!(intrinsicTable[i] < intrinsicTable[i + 1]))
      return false;
  return true;
}

namespace cudaq {

IRBuilder::IRBuilder(const OpBuilder &builder)
    : OpBuilder{builder.getContext()} {
  // Sets the insertion point to be the same as \p builder. New operations will
  // be inserted immediately before this insertion point and the insertion
  // points will remain the identical, upto and unless one of the builders
  // changes its insertion pointer.
  auto *block = builder.getBlock();
  auto point = builder.getInsertionPoint();
  setInsertionPoint(block, point);
}

LLVM::GlobalOp IRBuilder::genCStringLiteral(Location loc, ModuleOp module,
                                            llvm::StringRef cstring) {
  auto *ctx = getContext();
  auto cstringTy = opt::factory::getStringType(ctx, cstring.size());
  auto uniqName = "cstr." + hashStringByContent(cstring);
  if (auto stringLit = module.lookupSymbol<LLVM::GlobalOp>(uniqName))
    return stringLit;
  auto stringAttr = getStringAttr(cstring);
  OpBuilder::InsertionGuard guard(*this);
  setInsertionPointToEnd(module.getBody());
  return create<LLVM::GlobalOp>(loc, cstringTy, /*isConstant=*/true,
                                LLVM::Linkage::Private, uniqName, stringAttr,
                                /*alignment=*/0);
}

std::string IRBuilder::hashStringByContent(StringRef sref) {
  // For shorter names just use the string content in hex. (Consider replacing
  // this with a more compact, readable base-64 encoding.)
  if (sref.size() <= clOptions->nameLengthHashSize)
    return llvm::toHex(sref);

  // Use an MD5 hash for long cstrings. This can produce collisions between
  // different strings that hash to the same MD5 name.
  llvm::MD5 hash;
  hash.update(sref);
  llvm::MD5::MD5Result result;
  hash.final(result);
  llvm::SmallString<64> str;
  llvm::MD5::stringifyResult(result, str);
  return str.c_str();
}

LogicalResult IRBuilder::loadIntrinsic(ModuleOp module, StringRef intrinName) {
  // Check if this intrinsic was already loaded.
  if (module.lookupSymbol(intrinName))
    return success();
  assert(intrinsicTableIsSorted() && "intrinsic table must be sorted");
  auto iter = std::lower_bound(&intrinsicTable[0],
                               &intrinsicTable[intrinsicTableSize], intrinName);
  if (iter == &intrinsicTable[intrinsicTableSize]) {
    module.emitError(std::string("intrinsic") + intrinName + " not in table.");
    return failure();
  }
  assert(iter->name == intrinName);
  // First load the prereqs.
  for (std::size_t i = 0; i < DefaultPrerequisiteSize; ++i) {
    if (iter->preReqs[i].empty())
      break;
    if (failed(loadIntrinsic(module, iter->preReqs[i])))
      return failure();
  }
  // Now load the requested code.
  return parseSourceString(
      iter->code, module.getBody(),
      ParserConfig{module.getContext(), /*verifyAfterParse=*/false});
}

StringRef IRBuilder::getIntrinsicText(StringRef intrinName) {
  auto iter = std::lower_bound(&intrinsicTable[0],
                               &intrinsicTable[intrinsicTableSize], intrinName);
  if (iter == &intrinsicTable[intrinsicTableSize])
    return "";
  return iter->code;
}

LogicalResult IRBuilder::loadIntrinsicWithAliases(ModuleOp module,
                                                  StringRef intrinName,
                                                  StringRef prefix) {
  // Check if this intrinsic was already loaded.
  if (module.lookupSymbol(intrinName))
    return success();
  assert(intrinsicTableIsSorted() && "intrinsic table must be sorted");
  auto iter = std::lower_bound(&intrinsicTable[0],
                               &intrinsicTable[intrinsicTableSize], intrinName);
  if (iter == &intrinsicTable[intrinsicTableSize]) {
    module.emitError(std::string("intrinsic") + intrinName + " not in table.");
    return failure();
  }
  assert(iter->name == intrinName);
  // First load the prereqs.
  for (std::size_t i = 0; i < DefaultPrerequisiteSize; ++i) {
    if (iter->preReqs[i].empty())
      break;
    if (failed(loadIntrinsicWithAliases(module, iter->preReqs[i], prefix)))
      return failure();
  }
  // Now load the requested code.
  std::string code = prefix.str() + std::string(iter->code);
  return parseSourceString(
      code, module.getBody(),
      ParserConfig{module.getContext(), /*verifyAfterParse=*/false});
}

template <typename T>
DenseElementsAttr createDenseElementsAttr(const SmallVectorImpl<T> &values,
                                          Type eleTy) {
  auto newValues = ArrayRef<T>(values.data(), values.size());
  auto tensorTy = RankedTensorType::get(values.size(), eleTy);
  return DenseElementsAttr::get(tensorTy, newValues);
}

static DenseElementsAttr
createDenseElementsAttr(const SmallVectorImpl<bool> &values, Type eleTy) {
  SmallVector<std::byte> converted;
  for (auto it = values.begin(); it != values.end(); it++) {
    bool value = *it;
    converted.push_back(std::byte(value));
  }
  auto newValues = ArrayRef<bool>(reinterpret_cast<bool *>(converted.data()),
                                  converted.size());
  auto tensorTy = RankedTensorType::get(converted.size(), eleTy);
  return DenseElementsAttr::get(tensorTy, newValues);
}

static cc::GlobalOp buildVectorOfConstantElements(Location loc, ModuleOp module,
                                                  StringRef name,
                                                  DenseElementsAttr &arrayAttr,
                                                  IRBuilder &builder,
                                                  Type eleTy) {
  if (auto glob = module.lookupSymbol<cc::GlobalOp>(name))
    return glob;
  auto *ctx = builder.getContext();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto globalTy = cc::ArrayType::get(ctx, eleTy, arrayAttr.size());
  auto global =
      builder.create<cudaq::cc::GlobalOp>(loc, globalTy, name, arrayAttr,
                                          /*constant=*/true,
                                          /*external=*/false);
  global.setPrivate();
  return global;
}

template <typename A>
cc::GlobalOp buildVectorOfConstantElements(Location loc, ModuleOp module,
                                           StringRef name,
                                           const SmallVectorImpl<A> &values,
                                           IRBuilder &builder, Type eleTy) {
  auto arrayAttr = createDenseElementsAttr(values, eleTy);
  return buildVectorOfConstantElements(loc, module, name, arrayAttr, builder,
                                       eleTy);
}

cc::GlobalOp IRBuilder::genVectorOfConstants(Location loc, ModuleOp module,
                                             StringRef name,
                                             DenseElementsAttr values,
                                             Type elementType) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       elementType);
}

cc::GlobalOp IRBuilder::genVectorOfConstants(
    Location loc, ModuleOp module, StringRef name,
    const SmallVectorImpl<std::complex<double>> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       ComplexType::get(getF64Type()));
}

cc::GlobalOp IRBuilder::genVectorOfConstants(
    Location loc, ModuleOp module, StringRef name,
    const SmallVectorImpl<std::complex<float>> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       ComplexType::get(getF32Type()));
}

cc::GlobalOp
IRBuilder::genVectorOfConstants(Location loc, ModuleOp module, StringRef name,
                                const SmallVectorImpl<double> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       getF64Type());
}

cc::GlobalOp
IRBuilder::genVectorOfConstants(Location loc, ModuleOp module, StringRef name,
                                const SmallVectorImpl<float> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       getF32Type());
}

cc::GlobalOp
IRBuilder::genVectorOfConstants(Location loc, ModuleOp module, StringRef name,
                                const SmallVectorImpl<std::int64_t> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       getI64Type());
}

cc::GlobalOp
IRBuilder::genVectorOfConstants(Location loc, ModuleOp module, StringRef name,
                                const SmallVectorImpl<std::int32_t> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       getI32Type());
}

cc::GlobalOp
IRBuilder::genVectorOfConstants(Location loc, ModuleOp module, StringRef name,
                                const SmallVectorImpl<std::int16_t> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       getI16Type());
}

cc::GlobalOp
IRBuilder::genVectorOfConstants(Location loc, ModuleOp module, StringRef name,
                                const SmallVectorImpl<std::int8_t> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       getI8Type());
}

cc::GlobalOp
IRBuilder::genVectorOfConstants(Location loc, ModuleOp module, StringRef name,
                                const SmallVectorImpl<bool> &values) {
  return buildVectorOfConstantElements(loc, module, name, values, *this,
                                       getI1Type());
}

Value IRBuilder::getByteSizeOfType(Location loc, Type ty) {
  return cc::getByteSizeOfType(*this, loc, ty, /*useSizeOf=*/true);
}

} // namespace cudaq
