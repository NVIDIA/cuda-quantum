/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/CUDAQBuilder.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MD5.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

// Strings that are longer than this length will be hashed to MD5 names to avoid
// unnecessarily long symbol names. (This is a hidden command line option, so
// that hashing issues can be easily worked around.)
static llvm::cl::opt<std::size_t> nameLengthHashSize(
    "length-to-hash-string-literal",
    llvm::cl::desc("string literals that exceed this length will use a hash "
                   "value as their symbol name"),
    llvm::cl::init(32));

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
/// This table contains CUDA Quantum MLIR code for our inlined intrinsics as
/// well as prototypes for LLVM intrinsics and C library calls that are used by
/// the compiler. The table should be kept in sorted order.
static constexpr IntrinsicCode intrinsicTable[] = {
    {"__nvqpp_createDynamicResult",
     {llvmMemCopyIntrinsic, "malloc"},
     R"#(
  func.func private @__nvqpp_createDynamicResult(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: !cc.ptr<!cc.struct<{!cc.ptr<i8>, i64}>>) -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %0 = cc.compute_ptr %arg2[0, 1] : (!cc.ptr<!cc.struct<{!cc.ptr<i8>, i64}>>) -> !cc.ptr<i64>
    %1 = cc.load %0 : !cc.ptr<i64>
    %2 = arith.addi %arg1, %1 : i64
    %3 = call @malloc(%2) : (i64) -> !cc.ptr<i8>
    %false = arith.constant false
    call @llvm.memcpy.p0i8.p0i8.i64(%3, %arg0, %arg1, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    %4 = cc.compute_ptr %arg2[0, 0] : (!cc.ptr<!cc.struct<{!cc.ptr<i8>, i64}>>) -> !cc.ptr<!cc.ptr<i8>>
    %5 = cc.load %4 : !cc.ptr<!cc.ptr<i8>>
    %6 = cc.compute_ptr %arg0[%arg1] : (!cc.ptr<i8>, i64) -> !cc.ptr<i8>
    call @llvm.memcpy.p0i8.p0i8.i64(%6, %5, %1, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    %7 = cc.undef !cc.struct<{!cc.ptr<i8>, i64}>
    %8 = cc.insert_value %3, %7[0] : (!cc.struct<{!cc.ptr<i8>, i64}>, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %9 = cc.insert_value %2, %8[1] : (!cc.struct<{!cc.ptr<i8>, i64}>, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %9 : !cc.struct<{!cc.ptr<i8>, i64}>
  })#"},

    {stdvecBoolCtorFromInitList, // __nvqpp_initializer_list_to_vector_bool
     {},
     R"#(
  func.func private @__nvqpp_initializer_list_to_vector_bool(!cc.ptr<none>, !cc.ptr<none>, i64) -> ())#"},

    {"__nvqpp_vectorCopyCtor", {llvmMemCopyIntrinsic, "malloc"}, R"#(
  func.func private @__nvqpp_vectorCopyCtor(%arg0: !cc.ptr<i8>, %arg1: i64, %arg2: i64) -> !cc.ptr<i8> {
    %size = arith.muli %arg1, %arg2 : i64
    %0 = call @malloc(%size) : (i64) -> !cc.ptr<i8>
    %false = arith.constant false
    call @llvm.memcpy.p0i8.p0i8.i64(%0, %arg0, %arg1, %false) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
    return %0 : !cc.ptr<i8>
  })#"},

    {"__nvqpp_zeroDynamicResult", {}, R"#(
  func.func private @__nvqpp_zeroDynamicResult() -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cc.cast %c0_i64 : (i64) -> !cc.ptr<i8>
    %1 = cc.undef !cc.struct<{!cc.ptr<i8>, i64}>
    %2 = cc.insert_value %0, %1[0] : (!cc.struct<{!cc.ptr<i8>, i64}>, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %3 = cc.insert_value %c0_i64, %2[1] : (!cc.struct<{!cc.ptr<i8>, i64}>, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %3 : !cc.struct<{!cc.ptr<i8>, i64}>
  })#"},

    {cudaq::runtime::launchKernelFuncName, // altLaunchKernel
     {},
     R"#(
  func.func private @altLaunchKernel(!cc.ptr<i8>, !cc.ptr<i8>, !cc.ptr<i8>, i64, i64) -> ())#"},

    {llvmMemCopyIntrinsic, // llvm.memcpy.p0i8.p0i8.i64
     {},
     R"#(
  func.func private @llvm.memcpy.p0i8.p0i8.i64(!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ())#"},

    {"malloc", {}, "func.func private @malloc(i64) -> !cc.ptr<i8>"}};

static constexpr std::size_t intrinsicTableSize =
    sizeof(intrinsicTable) / sizeof(IntrinsicCode);

inline bool intrinsicTableIsSorted() {
  for (std::size_t i = 0; i < intrinsicTableSize - 1; i++)
    if (!(intrinsicTable[i] < intrinsicTable[i + 1]))
      return false;
  return true;
}

namespace cudaq {

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
  if (sref.size() <= nameLengthHashSize)
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

} // namespace cudaq
