/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This test is compiled inside the runtime directory tree. We include it as a
// regression test and use FileCheck to verify the output.

// RUN: test_argument_conversion | FileCheck %s

#include "common/ArgumentConversion.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/qis/pauli_word.h"
#include "cudaq/qis/state.h"
#include "mlir/Parser/Parser.h"
#include <memory>
#include <numeric>

/// @cond DO_NOT_DOCUMENT
/// @brief Fake simulation or quantum device state to use in tests.
class FakeDeviceState : public cudaq::SimulationState {
private:
  std::string kernelName;
  std::vector<void *> args;
  std::size_t size = 0;
  void *data = 0;

public:
  virtual std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *data,
                       std::size_t dataType) override {
    throw std::runtime_error("Not implemented");
  }

  FakeDeviceState() = default;
  FakeDeviceState(std::size_t size, void *data) : size(size), data(data) {}
  FakeDeviceState(const std::string &kernelName, const std::vector<void *> args)
      : kernelName(kernelName), args(args) {}
  FakeDeviceState(const FakeDeviceState &other)
      : kernelName(other.kernelName), args(other.args) {}

  virtual std::unique_ptr<cudaq::SimulationState>
  createFromData(const cudaq::state_data &data) override {
    throw std::runtime_error("Not implemented");
  }

  virtual bool hasData() const override { return data != nullptr; }

  virtual std::optional<std::pair<std::string, std::vector<void *>>>
  getKernelInfo() const override {
    if (!hasData())
      return std::make_pair(kernelName, args);
    throw std::runtime_error("Not implemented");
  }

  virtual Tensor getTensor(std::size_t tensorIdx = 0) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::vector<Tensor> getTensors() const override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::size_t getNumTensors() const override {
    if (hasData())
      return 1;
    throw std::runtime_error("Not implemented");
  }

  virtual std::size_t getNumQubits() const override {
    if (hasData())
      return std::countr_zero(size);
    throw std::runtime_error("Not implemented");
  }

  virtual std::complex<double> overlap(const SimulationState &other) override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::vector<std::complex<double>>
  getAmplitudes(const std::vector<std::vector<int>> &basisStates) override {
    throw std::runtime_error("Not implemented");
  }

  virtual void dump(std::ostream &os) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual precision getPrecision() const override {
    if (hasData())
      return cudaq::SimulationState::precision::fp64;
    throw std::runtime_error("Not implemented");
  }

  virtual void destroyState() override {}

  virtual std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (hasData()) {
      if (tensorIdx != 0)
        throw std::runtime_error("Non-zero tensor index is not supported");

      if (indices.size() != 1)
        throw std::runtime_error(
            "Multi-dimensional tensor index is not supported");

      return *(static_cast<std::complex<double> *>(data) + indices[0]);
    }
    throw std::runtime_error("Not implemented");
  }

  virtual std::size_t getNumElements() const override {
    if (hasData())
      return size;
    throw std::runtime_error("Not implemented");
  }

  virtual bool isDeviceData() const override { return false; }

  virtual bool isArrayLike() const override { return true; }

  virtual void toHost(std::complex<double> *clientAllocatedData,
                      std::size_t numElements) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual void toHost(std::complex<float> *clientAllocatedData,
                      std::size_t numElements) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual ~FakeDeviceState() override {}
};
/// @endcond

extern "C" void __cudaq_deviceCodeHolderAdd(const char *, const char *);

void dumpSubstitutionModules(cudaq::opt::ArgumentConverter &con) {
  // Dump the conversions
  for (auto *kInfo : con.getKernelSubstitutions())
    llvm::outs() << "========================================\n"
                    "Substitution module:\n"
                 << kInfo->getKernelName() << "\n"
                 << kInfo->getSubstitutionModule() << '\n';
}

void doSimpleTest(mlir::MLIRContext *ctx, const std::string &typeName,
                  std::vector<void *> args,
                  const std::string &additionalCode = "") {
  std::string code = additionalCode + R"#(
func.func private @callee(%0: )#" +
                     typeName + R"#()
func.func @__nvqpp__mlirgen__testy(%0: )#" +
                     typeName + R"#() {
  call @callee(%0) : ()#" +
                     typeName + R"#() -> ()
  return
})#";

  // Create the Module
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(code, ctx);
  llvm::outs() << "Source module:\n" << *mod << '\n';
  cudaq::opt::ArgumentConverter ab{"testy", *mod};
  // Create the argument conversions
  ab.gen(args);
  // Dump all conversions
  dumpSubstitutionModules(ab);
}

void doTest(mlir::MLIRContext *ctx, std::vector<std::string> &typeNames,
            std::vector<void *> args, std::size_t startingArgIdx = 0) {

  std::string code;
  llvm::raw_string_ostream ss(code);

  // Create code
  std::vector<int> indices(args.size());
  std::iota(indices.begin(), indices.end(), 0);
  auto argPairs = llvm::zip_equal(indices, typeNames);

  ss << "func.func private @callee(";
  llvm::interleaveComma(argPairs, ss, [&](auto p) {
    ss << "%" << std::get<0>(p) << ": " << std::get<1>(p);
  });
  ss << ")\n";

  ss << "func.func @__nvqpp__mlirgen__testy(";
  llvm::interleaveComma(argPairs, ss, [&](auto p) {
    ss << "%" << std::get<0>(p) << ": " << std::get<1>(p);
  });
  ss << ") {";

  ss << "  call @callee(";
  llvm::interleaveComma(indices, ss, [&](auto p) { ss << "%" << p; });

  ss << "): (";
  llvm::interleaveComma(typeNames, ss, [&](auto t) { ss << t; });
  ss << ") -> ()\n";

  ss << "  return\n";
  ss << "}\n";

  // Create the Module
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(code, ctx);
  llvm::outs() << "Source module:\n" << *mod << '\n';
  cudaq::opt::ArgumentConverter ab{"testy", *mod};
  // Create the argument conversions
  ab.gen_drop_front(args, startingArgIdx);
  // Dump all conversions
  dumpSubstitutionModules(ab);
}

void test_scalars(mlir::MLIRContext *ctx) {
  {
    bool x = true;
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "i1", v);
  }
  // clang-format off
// CHECK-LABEL: Source module:
// CHECK:         func.func private @callee(i1)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:         }
  // clang-format on
  {
    char x = 'X';
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "i8", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(i8)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 88 : i8
// CHECK:         }
  // clang-format on
  {
    std::int16_t x = 103;
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "i16", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(i16)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 103 : i16
// CHECK:         }
  // clang-format on
  {
    std::int32_t x = 14581;
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "i32", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(i32)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 14581 : i32
// CHECK:         }
  // clang-format on
  {
    std::int64_t x = 78190214;
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "i64", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(i64)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 78190214 : i64
// CHECK:         }
  // clang-format on

  {
    float x = 974.17244;
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "f32", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(f32)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 974.172424 : f32
// CHECK:         }
  // clang-format on
  {
    double x = 77.4782348;
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "f64", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(f64)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 77.478234799999996 : f64
// CHECK:         }
  // clang-format on

  {
    cudaq::pauli_word x{"XYZ"};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.charspan", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.charspan)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.string_literal "XYZ" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_1:.*]] = cc.cast %[[VAL_0]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_3:.*]] = cc.stdvec_init %[[VAL_1]], %[[VAL_2]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:         }
  // clang-format on
}

void test_vectors(mlir::MLIRContext *ctx) {
  {
    std::vector<std::int32_t> x;
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<i32>", v);
  }
  // clang-format off
// CHECK: Source module:
// CHECK:  func.func private @callee(!cc.stdvec<i32>)
// CHECK: Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK: %[[VAL_1:.*]] = cc.cast %[[VAL_0]] : (i64) -> !cc.ptr<i32>
// CHECK: %[[VAL_2:.*]] = cc.stdvec_init %[[VAL_1]], %[[VAL_0]] : (!cc.ptr<i32>, i64) -> !cc.stdvec<i32>
// CHECK: }
  // clang-format on

  {
    std::vector<std::int32_t> x = {14581, 0xcafe, 42, 0xbeef};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<i32>", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.stdvec<i32>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = cc.const_array [14581 : i32, 51966 : i32, 42 : i32, 48879 : i32] : !cc.array<i32 x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<i32 x ?>) -> !cc.stdvec<i32>
// CHECK:         }
  // clang-format on

  {
    std::vector<cudaq::pauli_word> x = {cudaq::pauli_word{"XX"},
                                        cudaq::pauli_word{"XY"}};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<!cc.charspan>", v);
  }
  // clang-format off
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = cc.const_array ["XX", "XY"] : !cc.array<!cc.array<i8 x ?> x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<!cc.array<i8 x ?> x ?>) -> !cc.stdvec<!cc.charspan>
 // CHECK:         }
  // clang-format on

  {
    // The code here is generated strictly for the device side. We will never
    // have the template specialization of std::vector<bool> present in any form
    // on the device side. Any such data will always be marshaled correctly. For
    // the test, this means we use std::vector<char> here to avoid the
    // template specialization.
    std::vector<char> x = {true, false};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<i1>", v);
  }
  // clang-format off
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = cc.const_array [true, false] : !cc.array<i1 x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<i1 x ?>) -> !cc.stdvec<i1>
 // CHECK:         }
  // clang-format on

  {
    std::vector<std::vector<cudaq::pauli_word>> x = {
        {cudaq::pauli_word{"XX"}, cudaq::pauli_word{"XY"}},
        {cudaq::pauli_word{"ZI"}, cudaq::pauli_word{"YY"}},
        {cudaq::pauli_word{"ZY"}, cudaq::pauli_word{"YX"}}};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<!cc.stdvec<!cc.charspan>>", v);
  }
  // clang-format off
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = cc.const_array {{\[}}["XX", "XY"], ["ZI", "YY"], ["ZY", "YX"]] : !cc.array<!cc.array<!cc.array<i8 x ?> x ?> x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<!cc.array<!cc.array<i8 x ?> x ?> x ?>) -> !cc.stdvec<!cc.stdvec<!cc.charspan>>
// CHECK:         }
  // clang-format on

  {
    std::vector<std::vector<double>> x = {
        {1.0, 2.0, 3.0}, {14.0, 15.0, 16.0}, {27.1, 28.2, 29.3}};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<!cc.stdvec<f64>>", v);
  }
  // clang-format off
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = cc.const_array {{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00], [1.400000e+01, 1.500000e+01, 1.600000e+01], [2.710000e+01, 2.820000e+01, 2.930000e+01]] : !cc.array<!cc.array<f64 x ?> x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<!cc.array<f64 x ?> x ?>) -> !cc.stdvec<!cc.stdvec<f64>>
// CHECK:         }
  // clang-format on

  {
    std::vector<std::vector<std::int64_t>> x = {
        {1, 2, 3, 0}, {14, 15, 16, 13}, {127, 128, 129, 126}};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<!cc.stdvec<i64>>", v);
  }
  // clang-format off
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = cc.const_array {{\[}}[1, 2, 3, 0], [14, 15, 16, 13], [127, 128, 129, 126]] : !cc.array<!cc.array<i64 x ?> x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<!cc.array<i64 x ?> x ?>) -> !cc.stdvec<!cc.stdvec<i64>>
// CHECK:         }
  // clang-format on

  {
    std::vector<std::vector<char>> x = {{true, true, false, true},
                                        {false, false, false, true},
                                        {true, false, false, true}};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<!cc.stdvec<i1>>", v);
  }
  // clang-format off
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = cc.const_array {{\[}}[true, true, false, true], [false, false, false, true], [true, false, false, true]] : !cc.array<!cc.array<i1 x ?> x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<!cc.array<i1 x ?> x ?>) -> !cc.stdvec<!cc.stdvec<i1>>
// CHECK:         }
  // clang-format on
}

void test_aggregates(mlir::MLIRContext *ctx) {
  {
    struct ure {
      int _0;
      double _1;
      char _2;
      short _3;
    };
    ure x = {static_cast<int>(0xcafebabe), 87.6545, 'A',
             static_cast<short>(0xfade)};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.struct<{i32,f64,i8,i16}>", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.struct<{i32, f64, i8, i16}>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.undef !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_1:.*]] = arith.constant -889275714 : i32
// CHECK:           %[[VAL_2:.*]] = cc.insert_value %[[VAL_0]][0], %[[VAL_1]]  : (!cc.struct<{i32, f64, i8, i16}>, i32) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_3:.*]] = arith.constant 87.654499999999998 : f64
// CHECK:           %[[VAL_4:.*]] = cc.insert_value %[[VAL_2]][1], %[[VAL_3]] : (!cc.struct<{i32, f64, i8, i16}>, f64) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_5:.*]] = arith.constant 65 : i8
// CHECK:           %[[VAL_6:.*]] = cc.insert_value %[[VAL_4]][2], %[[VAL_5]] : (!cc.struct<{i32, f64, i8, i16}>, i8) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_7:.*]] = arith.constant -1314 : i16
// CHECK:           %[[VAL_8:.*]] = cc.insert_value %[[VAL_6]][3], %[[VAL_7]] : (!cc.struct<{i32, f64, i8, i16}>, i16) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:         }
  // clang-format on
}

void test_recursive(mlir::MLIRContext *ctx) {
  {
    struct ure {
      int _0;
      double _1;
      char _2;
      short _3;
    };
    ure x0 = {static_cast<int>(0xcafebabe), 87.6545, 'A',
              static_cast<short>(0xfade)};
    ure x1 = {5412, 23894.5, 'B', 0xada};
    ure x2 = {90210, 782934.78923, 'C', 747};
    std::vector<ure> x = {x0, x1, x2};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<!cc.struct<{i32,f64,i8,i16}>>", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.stdvec<!cc.struct<{i32, f64, i8, i16}>>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.alloca !cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>
// CHECK:           %[[VAL_1:.*]] = cc.undef !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_2:.*]] = arith.constant -889275714 : i32
// CHECK:           %[[VAL_3:.*]] = cc.insert_value %[[VAL_1]][0], %[[VAL_2]] : (!cc.struct<{i32, f64, i8, i16}>, i32) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_4:.*]] = arith.constant 87.654499999999998 : f64
// CHECK:           %[[VAL_5:.*]] = cc.insert_value %[[VAL_3]][1], %[[VAL_4]] : (!cc.struct<{i32, f64, i8, i16}>, f64) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_6:.*]] = arith.constant 65 : i8
// CHECK:           %[[VAL_7:.*]] = cc.insert_value %[[VAL_5]][2], %[[VAL_6]] : (!cc.struct<{i32, f64, i8, i16}>, i8) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_8:.*]] = arith.constant -1314 : i16
// CHECK:           %[[VAL_9:.*]] = cc.insert_value %[[VAL_7]][3], %[[VAL_8]] : (!cc.struct<{i32, f64, i8, i16}>, i16) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_0]][0] : (!cc.ptr<!cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>>) -> !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_10]] : !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           %[[VAL_11:.*]] = cc.undef !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_12:.*]] = arith.constant 5412 : i32
// CHECK:           %[[VAL_13:.*]] = cc.insert_value %[[VAL_11]][0], %[[VAL_12]] : (!cc.struct<{i32, f64, i8, i16}>, i32) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_14:.*]] = arith.constant 2.389450e+04 : f64
// CHECK:           %[[VAL_15:.*]] = cc.insert_value %[[VAL_13]][1], %[[VAL_14]] : (!cc.struct<{i32, f64, i8, i16}>, f64) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_16:.*]] = arith.constant 66 : i8
// CHECK:           %[[VAL_17:.*]] = cc.insert_value %[[VAL_15]][2], %[[VAL_16]] : (!cc.struct<{i32, f64, i8, i16}>, i8) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_18:.*]] = arith.constant 2778 : i16
// CHECK:           %[[VAL_19:.*]] = cc.insert_value %[[VAL_17]][3], %[[VAL_18]] : (!cc.struct<{i32, f64, i8, i16}>, i16) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_0]][1] : (!cc.ptr<!cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>>) -> !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           cc.store %[[VAL_19]], %[[VAL_20]] : !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           %[[VAL_21:.*]] = cc.undef !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_22:.*]] = arith.constant 90210 : i32
// CHECK:           %[[VAL_23:.*]] = cc.insert_value %[[VAL_21]][0], %[[VAL_22]] : (!cc.struct<{i32, f64, i8, i16}>, i32) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_24:.*]] = arith.constant 782934.78922999999 : f64
// CHECK:           %[[VAL_25:.*]] = cc.insert_value %[[VAL_23]][1], %[[VAL_24]] : (!cc.struct<{i32, f64, i8, i16}>, f64) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_26:.*]] = arith.constant 67 : i8
// CHECK:           %[[VAL_27:.*]] = cc.insert_value %[[VAL_25]][2], %[[VAL_26]] : (!cc.struct<{i32, f64, i8, i16}>, i8) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_28:.*]] = arith.constant 747 : i16
// CHECK:           %[[VAL_29:.*]] = cc.insert_value %[[VAL_27]][3], %[[VAL_28]] : (!cc.struct<{i32, f64, i8, i16}>, i16) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_30:.*]] = cc.compute_ptr %[[VAL_0]][2] : (!cc.ptr<!cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>>) -> !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           cc.store %[[VAL_29]], %[[VAL_30]] : !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           %[[VAL_31:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_32:.*]] = cc.stdvec_init %[[VAL_0]], %[[VAL_31]] : (!cc.ptr<!cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>>, i64) -> !cc.stdvec<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:         }
  // clang-format on
}

void test_simulation_state(mlir::MLIRContext *ctx) {
  {
    std::vector<std::complex<double>> data{M_SQRT1_2, M_SQRT1_2, 0., 0.,
                                           0.,        0.,        0., 0.};
    auto x = cudaq::state(new FakeDeviceState(data.size(), data.data()));
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.ptr<!quake.state>", v);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.ptr<!quake.state>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i64
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (i64) -> !cc.ptr<!quake.state>
// CHECK:        }
  // clang-format on
}

void test_combinations(mlir::MLIRContext *ctx) {
  {
    bool x = true;
    std::vector<void *> v = {static_cast<void *>(&x)};
    std::vector<std::string> t = {"i1"};
    doTest(ctx, t, v);
  }
  // clang-format off
// CHECK-LABEL: Source module:
// CHECK:         func.func private @callee(i1)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:         }
  // clang-format on

  {
    bool x = true;
    bool y = false;
    std::vector<void *> v = {static_cast<void *>(&x), static_cast<void *>(&y)};
    std::vector<std::string> t = {"i1", "i1"};
    doTest(ctx, t, v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(i1, i1)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:         }
// CHECK-LABEL:   cc.arg_subst[1] {
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:         }
  // clang-format on

  {
    bool x = true;
    std::int32_t y = 42;
    std::vector<void *> v = {static_cast<void *>(&x), static_cast<void *>(&y)};
    std::vector<std::string> t = {"i1", "i32"};
    doTest(ctx, t, v, 1);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(i1, i32)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[1] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 42 : i32
// CHECK:         }
  // clang-format on

  {
    std::vector<std::complex<double>> data{M_SQRT1_2, M_SQRT1_2, 0., 0.,
                                           0.,        0.,        0., 0.};

    std::vector<double> x = {0.5, 0.6};
    cudaq::state y{new FakeDeviceState(data.size(), data.data())};
    std::vector<cudaq::pauli_word> z = {
        cudaq::pauli_word{"XX"},
        cudaq::pauli_word{"XY"},
    };

    std::vector<void *> v = {static_cast<void *>(&x), static_cast<void *>(&y),
                             static_cast<void *>(&z)};
    std::vector<std::string> t = {"!cc.stdvec<f32>", "!cc.ptr<!quake.state>",
                                  "!cc.stdvec<!cc.charspan>"};
    doTest(ctx, t, v);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.stdvec<f32>, !cc.ptr<!quake.state>, !cc.stdvec<!cc.charspan>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK: %[[VAL_0:.*]] = cc.const_array [0.000000e+00 : f32, 1.750000e+00 : f32, 4.17232506E-8 : f32, 1.775000e+00 : f32] : !cc.array<f32 x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<f32 x ?>) -> !cc.stdvec<f32>
// CHECK:         }
// CHECK-LABEL:   cc.arg_subst[1] {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i64
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_1]] : (i64) -> !cc.ptr<!quake.state>
// CHECK:         }
// CHECK-LABEL:   cc.arg_subst[2] {
// CHECK: %[[VAL_0:.*]] = cc.const_array ["XX", "XY"] : !cc.array<!cc.array<i8 x ?> x ?>
// CHECK: %[[VAL_1:.*]] = cc.reify_span %[[VAL_0]] : (!cc.array<!cc.array<i8 x ?> x ?>) -> !cc.stdvec<!cc.charspan>
// CHECK:         }
  // clang-format on
}

int main() {
  mlir::DialectRegistry registry;
  cudaq::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  test_scalars(&context);
  test_vectors(&context);
  test_aggregates(&context);
  test_recursive(&context);
  test_simulation_state(&context);
  test_combinations(&context);
  return 0;
}
