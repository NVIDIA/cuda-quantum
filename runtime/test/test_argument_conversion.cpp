/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
    std::vector<std::int32_t> x = {14581, 0xcafe, 42, 0xbeef};
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.stdvec<i32>", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.stdvec<i32>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.alloca !cc.array<i32 x 4>
// CHECK:           %[[VAL_1:.*]] = arith.constant 14581 : i32
// CHECK:           %[[VAL_2:.*]] = cc.compute_ptr %[[VAL_0]][0] : (!cc.ptr<!cc.array<i32 x 4>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 51966 : i32
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_0]][1] : (!cc.ptr<!cc.array<i32 x 4>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_0]][2] : (!cc.ptr<!cc.array<i32 x 4>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 48879 : i32
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_0]][3] : (!cc.ptr<!cc.array<i32 x 4>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_7]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_10:.*]] = cc.stdvec_init %[[VAL_0]], %[[VAL_9]] : (!cc.ptr<!cc.array<i32 x 4>>, i64) -> !cc.stdvec<i32>
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
// CHECK:           %[[VAL_0:.*]] = cc.alloca !cc.array<!cc.charspan x 2>
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "XX" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_init %[[VAL_2]], %[[VAL_3]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_0]][0] : (!cc.ptr<!cc.array<!cc.charspan x 2>>) -> !cc.ptr<!cc.charspan>
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_6:.*]] = cc.string_literal "XY" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_9:.*]] = cc.stdvec_init %[[VAL_7]], %[[VAL_8]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_0]][1] : (!cc.ptr<!cc.array<!cc.charspan x 2>>) -> !cc.ptr<!cc.charspan>
// CHECK:           cc.store %[[VAL_9:.*]], %[[VAL_10:.*]] : !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_11:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_12:.*]] = cc.stdvec_init %[[VAL_0]], %[[VAL_11]] : (!cc.ptr<!cc.array<!cc.charspan x 2>>, i64) -> !cc.stdvec<!cc.charspan>
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
// CHECK:           %[[VAL_0:.*]] = cc.address_of @[[VAL_GC:.*]] : !cc.ptr<!cc.array<complex<f64> x 8>>
// CHECK:           %[[VAL_1:.*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_2:.*]] = quake.create_state %[[VAL_0]], %[[VAL_1]] : (!cc.ptr<!cc.array<complex<f64> x 8>>, i64) -> !cc.ptr<!quake.state>
// CHECK:        }
// CHECK-DAG:    cc.global constant private @[[VAL_GC]] (dense<[(0.70710678118654757,0.000000e+00), (0.70710678118654757,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<8xcomplex<f64>>) : !cc.array<complex<f64> x 8>
  // clang-format on
}

void test_quantum_state(mlir::MLIRContext *ctx) {

  {
    // @cudaq.kernel
    // def init():
    //    q = cudaq.qvector(2)
    //
    // def kernel(s: cudaq.State):
    //   ...
    //
    // s = cudaq.get_state(init)
    // cudaq.sample(kernel, s)
    auto init = "init";
    auto initCode = "func.func private @__nvqpp__mlirgen__init() {\n"
                    "  %0 = quake.alloca !quake.veq<2>\n"
                    "  return\n"
                    "}\n";
    __cudaq_deviceCodeHolderAdd(init, initCode);

    auto s = cudaq::state(new FakeDeviceState(init, {}));
    std::vector<void *> v = {static_cast<void *>(&s)};
    doSimpleTest(ctx, "!cc.ptr<!quake.state>", v, initCode);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.ptr<!quake.state>)

// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         testy
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = quake.materialize_state @__nvqpp__mlirgen__init.num_qubits_[[HASH_0:.*]], @__nvqpp__mlirgen__init.init_[[HASH_0]] : !cc.ptr<!quake.state>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init.init_[[HASH_0]](%arg0: !quake.veq<?>) -> !quake.veq<?> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_3:.*]] = arith.subi %[[VAL_2]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_4:.*]] = quake.subveq %arg0, %[[VAL_0]], %[[VAL_3]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_0]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_7:.*]] = arith.subi %[[VAL_6]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_8:.*]] = quake.subveq %arg0, %[[VAL_0]], %[[VAL_7]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           return %[[VAL_8]] : !quake.veq<?>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init.num_qubits_[[HASH_0]]() -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init.init_[[HASH_0]]
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init.num_qubits_[[HASH_0]]
  // clang-format on

  {
    // @cudaq.kernel
    // def init0(n: int):
    //    q = cudaq.qvector(n)
    //    x(q[0])
    //
    // def kernel(s: cudaq.State):
    //   ...
    //
    // s = cudaq.get_state(init0, 2)
    // cudaq.sample(kernel, s)
    auto init = "init0";
    auto initCode =
        "func.func private @__nvqpp__mlirgen__init0(%arg0: i64) {\n"
        "  %0 = quake.alloca !quake.veq<?>[%arg0 : i64]\n"
        "  %1 = quake.extract_ref %0[0] : (!quake.veq<?>) -> !quake.ref\n"
        "  quake.x %1 : (!quake.ref) -> ()\n"
        "  return\n"
        "}\n";
    __cudaq_deviceCodeHolderAdd(init, initCode);

    std::int64_t n = 2;
    std::vector<void *> a = {static_cast<void *>(&n)};
    auto s = cudaq::state(new FakeDeviceState(init, a));
    std::vector<void *> v = {static_cast<void *>(&s)};
    doSimpleTest(ctx, "!cc.ptr<!quake.state>", v, initCode);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.ptr<!quake.state>)

// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         testy
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = quake.materialize_state @__nvqpp__mlirgen__init0.num_qubits_[[HASH_0:.*]], @__nvqpp__mlirgen__init0.init_[[HASH_0]] : !cc.ptr<!quake.state>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init0.init_[[HASH_0]](%arg0: i64, %arg1: !quake.veq<?>) -> !quake.veq<?> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.subi %arg0, %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_2]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           quake.x %[[VAL_6]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_7:.*]] = arith.subi %[[VAL_5]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_8:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_7]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           return %[[VAL_8]] : !quake.veq<?>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init0.num_qubits_[[HASH_0]](%arg0: i64) -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           return %[[VAL_1]] : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init0.init_[[HASH_0]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init0.num_qubits_[[HASH_0]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:         }
  // clang-format on

  {
    // @cudaq.kernel
    // def init1(n: int):
    //    q = cudaq.qvector(n)
    //    x(q[0])
    //
    // def state_param(s: cudaq.State)
    //    q = cudaq.qvector(s)
    //    x(q[0])
    //
    // def kernel(s: cudaq.State):
    //   ...
    //
    // s0 = cudaq.get_state(init1, 2)
    // s1 = cudaq.get_state(state_param, s0)
    // s2 = cudaq.get_state(state_param, s1)
    // s3 = cudaq.get_state(state_param, s2)
    // cudaq.sample(kernel, s3)
    auto init = "init1";
    auto initCode = "func.func private @__nvqpp__mlirgen__init1(%arg0: i64) {\n"
                    "  %0 = quake.alloca !quake.veq<?>[%arg0 : i64]\n"
                    "  return\n"
                    "}\n";
    __cudaq_deviceCodeHolderAdd(init, initCode);

    auto stateParam = "state_param";
    auto stateParamCode =
        "func.func private @__nvqpp__mlirgen__state_param(%arg0: "
        "!cc.ptr<!quake.state>) {\n"
        "  %0 = quake.get_number_of_qubits %arg0 : (!cc.ptr<!quake.state>) -> "
        "i64\n"
        "  %1 = quake.alloca !quake.veq<?>[%0 : i64]\n"
        "  %2 = quake.init_state %1, %arg0 : (!quake.veq<?>, "
        "!cc.ptr<!quake.state>) -> !quake.veq<?>\n"
        "  %3 = quake.extract_ref %2[0] : (!quake.veq<?>) -> !quake.ref\n"
        "  quake.x %3 : (!quake.ref) -> ()\n"
        "  return\n"
        "}\n";

    __cudaq_deviceCodeHolderAdd(stateParam, stateParamCode);

    std::int64_t n = 2;
    std::vector<void *> a = {static_cast<void *>(&n)};
    auto s0 = cudaq::state(new FakeDeviceState(init, a));
    std::vector<void *> v0 = {static_cast<void *>(&s0)};
    auto s1 = cudaq::state(new FakeDeviceState(stateParam, v0));
    std::vector<void *> v1 = {static_cast<void *>(&s1)};
    auto s2 = cudaq::state(new FakeDeviceState(stateParam, v1));
    std::vector<void *> v2 = {static_cast<void *>(&s2)};

    auto code = std::string{initCode} + std::string{stateParamCode};
    doSimpleTest(ctx, "!cc.ptr<!quake.state>", v2, code);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.ptr<!quake.state>)

// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         testy
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %0 = quake.materialize_state @__nvqpp__mlirgen__state_param.num_qubits_[[HASH_0:.*]], @__nvqpp__mlirgen__state_param.init_[[HASH_0]] : !cc.ptr<!quake.state>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__state_param.init_[[HASH_0]](%arg0: !cc.ptr<!quake.state>, %arg1: !quake.veq<?>) -> !quake.veq<?> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = quake.get_number_of_qubits %arg0 : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_3:.*]] = arith.subi %[[VAL_2]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_4:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_3]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_0]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_7:.*]] = quake.init_state %[[VAL_4]], %arg0 : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_6]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_9:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_8]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           return %[[VAL_9]] : !quake.veq<?>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__state_param.num_qubits_[[HASH_0]](%arg0: !cc.ptr<!quake.state>) -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = quake.get_number_of_qubits %arg0 : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         state_param.init_[[HASH_0]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %0 = quake.materialize_state @__nvqpp__mlirgen__state_param.num_qubits_[[HASH_1:.*]], @__nvqpp__mlirgen__state_param.init_[[HASH_1]] : !cc.ptr<!quake.state>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__state_param.init_[[HASH_1]](%arg0: !cc.ptr<!quake.state>, %arg1: !quake.veq<?>) -> !quake.veq<?> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = quake.get_number_of_qubits %arg0 : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_3:.*]] = arith.subi %[[VAL_2]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_4:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_3]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_0]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_7:.*]] = quake.init_state %[[VAL_4]], %arg0 : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_6]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_9:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_8]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           return %[[VAL_9]] : !quake.veq<?>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__state_param.num_qubits_[[HASH_1]](%arg0: !cc.ptr<!quake.state>) -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = quake.get_number_of_qubits %arg0 : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         state_param.init_[[HASH_1]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %0 = quake.materialize_state @__nvqpp__mlirgen__init1.num_qubits_[[HASH_2:.*]], @__nvqpp__mlirgen__init1.init_[[HASH_2]] : !cc.ptr<!quake.state>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init1.init_[[HASH_2]](%arg0: i64,  %arg1: !quake.veq<?>) -> !quake.veq<?> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.subi %arg0, %[[VAL_1]] : i64
// CHECK:           %[[VAL_4:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_2]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_6]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_9:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_8]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           return %[[VAL_9]] : !quake.veq<?>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init1.num_qubits_[[HASH_2]](%arg0: i64) -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           return %[[VAL_1]] : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init1.init_[[HASH_2]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init1.num_qubits_[[HASH_2]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         state_param.num_qubits_[[HASH_1]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %0 = quake.materialize_state @__nvqpp__mlirgen__init1.num_qubits_[[HASH_2]], @__nvqpp__mlirgen__init1.init_[[HASH_2]] : !cc.ptr<!quake.state>
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         state_param.num_qubits_[[HASH_0]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %0 = quake.materialize_state @__nvqpp__mlirgen__state_param.num_qubits_[[HASH_1]], @__nvqpp__mlirgen__state_param.init_[[HASH_1]] : !cc.ptr<!quake.state>
// CHECK:         }
  // clang-format on

  {
    // @cudaq.kernel
    // def init2(n: int):
    //    q0 = cudaq.qvector(n)
    //    x(q0[0])
    //    r = mz(q0[0])
    //    if (r):
    //       q1 = cudaq.qvector(n)
    //       x(q1[0])
    //       y(q0[0])
    //
    // def kernel(s: cudaq.State):
    //   ...
    //
    // s = cudaq.get_state(init2, 2)
    // cudaq.sample(kernel, s)
    auto init = "init2";
    auto initCode =
        " func.func private @__nvqpp__mlirgen__init2(%arg0: i64) {\n"
        "   %2 = quake.alloca !quake.veq<?>[%arg0 : i64]\n"
        "   %3 = quake.extract_ref %2[0] : (!quake.veq<?>) -> !quake.ref\n"
        "   quake.x %3 : (!quake.ref) -> ()\n"
        "   %measOut = quake.mz %3 name \"q0\" : (!quake.ref) -> "
        "!quake.measure\n"
        "   %4 = quake.discriminate %measOut : (!quake.measure) -> i1\n"
        "   cc.if(%4) {\n"
        "    %6 = quake.alloca !quake.veq<?>[%arg0 : i64]\n"
        "    %7 = quake.extract_ref %6[0] : (!quake.veq<?>) -> !quake.ref\n"
        "    quake.x %7 : (!quake.ref) -> ()\n"
        "    %8 = quake.extract_ref %2[1] : (!quake.veq<?>) -> !quake.ref\n"
        "    quake.y %8 : (!quake.ref) -> ()\n"
        "   }\n"
        "   return\n"
        "}\n";

    __cudaq_deviceCodeHolderAdd(init, initCode);

    std::int64_t n = 2;
    std::vector<void *> a = {static_cast<void *>(&n)};
    auto s = cudaq::state(new FakeDeviceState(init, a));
    std::vector<void *> v = {static_cast<void *>(&s)};
    doSimpleTest(ctx, "!cc.ptr<!quake.state>", v, initCode);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.ptr<!quake.state>)

// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         testy
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = quake.materialize_state @__nvqpp__mlirgen__init2.num_qubits_[[HASH_1:.*]], @__nvqpp__mlirgen__init2.init_[[HASH_1]] : !cc.ptr<!quake.state>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init2.init_[[HASH_1]](%arg0: i64, %arg1: !quake.veq<?>) -> !quake.veq<?> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.subi %arg0, %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_2]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           quake.x %[[VAL_6]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.mz %[[VAL_6]] name "q0" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_8:.*]] = quake.discriminate %[[VAL_7]] : (!quake.measure) -> i1
// CHECK:           cc.if(%[[VAL_8]]) {
// CHECK:             %[[VAL_11:.*]] = quake.alloca !quake.veq<?>[%arg0 : i64]
// CHECK:             %[[VAL_12:.*]] = quake.extract_ref %[[VAL_11]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:             quake.x %[[VAL_12]] : (!quake.ref) -> ()
// CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:             quake.y %[[VAL_13]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_5]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_10:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_9]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           return %[[VAL_10]] : !quake.veq<?>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init2.num_qubits_[[HASH_1]](%arg0: i64) -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.addi %[[VAL_0]], %arg0 : i64
// CHECK:           return %[[VAL_1]] : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init2.init_[[HASH_1]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init2.num_qubits_[[HASH_1]]
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:         }
  // clang-format on

  {
    // (No memtoreg pass before argument conversion)
    // @cudaq.kernel
    // def init3(n: int):
    //    q0 = cudaq.qvector(n)
    //
    // def kernel(s: cudaq.State):
    //   ...
    //
    // s = cudaq.get_state(init3, 2)
    // cudaq.sample(kernel, s)
    auto init = "init3";
    auto initCode = " func.func @__nvqpp__mlirgen__init3(%arg0: i64) {\n"
                    "   %0 = cc.alloca i64\n"
                    "   cc.store %arg0, %0 : !cc.ptr<i64>\n"
                    "   %1 = cc.load %0 : !cc.ptr<i64>\n"
                    "   %2 = quake.alloca !quake.veq<?>[%1 : i64]\n"
                    "   return\n"
                    "}\n";

    __cudaq_deviceCodeHolderAdd(init, initCode);

    std::int64_t n = 2;
    std::vector<void *> a = {static_cast<void *>(&n)};
    auto s = cudaq::state(new FakeDeviceState(init, a));
    std::vector<void *> v = {static_cast<void *>(&s)};
    doSimpleTest(ctx, "!cc.ptr<!quake.state>", v, initCode);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.ptr<!quake.state>)

// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         testy
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = quake.materialize_state @__nvqpp__mlirgen__init3.num_qubits_[[HASH_0:.*]], @__nvqpp__mlirgen__init3.init_[[HASH_0]] : !cc.ptr<!quake.state>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init3.init_[[HASH_0]](%arg0: i64, %arg1: !quake.veq<?>) -> !quake.veq<?> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = cc.alloca i64
// CHECK:           cc.store %arg0, %[[VAL_2]] : !cc.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_3]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_5:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_4]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_0]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_0]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_9:.*]] = quake.subveq %arg1, %[[VAL_0]], %[[VAL_8]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           return %[[VAL_9]] : !quake.veq<?>
// CHECK:         }
// CHECK:         func.func private @__nvqpp__mlirgen__init3.num_qubits_[[HASH_0]](%arg0: i64) -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = cc.alloca i64
// CHECK:           cc.store %arg0, %[[VAL_1]] : !cc.ptr<i64>
// CHECK:           %[[VAL_2:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_0]], %[[VAL_2]] : i64
// CHECK:           return %[[VAL_3]] : i64
// CHECK:         }
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init3.init_[[HASH_0]]
// CHECK:         ========================================
// CHECK:         Substitution module:
// CHECK:         init3.num_qubits_[[HASH_0]]
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
// CHECK:           %[[VAL_0:.*]] = cc.alloca !cc.array<f32 x 4>
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = cc.compute_ptr %[[VAL_0]][0] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<f32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1.750000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_0]][1] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<f32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 4.17232506E-8 : f32
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_0]][2] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<f32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 1.775000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_0]][3] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_7]], %[[VAL_8]] : !cc.ptr<f32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_10:.*]] = cc.stdvec_init %[[VAL_0]], %[[VAL_9]] : (!cc.ptr<!cc.array<f32 x 4>>, i64) -> !cc.stdvec<f32>
// CHECK:         }
// CHECK-LABEL:   cc.arg_subst[1] {
// CHECK:           %[[VAL_0:.*]] = cc.address_of @[[VAL_GC:.*]] : !cc.ptr<!cc.array<complex<f64> x 8>>
// CHECK:           %[[VAL_1:.*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_5:.*]] = quake.create_state %[[VAL_0]], %[[VAL_1]] : (!cc.ptr<!cc.array<complex<f64> x 8>>, i64) -> !cc.ptr<!quake.state>
// CHECK:         }
// CHECK-DAG:     cc.global constant private @[[VAL_GC]] (dense<[(0.70710678118654757,0.000000e+00), (0.70710678118654757,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<8xcomplex<f64>>) : !cc.array<complex<f64> x 8>
// CHECK-LABEL:   cc.arg_subst[2] {
// CHECK:           %[[VAL_0:.*]] = cc.alloca !cc.array<!cc.charspan x 2>
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "XX" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_init %[[VAL_2]], %[[VAL_3]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_0]][0] : (!cc.ptr<!cc.array<!cc.charspan x 2>>) -> !cc.ptr<!cc.charspan>
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_6:.*]] = cc.string_literal "XY" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_9:.*]] = cc.stdvec_init %[[VAL_7]], %[[VAL_8]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_0]][1] : (!cc.ptr<!cc.array<!cc.charspan x 2>>) -> !cc.ptr<!cc.charspan>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_10]] : !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_11:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_12:.*]] = cc.stdvec_init %[[VAL_0]], %[[VAL_11]] : (!cc.ptr<!cc.array<!cc.charspan x 2>>, i64) -> !cc.stdvec<!cc.charspan>
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
  test_quantum_state(&context);
  test_combinations(&context);
  return 0;
}
