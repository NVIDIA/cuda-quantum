/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This test is compiled inside the runtime directory tree. We include it as a
// regression test and use FileCheck to verify the output.

// RUN: test_argument_conversion | FileCheck %s

#include "FakeSimulationState.h"
#include "common/ArgumentConversion.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/qis/pauli_word.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include <sstream>

void doTest(mlir::MLIRContext *ctx, std::vector<std::string> &typeNames,
            std::vector<void *> args, std::size_t startingArgIdx = 0) {

  // Create code
  std::ostringstream os;
  os << "func.func private @callee(%0: " << typeNames[0];
  for (auto i = 1; i < typeNames.size(); i++)
    os << ", %" << i << ": " << typeNames[i];
  os << ")\n";

  os << "func.func @__nvqpp__mlirgen__testy(%0: " << typeNames[0];
  for (auto i = 1; i < typeNames.size(); i++)
    os << ", %" << i << ": " << typeNames[i];
  os << ") {\n";

  os << "  call @callee(%0";
  for (auto i = 1; i < typeNames.size(); i++)
    os << ", %" << i;
  os << "): ";

  os << "(" << typeNames[0];
  for (auto i = 1; i < typeNames.size(); i++) {
    os << ", " << typeNames[i];
  }
  os << ") -> ()\n";

  os << "  return\n";
  os << "}\n";

  auto code = os.str();

  // Create the Module
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(code, ctx);
  llvm::outs() << "Source module:\n" << *mod << '\n';
  cudaq::opt::ArgumentConverter ab{"testy", *mod};

  // Create the argument conversions
  ab.gen_drop_front(args, startingArgIdx);

  // Dump the conversions
  llvm::outs() << "========================================\n"
                  "Substitution module:\n"
               << ab.getSubstitutionModule() << '\n';
}

void doSimpleTest(mlir::MLIRContext *ctx, const std::string &typeName,
                  void *arg) {
  std::vector<std::string> typeNames{typeName};
  std::vector<void *> args{arg};
  doTest(ctx, typeNames, args, 0);
}

void test_scalars(mlir::MLIRContext *ctx) {
  {
    bool x = true;
    doSimpleTest(ctx, "i1", &x);
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
    doSimpleTest(ctx, "i8", &x);
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
    doSimpleTest(ctx, "i16", &x);
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
    doSimpleTest(ctx, "i32", &x);
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
    doSimpleTest(ctx, "i64", &x);
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
    doSimpleTest(ctx, "f32", &x);
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
    doSimpleTest(ctx, "f64", &x);
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
    doSimpleTest(ctx, "!cc.charspan", &x);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.charspan)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.address_of @cstr.58595A00 : !cc.ptr<!llvm.array<4 x i8>>
// CHECK:           %[[VAL_1:.*]] = cc.cast %[[VAL_0]] : (!cc.ptr<!llvm.array<4 x i8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_3:.*]] = cc.stdvec_init %[[VAL_1]], %[[VAL_2]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:         }
// CHECK-DAG:     llvm.mlir.global private constant @cstr.58595A00("XYZ\00") {addr_space = 0 : i32}
  // clang-format on
}

void test_vectors(mlir::MLIRContext *ctx) {
  {
    std::vector<std::int32_t> x = {14581, 0xcafe, 42, 0xbeef};
    doSimpleTest(ctx, "!cc.stdvec<i32>", &x);
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
    doSimpleTest(ctx, "!cc.stdvec<!cc.charspan>", &x);
  }
  // clang-format off
// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.alloca !cc.array<!cc.charspan x 2>
// CHECK:           %[[VAL_1:.*]] = cc.address_of @cstr.585800 : !cc.ptr<!llvm.array<3 x i8>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!llvm.array<3 x i8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_init %[[VAL_2]], %[[VAL_3]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_0]][0] : (!cc.ptr<!cc.array<!cc.charspan x 2>>) -> !cc.ptr<!cc.charspan>
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_6:.*]] = cc.address_of @cstr.585900 : !cc.ptr<!llvm.array<3 x i8>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!llvm.array<3 x i8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_9:.*]] = cc.stdvec_init %[[VAL_7]], %[[VAL_8]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_0]][1] : (!cc.ptr<!cc.array<!cc.charspan x 2>>) -> !cc.ptr<!cc.charspan>
// CHECK:           cc.store %[[VAL_9:.*]], %[[VAL_10:.*]] : !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_11:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_12:.*]] = cc.stdvec_init %[[VAL_0]], %[[VAL_11]] : (!cc.ptr<!cc.array<!cc.charspan x 2>>, i64) -> !cc.stdvec<!cc.charspan>
// CHECK:         }
// CHECK-DAG:     llvm.mlir.global private constant @cstr.585800("XX\00") {addr_space = 0 : i32}
// CHECK-DAG:     llvm.mlir.global private constant @cstr.585900("XY\00") {addr_space = 0 : i32}
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
    doSimpleTest(ctx, "!cc.struct<{i32,f64,i8,i16}>", &x);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.struct<{i32, f64, i8, i16}>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.undef !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_1:.*]] = arith.constant -889275714 : i32
// CHECK:           %[[VAL_2:.*]] = cc.insert_value %[[VAL_1]], %[[VAL_0]][0] : (!cc.struct<{i32, f64, i8, i16}>, i32) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_3:.*]] = arith.constant 87.654499999999998 : f64
// CHECK:           %[[VAL_4:.*]] = cc.insert_value %[[VAL_3]], %[[VAL_2]][1] : (!cc.struct<{i32, f64, i8, i16}>, f64) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_5:.*]] = arith.constant 65 : i8
// CHECK:           %[[VAL_6:.*]] = cc.insert_value %[[VAL_5]], %[[VAL_4]][2] : (!cc.struct<{i32, f64, i8, i16}>, i8) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_7:.*]] = arith.constant -1314 : i16
// CHECK:           %[[VAL_8:.*]] = cc.insert_value %[[VAL_7]], %[[VAL_6]][3] : (!cc.struct<{i32, f64, i8, i16}>, i16) -> !cc.struct<{i32, f64, i8, i16}>
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
    doSimpleTest(ctx, "!cc.stdvec<!cc.struct<{i32,f64,i8,i16}>>", &x);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.stdvec<!cc.struct<{i32, f64, i8, i16}>>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.alloca !cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>
// CHECK:           %[[VAL_1:.*]] = cc.undef !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_2:.*]] = arith.constant -889275714 : i32
// CHECK:           %[[VAL_3:.*]] = cc.insert_value %[[VAL_2]], %[[VAL_1]][0] : (!cc.struct<{i32, f64, i8, i16}>, i32) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_4:.*]] = arith.constant 87.654499999999998 : f64
// CHECK:           %[[VAL_5:.*]] = cc.insert_value %[[VAL_4]], %[[VAL_3]][1] : (!cc.struct<{i32, f64, i8, i16}>, f64) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_6:.*]] = arith.constant 65 : i8
// CHECK:           %[[VAL_7:.*]] = cc.insert_value %[[VAL_6]], %[[VAL_5]][2] : (!cc.struct<{i32, f64, i8, i16}>, i8) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_8:.*]] = arith.constant -1314 : i16
// CHECK:           %[[VAL_9:.*]] = cc.insert_value %[[VAL_8]], %[[VAL_7]][3] : (!cc.struct<{i32, f64, i8, i16}>, i16) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_0]][0] : (!cc.ptr<!cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>>) -> !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_10]] : !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           %[[VAL_11:.*]] = cc.undef !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_12:.*]] = arith.constant 5412 : i32
// CHECK:           %[[VAL_13:.*]] = cc.insert_value %[[VAL_12]], %[[VAL_11]][0] : (!cc.struct<{i32, f64, i8, i16}>, i32) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_14:.*]] = arith.constant 2.389450e+04 : f64
// CHECK:           %[[VAL_15:.*]] = cc.insert_value %[[VAL_14]], %[[VAL_13]][1] : (!cc.struct<{i32, f64, i8, i16}>, f64) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_16:.*]] = arith.constant 66 : i8
// CHECK:           %[[VAL_17:.*]] = cc.insert_value %[[VAL_16]], %[[VAL_15]][2] : (!cc.struct<{i32, f64, i8, i16}>, i8) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_18:.*]] = arith.constant 2778 : i16
// CHECK:           %[[VAL_19:.*]] = cc.insert_value %[[VAL_18]], %[[VAL_17]][3] : (!cc.struct<{i32, f64, i8, i16}>, i16) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_0]][1] : (!cc.ptr<!cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>>) -> !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           cc.store %[[VAL_19]], %[[VAL_20]] : !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           %[[VAL_21:.*]] = cc.undef !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_22:.*]] = arith.constant 90210 : i32
// CHECK:           %[[VAL_23:.*]] = cc.insert_value %[[VAL_22]], %[[VAL_21]][0] : (!cc.struct<{i32, f64, i8, i16}>, i32) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_24:.*]] = arith.constant 782934.78922999999 : f64
// CHECK:           %[[VAL_25:.*]] = cc.insert_value %[[VAL_24]], %[[VAL_23]][1] : (!cc.struct<{i32, f64, i8, i16}>, f64) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_26:.*]] = arith.constant 67 : i8
// CHECK:           %[[VAL_27:.*]] = cc.insert_value %[[VAL_26]], %[[VAL_25]][2] : (!cc.struct<{i32, f64, i8, i16}>, i8) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_28:.*]] = arith.constant 747 : i16
// CHECK:           %[[VAL_29:.*]] = cc.insert_value %[[VAL_28]], %[[VAL_27]][3] : (!cc.struct<{i32, f64, i8, i16}>, i16) -> !cc.struct<{i32, f64, i8, i16}>
// CHECK:           %[[VAL_30:.*]] = cc.compute_ptr %[[VAL_0]][2] : (!cc.ptr<!cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>>) -> !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           cc.store %[[VAL_29]], %[[VAL_30]] : !cc.ptr<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:           %[[VAL_31:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_32:.*]] = cc.stdvec_init %[[VAL_0]], %[[VAL_31]] : (!cc.ptr<!cc.array<!cc.struct<{i32, f64, i8, i16}> x 3>>, i64) -> !cc.stdvec<!cc.struct<{i32, f64, i8, i16}>>
// CHECK:         }
  // clang-format on
}

void test_state(mlir::MLIRContext *ctx) {
  {
    std::vector<std::complex<double>> data{M_SQRT1_2, M_SQRT1_2, 0., 0.,
                                           0.,        0.,        0., 0.};
    auto x = cudaq::state(new FakeSimulationState(data.size(), data.data()));
    doSimpleTest(ctx, "!cc.ptr<!cc.state>", &x);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.ptr<!cc.state>)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.address_of @[[VAL_GC:.*]] : !cc.ptr<!cc.array<complex<f64> x 8>>
// CHECK:           %[[VAL_1:.*]] = cc.load %[[VAL_0]] : !cc.ptr<!cc.array<complex<f64> x 8>>
// CHECK:           %[[VAL_2:.*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<complex<f64> x 8>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<!cc.array<complex<f64> x 8>>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<complex<f64> x 8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = func.call @__nvqpp_cudaq_state_createFromData_fp64(%[[VAL_4]], %[[VAL_2]]) : (!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.state>) -> !cc.ptr<!cc.state>
// CHECK:        }
// CHECK-DAG:    cc.global constant @[[VAL_GC]] (dense<[(0.70710678118654757,0.000000e+00), (0.70710678118654757,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<8xcomplex<f64>>) : !cc.array<complex<f64> x 8>
// CHECK-DAG:    func.func private @__nvqpp_cudaq_state_createFromData_fp64(!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
  // clang-format on
}

void test_combinations(mlir::MLIRContext *ctx) {
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
    cudaq::state y{new FakeSimulationState(data.size(), data.data())};
    std::vector<cudaq::pauli_word> z = {
        cudaq::pauli_word{"XX"},
        cudaq::pauli_word{"XY"},
    };

    std::vector<void *> v = {static_cast<void *>(&x), static_cast<void *>(&y),
                             static_cast<void *>(&z)};
    std::vector<std::string> t = {"!cc.stdvec<f32>", "!cc.ptr<!cc.state>",
                                  "!cc.stdvec<!cc.charspan>"};
    doTest(ctx, t, v);
  }

  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.stdvec<f32>, !cc.ptr<!cc.state>, !cc.stdvec<!cc.charspan>)
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
// CHECK:           %[[VAL_1:.*]] = cc.load %[[VAL_0]] : !cc.ptr<!cc.array<complex<f64> x 8>>
// CHECK:           %[[VAL_2:.*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<complex<f64> x 8>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<!cc.array<complex<f64> x 8>>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<complex<f64> x 8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = func.call @__nvqpp_cudaq_state_createFromData_fp64(%[[VAL_4]], %[[VAL_2]]) : (!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.state>) -> !cc.ptr<!cc.state>
// CHECK:         }
// CHECK-DAG:     cc.global constant @[[VAL_GC]] (dense<[(0.70710678118654757,0.000000e+00), (0.70710678118654757,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<8xcomplex<f64>>) : !cc.array<complex<f64> x 8>
// CHECK-DAG:     func.func private @__nvqpp_cudaq_state_createFromData_fp64(!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
// CHECK-LABEL:   cc.arg_subst[2] {
// CHECK:           %[[VAL_0:.*]] = cc.alloca !cc.array<!cc.charspan x 2>
// CHECK:           %[[VAL_1:.*]] = cc.address_of @cstr.585800 : !cc.ptr<!llvm.array<3 x i8>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!llvm.array<3 x i8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_init %[[VAL_2]], %[[VAL_3]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_0]][0] : (!cc.ptr<!cc.array<!cc.charspan x 2>>) -> !cc.ptr<!cc.charspan>
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_6:.*]] = cc.address_of @cstr.585900 : !cc.ptr<!llvm.array<3 x i8>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!llvm.array<3 x i8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_9:.*]] = cc.stdvec_init %[[VAL_7]], %[[VAL_8]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_0]][1] : (!cc.ptr<!cc.array<!cc.charspan x 2>>) -> !cc.ptr<!cc.charspan>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_10]] : !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_11:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_12:.*]] = cc.stdvec_init %[[VAL_0]], %[[VAL_11]] : (!cc.ptr<!cc.array<!cc.charspan x 2>>, i64) -> !cc.stdvec<!cc.charspan>
// CHECK:         }
// CHECK-DAG:     llvm.mlir.global private constant @cstr.585800("XX\00") {addr_space = 0 : i32}
// CHECK-DAG:     llvm.mlir.global private constant @cstr.585900("XY\00") {addr_space = 0 : i32}
  // clang-format on
}

int main() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<cudaq::cc::CCDialect, quake::QuakeDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  test_scalars(&context);
  test_vectors(&context);
  test_aggregates(&context);
  test_recursive(&context);
  test_state(&context);
  test_combinations(&context);
  return 0;
}
