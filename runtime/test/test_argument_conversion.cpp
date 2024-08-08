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

#include "common/ArgumentConversion.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

void doSimpleTest(mlir::MLIRContext *ctx, const std::string &typeName,
                  std::vector<void *> args) {
  std::string code = R"#(
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

  // Dump the conversions
  llvm::outs() << "========================================\n"
                  "Substitution module:\n"
               << ab.getSubstitutionModule() << '\n';
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
    std::string x = "Hi, there!";
    std::vector<void *> v = {static_cast<void *>(&x)};
    doSimpleTest(ctx, "!cc.charspan", v);
  }
  // clang-format off
// CHECK:       Source module:
// CHECK:         func.func private @callee(!cc.charspan)
// CHECK:       Substitution module:

// CHECK-LABEL:   cc.arg_subst[0] {
// CHECK:           %[[VAL_0:.*]] = cc.address_of @cstr.48692C2074686572652100 : !cc.ptr<!llvm.array<11 x i8>>
// CHECK:           %[[VAL_1:.*]] = cc.cast %[[VAL_0]] : (!cc.ptr<!llvm.array<11 x i8>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_3:.*]] = cc.stdvec_init %[[VAL_1]], %[[VAL_2]] : (!cc.ptr<i8>, i64) -> !cc.charspan
// CHECK:         }
// CHECK:         llvm.mlir.global private constant @cstr.48692C2074686572652100("Hi, there!\00") {addr_space = 0 : i32}
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
  return 0;
}
