/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/platform/qpu_state.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using namespace cudaq;

namespace {
// The quake of a kernel class with data members, as the frontend makes it. The
// entry point carries the `cudaq-member-args` attribute.
std::string quakeWithMembers() {
  return std::string("module {\n"
                     "  func.func @__nvqpp__mlirgen__Prep(%arg0: i32, "
                     "%arg1: f64) attributes {\"cudaq-entrypoint\", "
                     "\"cudaq-kernel\", \"") +
         kernelMemberArgsAttrName + "\" = 1 : i64} {\n    return\n  }\n}\n";
}

std::string quakeWithoutMembers() {
  return "module {\n"
         "  func.func @__nvqpp__mlirgen__Prep(%arg0: f64) attributes "
         "{\"cudaq-entrypoint\", \"cudaq-kernel\"} {\n    return\n  }\n}\n";
}
} // namespace

TEST(QPUStateTester, capturesNameQuakeAndArguments) {
  double theta = 2.5;
  int count = 3;
  QPUState state("Prep", quakeWithoutMembers(), theta, count);
  auto info = state.getKernelInfo();
  ASSERT_TRUE(info.has_value());
  auto &[name, quake, args] = *info;
  EXPECT_EQ(name, "Prep");
  EXPECT_EQ(quake, quakeWithoutMembers());
  ASSERT_EQ(args.size(), 2u);
  EXPECT_EQ(*static_cast<double *>(args[0]), 2.5);
  EXPECT_EQ(*static_cast<int *>(args[1]), 3);
}

TEST(QPUStateTester, argumentsAreCopies) {
  double theta = 2.5;
  QPUState state("Prep", quakeWithoutMembers(), theta);
  theta = 7.0;
  auto [name, quake, args] = state.getKernelInfo().value();
  ASSERT_EQ(args.size(), 1u);
  EXPECT_EQ(*static_cast<double *>(args[0]), 2.5);
}

TEST(QPUStateTester, pointerArgumentCopiesThePointee) {
  double theta = 2.5;
  QPUState state("Prep", quakeWithoutMembers(), &theta);
  theta = 7.0;
  auto [name, quake, args] = state.getKernelInfo().value();
  ASSERT_EQ(args.size(), 1u);
  EXPECT_NE(args[0], static_cast<void *>(&theta));
  EXPECT_EQ(*static_cast<double *>(args[0]), 2.5);
}

TEST(QPUStateTester, noArguments) {
  QPUState state("Prep", quakeWithoutMembers());
  auto [name, quake, args] = state.getKernelInfo().value();
  EXPECT_TRUE(args.empty());
}

// The state of a kernel is its name and arguments. The object of a kernel class
// is not part of that, so a kernel class with data members has no state to
// capture: the values of the data members would be lost.
TEST(QPUStateTester, rejectsKernelClassWithDataMembers) {
  try {
    QPUState state("Prep", quakeWithMembers(), 1.5);
    FAIL() << "expected an exception";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("Prep"), std::string::npos) << msg;
    EXPECT_NE(msg.find("data members"), std::string::npos) << msg;
  }
}

TEST(QPUStateTester, rejectionDoesNotDependOnTheArguments) {
  EXPECT_THROW(QPUState("Prep", quakeWithMembers()), std::runtime_error);
  EXPECT_THROW(QPUState("Prep", quakeWithMembers(), 1, 1.5),
               std::runtime_error);
}

TEST(QPUStateTester, acceptsKernelWithoutDataMembers) {
  EXPECT_NO_THROW(QPUState("Prep", quakeWithoutMembers(), 1.5));
}

TEST(QPUStateTester, hasNoDataOnTheHost) {
  QPUState state("Prep", quakeWithoutMembers(), 1.5);
  EXPECT_FALSE(state.hasData());
  EXPECT_THROW(state.getNumQubits(), std::runtime_error);
  EXPECT_THROW(state.getAmplitude({0, 1}), std::runtime_error);
}
