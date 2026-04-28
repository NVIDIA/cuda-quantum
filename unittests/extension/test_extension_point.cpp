/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/extension_point.h"

#include <gtest/gtest.h>

namespace cudaq::testing {

// Define a new extension point for the framework
class MyExtensionPoint : public cudaq::extension_point<MyExtensionPoint> {
public:
  virtual std::string parrotBack(const std::string &msg) const = 0;
  virtual ~MyExtensionPoint() = default;
};

} // namespace cudaq::testing

CUDAQ_INSTANTIATE_REGISTRY_NO_ARGS(cudaq::testing::MyExtensionPoint)

namespace cudaq::testing {

// Define a concrete realization of that extension point
class RepeatBackOne : public MyExtensionPoint {
public:
  std::string parrotBack(const std::string &msg) const override {
    return msg + " from RepeatBackOne.";
  }

  // Extension must provide a creator function
  CUDAQ_ADD_EXTENSION_CREATOR_FUNCTION(MyExtensionPoint, RepeatBackOne)
};

// Extensions must register themselves
CUDAQ_REGISTER_EXTENSION(RepeatBackOne)

class RepeatBackTwo : public MyExtensionPoint {
public:
  std::string parrotBack(const std::string &msg) const override {
    return msg + " from RepeatBackTwo.";
  }
  CUDAQ_ADD_EXTENSION_CREATOR_FUNCTION(MyExtensionPoint, RepeatBackTwo)
};
CUDAQ_REGISTER_EXTENSION(RepeatBackTwo)

} // namespace cudaq::testing

TEST(ExtensionPointTester, checkSimpleExtensionPoint) {

  auto registeredNames = cudaq::testing::MyExtensionPoint::get_registered();
  EXPECT_EQ(registeredNames.size(), 2);
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackTwo") != registeredNames.end());
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackOne") != registeredNames.end());
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackThree") == registeredNames.end());

  {
    auto var = cudaq::testing::MyExtensionPoint::get("RepeatBackOne");
    EXPECT_EQ(var->parrotBack("Hello World"),
              "Hello World from RepeatBackOne.");
  }
  {
    auto var = cudaq::testing::MyExtensionPoint::get("RepeatBackTwo");
    EXPECT_EQ(var->parrotBack("Hello World"),
              "Hello World from RepeatBackTwo.");
  }
}

namespace cudaq::testing {

class MyExtensionPointWithArgs
    : public cudaq::extension_point<MyExtensionPointWithArgs, int, double> {
protected:
  int i;
  double d;

public:
  MyExtensionPointWithArgs(int i, double d) : i(i), d(d) {}
  virtual std::tuple<int, double, std::string> parrotBack() const = 0;
  virtual ~MyExtensionPointWithArgs() = default;
};

} // namespace cudaq::testing

CUDAQ_INSTANTIATE_REGISTRY(cudaq::testing::MyExtensionPointWithArgs, int,
                           double)

namespace cudaq::testing {

class RepeatBackOneWithArgs : public MyExtensionPointWithArgs {
public:
  using MyExtensionPointWithArgs::MyExtensionPointWithArgs;
  std::tuple<int, double, std::string> parrotBack() const override {
    return std::make_tuple(i, d, "RepeatBackOne");
  }

  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      RepeatBackOneWithArgs,
      static std::unique_ptr<MyExtensionPointWithArgs> create(int i, double d) {
        return std::make_unique<RepeatBackOneWithArgs>(i, d);
      })
};

CUDAQ_REGISTER_EXTENSION(RepeatBackOneWithArgs)

class RepeatBackTwoWithArgs : public MyExtensionPointWithArgs {
public:
  using MyExtensionPointWithArgs::MyExtensionPointWithArgs;
  std::tuple<int, double, std::string> parrotBack() const override {
    return std::make_tuple(i, d, "RepeatBackTwo");
  }

  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      RepeatBackTwoWithArgs,
      static std::unique_ptr<MyExtensionPointWithArgs> create(int i, double d) {
        return std::make_unique<RepeatBackTwoWithArgs>(i, d);
      })
};

CUDAQ_REGISTER_EXTENSION(RepeatBackTwoWithArgs)

} // namespace cudaq::testing

TEST(CoreTester, checkSimpleExtensionPointWithArgs) {

  auto registeredNames =
      cudaq::testing::MyExtensionPointWithArgs::get_registered();
  EXPECT_EQ(registeredNames.size(), 2);
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackTwoWithArgs") != registeredNames.end());
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackOneWithArgs") != registeredNames.end());
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackThree") == registeredNames.end());

  {
    auto var = cudaq::testing::MyExtensionPointWithArgs::get(
        "RepeatBackOneWithArgs", 5, 2.2);
    auto [i, d, msg] = var->parrotBack();
    EXPECT_EQ(msg, "RepeatBackOne");
    EXPECT_EQ(i, 5);
    EXPECT_NEAR(d, 2.2, 1e-2);
  }
  {
    auto var = cudaq::testing::MyExtensionPointWithArgs::get(
        "RepeatBackTwoWithArgs", 15, 12.2);
    auto [i, d, msg] = var->parrotBack();
    EXPECT_EQ(msg, "RepeatBackTwo");
    EXPECT_EQ(i, 15);
    EXPECT_NEAR(d, 12.2, 1e-2);
  }
}
