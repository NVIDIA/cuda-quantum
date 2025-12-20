/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iomanip>
#include <iostream>

// Tests that we can take a vector or ragged cube as an argument and return
// same.
// Currently, DefaultQPU::launchKernel does not handle return values at all.

using RaggedMatrix = std::vector<std::vector<double>>;
using RaggedCube = std::vector<std::vector<std::vector<double>>>;

RaggedCube modifyCube(RaggedCube &res) {
  for (std::size_t i = 0, I = res.size(); i != I; ++i)
    for (std::size_t j = 0, J = res[i].size(); j != J; ++j)
      for (std::size_t k = 0, K = res[i][j].size(); k != K; ++k)
        res[i][j][k] += 1.0;
  return res;
}

// FIXME: bugs in the bridge
#define NYI /*__qpu__*/

struct OutOfPocketCubeBlender {
  RaggedCube operator()(RaggedCube res) NYI { return modifyCube(res); }
};

struct BasicCubeBlender {
  RaggedCube operator()(RaggedCube res) NYI {
    for (std::size_t i = 0, I = res.size(); i != I; ++i)
      for (std::size_t j = 0, J = res[i].size(); j != J; ++j)
        for (std::size_t k = 0, K = res[i][j].size(); k != K; ++k)
          res[i][j][k] += 2.0;
    return res;
  }
};

struct MoveItCubeBlender {
  RaggedCube operator()(RaggedCube res) NYI {
    for (std::size_t i = 0, I = res.size(); i != I; ++i)
      for (std::size_t j = 0, J = res[i].size(); j != J; ++j)
        for (std::size_t k = 0, K = res[i][j].size(); k != K; ++k)
          res[i][j][k] += 3.0;
    return std::move(res);
  }
};

void dumpCube(const RaggedCube &res) {
  std::cout << "{\n" << std::fixed << std::setw(11) << std::setprecision(3);
  for (std::size_t i = 0, I = res.size(); i != I; ++i) {
    std::cout << "    [ ";
    for (std::size_t j = 0, J = res[i].size(); j != J; ++j) {
      for (std::size_t k = 0, K = res[i][j].size(); k != K; ++k) {
        std::cout << res[i][j][k] << ' ';
      }
    }
    std::cout << "    ]\n";
  }
  std::cout << "}\n";
}

std::vector<double> aroundTheBlock(std::vector<double> &dub) {
  for (std::size_t i = 0, I = dub.size(); i != I; ++i)
    dub[i] += 2.0;
  return dub;
}

struct ShortList {
  std::vector<double> operator()(std::vector<double> dub) NYI {
    return aroundTheBlock(dub);
  }
};

struct GroceryList {
  std::vector<double> operator()(std::vector<double> dub) NYI {
    for (std::size_t i = 0, I = dub.size(); i != I; ++i)
      dub[i] += 22.0;
    return dub;
  }
};

class ShoppingList {
public:
  std::vector<double> operator()(std::vector<double> dub) NYI {
    for (std::size_t i = 0, I = dub.size(); i != I; ++i)
      dub[i] += 12.0;
    return std::move(dub);
  }
};

void dumpList(const std::vector<double> &res) {
  std::cout << "-- " << std::fixed << std::setw(11) << std::setprecision(3);
  for (std::size_t i = 0, I = res.size(); i != I; ++i)
    std::cout << res[i] << ' ';
  std::cout << "--\n";
}

int main() {
  std::vector<double> v0 = {1.0, 2.0, 3.0};
  std::vector<double> v1 = {11.0, 12.0, 13.0, 14.0};
  std::vector<double> v2 = {0.0, -2.2};
  std::vector<double> v3 = {82.64};
  std::vector<double> v4 = {21.0, 22.0, 32.3, 16.4, 18.77};
  std::vector<double> v5 = {18.4, 9385.1, 8.3920156e5};
  std::vector<double> v6 = {234799.872, 0.0};
  std::vector<double> v7 = {9.0, 2348.1, 92341.453, 81904.23, 8.0};
  std::vector<double> v8 = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0};
  std::vector<double> v9 = {34552.8, -1233.45, 0.6728, -88.98, 14.92, 18.12};
  RaggedMatrix vv0 = {v0, v1};
  RaggedMatrix vv1 = {v2, v3, v4, v5};
  RaggedMatrix vv2 = {v6, v7, v8};
  RaggedMatrix vv3 = {v9};
  RaggedCube vvv = {vv0, vv1, vv2, vv3};

  std::vector<double> dub1 = GroceryList{}(v0);
  dumpList(dub1);

  std::vector<double> dub2 = ShortList{}(v0);
  dumpList(dub2);

  std::vector<double> dub3 = ShoppingList{}(v0);
  dumpList(dub3);

  RaggedCube res1 = OutOfPocketCubeBlender{}(vvv);
  dumpCube(res1);

  RaggedCube res2 = BasicCubeBlender{}(vvv);
  dumpCube(res2);

  RaggedCube res3 = MoveItCubeBlender{}(vvv);
  dumpCube(res3);

  return 0;
}

// clang-format off
// CHECK: --      23.000 24.000 25.000 --
// CHECK-NEXT: --       3.000 4.000 5.000 --
// CHECK-NEXT: --      13.000 14.000 15.000 --
// CHECK-LABEL: {
// CHECK-NEXT: [ 2.000 3.000 4.000 12.000 13.000 14.000 15.000     ]
// CHECK-NEXT: [ 1.000 -1.200 83.640 22.000 23.000 33.300 17.400 19.770 19.400 9386.100 839202.560     ]
// CHECK-NEXT: [ 234800.872 1.000 10.000 2349.100 92342.453 81905.230 9.000 1.000 2.000 1.000 2.000 1.000 2.000 3.000     ]
// CHECK-NEXT: [ 34553.800 -1232.450 1.673 -87.980 15.920 19.120     ]
// CHECK-LABEL: {
// CHECK-NEXT: [ 3.000 4.000 5.000 13.000 14.000 15.000 16.000     ]
// CHECK-NEXT: [ 2.000 -0.200 84.640 23.000 24.000 34.300 18.400 20.770 20.400 9387.100 839203.560     ]
// CHECK-NEXT: [ 234801.872 2.000 11.000 2350.100 92343.453 81906.230 10.000 2.000 3.000 2.000 3.000 2.000 3.000 4.000     ]
// CHECK-NEXT: [ 34554.800 -1231.450 2.673 -86.980 16.920 20.120     ]
// CHECK-LABEL: {
// CHECK-NEXT: [ 4.000 5.000 6.000 14.000 15.000 16.000 17.000     ]
// CHECK-NEXT: [ 3.000 0.800 85.640 24.000 25.000 35.300 19.400 21.770 21.400 9388.100 839204.560     ]
// CHECK-NEXT: [ 234802.872 3.000 12.000 2351.100 92344.453 81907.230 11.000 3.000 4.000 3.000 4.000 3.000 4.000 5.000     ]
// CHECK-NEXT: [ 34555.800 -1230.450 3.673 -85.980 17.920 21.120     ]
