/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s

// All tests should pass.

// Tests that arguments of vector, vector of vector, and vector of vector of
// vector can be passed through altLaunchKernel() and get the correct values in
// the span structures in the kernel.

#include <cudaq.h>
#include <iomanip>
#include <iostream>

void openvec(int level) {
  switch (level) {
  case 0:
    std::cout << "{\n";
    break;
  case 1:
    std::cout << "  [\n";
    break;
  case 2:
    std::cout << "    ";
    break;
  default:
    break;
  }
}

void closevec(int level) {
  switch (level) {
  case 0:
    std::cout << "}\n";
    break;
  case 1:
    std::cout << "  ]\n";
    break;
  case 2:
    std::cout << '\n';
  default:
    break;
  }
}

void showvalue(double dub) {
  std::cout << std::fixed << std::setw(11) << std::setprecision(3) << dub
            << ' ';
}

using RaggedCube = std::vector<std::vector<std::vector<double>>>;
struct CubeDumper {
  void operator()(RaggedCube raggedCube) __qpu__ {
    openvec(0);
    for (std::size_t i = 0, I = raggedCube.size(); i != I; ++i) {
      openvec(1);
      for (std::size_t j = 0, J = raggedCube[i].size(); j != J; ++j) {
        openvec(2);
        for (std::size_t k = 0, K = raggedCube[i][j].size(); k != K; ++k) {
          showvalue(raggedCube[i][j][k]);
        }
        closevec(2);
      }
      closevec(1);
    }
    closevec(0);
  }
};

using RaggedMatrix = std::vector<std::vector<double>>;
struct MatrixDumper {
  void operator()(RaggedMatrix theMatrix) __qpu__ {
    openvec(1);
    for (std::size_t j = 0, J = theMatrix.size(); j != J; ++j) {
      openvec(2);
      for (std::size_t k = 0, K = theMatrix[j].size(); k != K; ++k) {
        showvalue(theMatrix[j][k]);
      }
      closevec(2);
    }
    closevec(1);
  }
};

struct VectorDumper {
  void operator()(std::vector<double> vanillaVector) __qpu__ {
    openvec(2);
    for (std::size_t k = 0, K = vanillaVector.size(); k != K; ++k) {
      showvalue(vanillaVector[k]);
    }
    closevec(2);
  }
};

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

  VectorDumper{}(v9);

  MatrixDumper{}(vv1);

  CubeDumper{}(vvv);

  return 0;
}

// clang-format off
// CHECK: 34552.800   -1233.450       0.673     -88.980      14.920      18.120 
// CHECK-LABEL: [
// CHECK-NEXT: 0.000      -2.200 
// CHECK-NEXT: 82.640 
// CHECK-NEXT: 21.000      22.000      32.300      16.400      18.770 
// CHECK-NEXT: 18.400    9385.100  839201.560 
// CHECK-NEXT: ]
// CHECK-LABEL: {
// CHECK-NEXT: [
// CHECK-NEXT: 1.000       2.000       3.000 
// CHECK-NEXT: 11.000      12.000      13.000      14.000 
// CHECK-NEXT: ]
// CHECK-NEXT: [
// CHECK-NEXT: 0.000      -2.200 
// CHECK-NEXT: 82.640 
// CHECK-NEXT: 21.000      22.000      32.300      16.400      18.770 
// CHECK-NEXT: 18.400    9385.100  839201.560 
// CHECK-NEXT: ]
// CHECK-NEXT: [
// CHECK-NEXT: 234799.872       0.000 
// CHECK-NEXT: 9.000    2348.100   92341.453   81904.230       8.000 
// CHECK-NEXT: 0.000       1.000       0.000       1.000       0.000       1.000       2.000 
// CHECK-NEXT: ]
// CHECK-NEXT: [
// CHECK-NEXT: 34552.800   -1233.450       0.673     -88.980      14.920      18.120 
// CHECK-NEXT: ]
// CHECK-NEXT: }
