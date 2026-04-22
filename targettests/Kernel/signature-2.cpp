/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s

// All tests should pass.

// Test that we pass vector of struct, struct of struct, and a struct of vector
// through altLaunchKernel() and get the correct values in the spans in the
// kernel.

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

struct Data {
  int _0;
  float _1;
  double _2;
};

void show1(const Data &val) {
  std::cout << std::fixed << std::setw(11) << std::setprecision(3) << '{'
            << val._0 << ", " << val._1 << ", " << val._2 << "}\n";
}

using VectorOfStruct = std::vector<Data>;
struct VS {
  void operator()(VectorOfStruct vs) __qpu__ {
    openvec(0);
    for (std::size_t i = 0, I = vs.size(); i != I; ++i)
      show1(vs[i]);
    closevec(0);
  }
};

struct StructA {
  int _0;
  double _1;
};

struct StructB {
  float _0;
  short _1;
};

struct StructOfStruct {
  StructA _0;
  StructB _1;
};

void show2(const StructOfStruct &ss) {
  std::cout << std::fixed << std::setw(11) << std::setprecision(3) << "A { "
            << ss._0._0 << ' ' << ss._0._1 << " }; B { " << ss._1._0 << ' '
            << ss._1._1 << " }\n";
}

struct SS {
  void operator()(StructOfStruct ss) __qpu__ {
    openvec(0);
    show2(ss);
    closevec(0);
  }
};

struct StructOfVector {
  std::vector<int> _0;
  std::vector<double> _1;
};

void show3(int c) { std::cout << c << ' '; }
void show4(double d) {
  std::cout << std::fixed << std::setw(11) << std::setprecision(3) << d << ' ';
}

struct SV {
  void operator()(StructOfVector sv) __qpu__ {
    openvec(0);
    openvec(2);
    for (std::size_t i = 0, I = sv._0.size(); i != I; ++i) {
      show3(sv._0[i]);
    }
    closevec(2);
    openvec(2);
    for (std::size_t i = 0, I = sv._1.size(); i != I; ++i) {
      show4(sv._1[i]);
    }
    closevec(2);
    closevec(0);
  }
};

void show5(long john) { std::cout << john << ' '; }
void show6(char actor) { std::cout << actor << '\n'; }

struct T {
  void operator()(std::tuple<int, long, double, char> tup) __qpu__ {
    show3(std::get<0>(tup));
    show5(std::get<1>(tup));
    show4(std::get<double>(tup));
    show6(std::get<3>(tup));
  }
};

void show7(double d) {
  std::cout << std::fixed << std::setw(11) << std::setprecision(3) << d << '\n';
}

struct T2 {
  void operator()(std::tuple<long, double> tup) __qpu__ {
    show5(std::get<0>(tup));
    show7(std::get<1>(tup));
  }
};

struct T3 {
  void operator()(std::pair<long, double> tup) __qpu__ {
    show5(std::get<0>(tup));
    show7(std::get<1>(tup));
  }
};

int main() {
  VectorOfStruct vsData = {{1, 1.0f, 95.0}, {2, 18.4f, 86.945}};
  VS{}(vsData);

  StructOfStruct ssData = {{737, 87.25}, {.75, 639}};
  SS{}(ssData);

  StructOfVector svData = {{1, 10, 3, 100}, {1.2, 2.4, 4.8}};
  SV{}(svData);

  std::tuple<int, long, double, char> t1{234, 89238, 3.14, 'Z'};
  T{}(t1);

  std::tuple<long, double> t2{2098, 99.5};
  T2{}(t2);
  
  std::pair<long, double> t3{34061, 1999.2};
  T3{}(t3);

  return 0;
}

// CHECK: {1, 1.000, 95.000}
// CHECK: {2, 18.400, 86.945}
// CHECK: A { 737 87.250 }; B { 0.750 639 }
// CHECK: 1 10 3 100
// CHECK: 1.200 2.400 4.800
// CHECK: 234 89238 3.140 Z
// CHECK: 2098 99.5
// CHECK: 34061 1999.2
