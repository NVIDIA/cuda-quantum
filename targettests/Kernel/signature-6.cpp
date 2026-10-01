/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s

// All tests should pass.

// Tests the host calling convention of a kernel launch, in particular the
// hidden parameters that the host ABI adds to a kernel's signature: a hidden
// `sret` pointer for a result that is returned in memory, and `byval` for a
// struct that is passed in memory. The kernel reports each argument it
// receives with a (non-variadic) hook function, so a misplaced or missing
// hidden parameter shows up as shifted or garbage values. The host checks the
// values that are returned.
//
// The opposite direction, a callback invoked from a kernel, is covered by
// targettests/execution/device_call_abi.cpp.

#include <cudaq.h>
#include <iostream>

void hookInt(const char *who, long v) { std::cout << who << " " << v << '\n'; }
void hookDouble(const char *who, double v) {
  std::cout << who << " " << v << '\n';
}
void hookBig(long a, long b, long c, long d) {
  std::cout << "arg big " << a << ' ' << b << ' ' << c << ' ' << d << '\n';
}

// Too large to be passed or returned in registers. Passed `byval` on x86_64
// and returned via a hidden `sret` pointer.
struct Big {
  long a, b, c, d;
};

// Small enough to be passed and returned in registers.
struct Small {
  int a;
  double b;
};

// A `sret` result that is a static struct, with an argument before and after.
Big freeBig(int n, Big x, double d) __qpu__ {
  hookInt("arg n", n);
  hookBig(x.a, x.b, x.c, x.d);
  hookDouble("arg d", d);
  return {x.d + n, x.c + n, x.b + n, x.a + n};
}

// A result and argument that are small structs.
Small freeSmall(Small s) __qpu__ {
  hookInt("arg small.a", s.a);
  hookDouble("arg small.b", s.b);
  return {s.a + 1, s.b + 1.0};
}

// A `sret` result that is a vector, with a `byval` argument in the middle of
// other arguments.
std::vector<int> freeVec(int n, std::vector<int> v, Big x, float f) __qpu__ {
  hookInt("arg n", n);
  hookInt("arg v.size", v.size());
  hookBig(x.a, x.b, x.c, x.d);
  hookDouble("arg f", f);
  std::vector<int> r(v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    r[i] = v[i] * n;
  return r;
}

std::vector<double> freeVecDouble(std::vector<double> v) __qpu__ {
  hookInt("arg v.size", v.size());
  std::vector<double> r(v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    r[i] = v[i] + 0.5;
  return r;
}

// Results with dynamic parts that are not a flat vector. The host's layout of
// such a value differs from the device's, so the host value is built from the
// device's value.
struct WithVec {
  int tag;
  std::vector<int> data;
};

WithVec freeWithVec(WithVec w) __qpu__ {
  hookInt("arg tag", w.tag);
  hookInt("arg data.size", w.data.size());
  w.tag = w.tag + 1;
  w.data[0] = 100;
  return w;
}

std::vector<std::vector<double>>
freeVecVec(std::vector<std::vector<double>> v) __qpu__ {
  hookInt("arg v.size", v.size());
  hookInt("arg v[1].size", v[1].size());
  v[0][0] = -1.0;
  return v;
}

std::vector<Small> freeVecSmall(std::vector<Small> v) __qpu__ {
  hookInt("arg v.size", v.size());
  v[1].a = 42;
  return v;
}

// The same kernels written as function objects.
struct BigKernel {
  Big operator()(Big x, int n) __qpu__ {
    hookBig(x.a, x.b, x.c, x.d);
    hookInt("arg n", n);
    return {x.a * n, x.b * n, x.c * n, x.d * n};
  }
};

struct VecKernel {
  std::vector<int> operator()(std::vector<int> v, Big x) __qpu__ {
    hookInt("arg v.size", v.size());
    hookBig(x.a, x.b, x.c, x.d);
    std::vector<int> r(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
      r[i] = v[i] + 1;
    return r;
  }
};

int main() {
  Big b = freeBig(9, {1, 2, 3, 4}, 2.5);
  std::cout << "ret big " << b.a << ' ' << b.b << ' ' << b.c << ' ' << b.d
            << '\n';

  Small s = freeSmall({5, 2.5});
  std::cout << "ret small " << s.a << ' ' << s.b << '\n';

  auto v = freeVec(3, {1, 2, 3}, {10, 20, 30, 40}, 1.5f);
  std::cout << "ret vec";
  for (auto x : v)
    std::cout << ' ' << x;
  std::cout << '\n';

  auto vd = freeVecDouble({1.0, 2.0});
  std::cout << "ret vecd";
  for (auto x : vd)
    std::cout << ' ' << x;
  std::cout << '\n';

  WithVec w = freeWithVec({7, {1, 2, 3}});
  std::cout << "ret withvec " << w.tag << ':';
  for (auto x : w.data)
    std::cout << ' ' << x;
  std::cout << '\n';

  auto vv = freeVecVec({{1.0, 2.0}, {3.0}, {}});
  std::cout << "ret vecvec " << vv.size() << ':';
  for (auto &inner : vv) {
    std::cout << " [";
    for (auto x : inner)
      std::cout << ' ' << x;
    std::cout << " ]";
  }
  std::cout << '\n';

  auto vs = freeVecSmall({{1, 1.5}, {2, 2.5}});
  std::cout << "ret vecsmall";
  for (auto &x : vs)
    std::cout << " (" << x.a << ", " << x.b << ")";
  std::cout << '\n';

  Big b2 = BigKernel{}({1, 2, 3, 4}, 2);
  std::cout << "ret bigk " << b2.a << ' ' << b2.b << ' ' << b2.c << ' ' << b2.d
            << '\n';

  auto v2 = VecKernel{}({1, 2, 3}, {5, 6, 7, 8});
  std::cout << "ret veck";
  for (auto x : v2)
    std::cout << ' ' << x;
  std::cout << '\n';
  return 0;
}

// CHECK: arg n 9
// CHECK-NEXT: arg big 1 2 3 4
// CHECK-NEXT: arg d 2.5
// CHECK-NEXT: ret big 13 12 11 10
// CHECK-NEXT: arg small.a 5
// CHECK-NEXT: arg small.b 2.5
// CHECK-NEXT: ret small 6 3.5
// CHECK-NEXT: arg n 3
// CHECK-NEXT: arg v.size 3
// CHECK-NEXT: arg big 10 20 30 40
// CHECK-NEXT: arg f 1.5
// CHECK-NEXT: ret vec 3 6 9
// CHECK-NEXT: arg v.size 2
// CHECK-NEXT: ret vecd 1.5 2.5
// CHECK-NEXT: arg tag 7
// CHECK-NEXT: arg data.size 3
// CHECK-NEXT: ret withvec 8: 100 2 3
// CHECK-NEXT: arg v.size 3
// CHECK-NEXT: arg v[1].size 1
// CHECK-NEXT: ret vecvec 3: [ -1 2 ] [ 3 ] [ ]
// CHECK-NEXT: arg v.size 2
// CHECK-NEXT: ret vecsmall (1, 1.5) (42, 2.5)
// CHECK-NEXT: arg big 1 2 3 4
// CHECK-NEXT: arg n 2
// CHECK-NEXT: ret bigk 2 4 6 8
// CHECK-NEXT: arg v.size 3
// CHECK-NEXT: arg big 5 6 7 8
// CHECK-NEXT: ret veck 2 3 4
