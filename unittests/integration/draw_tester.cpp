/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithms/draw.h>

CUDAQ_TEST(DrawTester, checkEmpty) {

  auto kernel = []() __qpu__ {};

  std::string expected_str = "";
  auto produced_str = cudaq::contrib::draw(kernel);
  EXPECT_EQ(expected_str, produced_str);
}

namespace {
__qpu__ void bar(cudaq::qvector<> &q) {
  double pi_d = M_PI;
  float pi_f = M_PI;
  rx(M_E, q[0]);
  ry(pi_d, q[1]);
  rz<cudaq::adj>(pi_f, q[2]);
}

__qpu__ void zaz(cudaq::qubit &q) { s<cudaq::adj>(q); }

auto kernel = []() __qpu__ {
  cudaq::qvector q(4);
  h(q); // Broadcast
  x<cudaq::ctrl>(q[0], q[1]);
  y<cudaq::ctrl>(q[0], q[1], q[2]);
  y<cudaq::ctrl>(q[2], q[0], q[1]);
  y<cudaq::ctrl>(q[1], q[2], q[0]);
  z(q[2]);

  r1(3.14159, q[0]);
  t<cudaq::adj>(q[1]);
  s(q[2]);

  swap(q[0], q[2]);
  swap(q[1], q[2]);
  swap(q[0], q[1]);
  swap(q[0], q[2]);
  swap(q[1], q[2]);
  swap<cudaq::ctrl>(q[3], q[0], q[1]);
  swap<cudaq::ctrl>(q[0], q[3], q[1], q[2]);
  swap<cudaq::ctrl>(q[1], q[0], q[3]);
  swap<cudaq::ctrl>(q[1], q[2], q[0], q[3]);
  bar(q);
  cudaq::control(zaz, q[1], q[0]);
  cudaq::adjoint(bar, q);
};
} // namespace

CUDAQ_TEST(DrawTester, checkOps) {
  // clang-format off
  // CAUTION: Changing white spaces here will cause the test to fail. Thus be
  // careful that your editor does not remove them automatically!
  std::string expected_str = R"(
     ╭───╮               ╭───╮╭───────────╮                          ╭───────╮»
q0 : ┤ h ├──●────●────●──┤ y ├┤ r1(3.142) ├──────╳─────╳──╳─────╳──●─┤>      ├»
     ├───┤╭─┴─╮  │  ╭─┴─╮╰─┬─╯╰──┬─────┬──╯      │     │  │     │  │ │       │»
q1 : ┤ h ├┤ x ├──●──┤ y ├──●─────┤ tdg ├─────────┼──╳──╳──┼──╳──╳──╳─┤●      ├»
     ├───┤╰───╯╭─┴─╮╰─┬─╯  │     ╰┬───┬╯   ╭───╮ │  │     │  │  │  │ │  swap │»
q2 : ┤ h ├─────┤ y ├──●────●──────┤ z ├────┤ s ├─╳──╳─────╳──╳──┼──╳─│       │»
     ├───┤     ╰───╯              ╰───╯    ╰───╯                │  │ │       │»
q3 : ┤ h ├──────────────────────────────────────────────────────●──●─┤>      ├»
     ╰───╯                                                           ╰───────╯»

################################################################################

╭───────╮╭───────────╮    ╭─────╮   ╭────────────╮
┤>      ├┤ rx(2.718) ├────┤ sdg ├───┤ rx(-2.718) ├
│       │├───────────┤    ╰──┬──╯   ├────────────┤
┤●      ├┤ ry(3.142) ├───────●──────┤ ry(-3.142) ├
│  swap │├───────────┴╮╭───────────╮╰────────────╯
┤●      ├┤ rz(-3.142) ├┤ rz(3.142) ├──────────────
│       │╰────────────╯╰───────────╯              
┤>      ├─────────────────────────────────────────
╰───────╯                                         
)";
  // clang-format on

  expected_str = expected_str.substr(1);
  std::string produced_str = cudaq::contrib::draw(kernel);
  EXPECT_EQ(expected_str.size(), produced_str.size());
  EXPECT_EQ(expected_str, produced_str);
}

CUDAQ_TEST(LatexDrawTester, checkOps) {
  // clang-format off
  std::string expected_str = R"(
\documentclass{minimal}
\usepackage{quantikz}
\begin{document}
\begin{quantikz}
  \lstick{$q_0$} & \gate{H} & \ctrl{1} & \ctrl{2} & \ctrl{1} & \gate{Y} & \gate{R_1(3.142)} & \qw & \swap{2} & \qw & \swap{1} & \swap{2} & \qw & \swap{1} & \ctrl{2} & \swap{3} & \swap{3} & \gate{R_x(2.718)} & \gate{S^\dag} & \gate{R_x(-2.718)} & \qw \\
  \lstick{$q_1$} & \gate{H} & \gate{X} & \ctrl{1} & \gate{Y} & \ctrl{-1} & \gate{T^\dag} & \qw & \qw & \swap{1} & \targX{} & \qw & \swap{1} & \targX{} & \swap{1} & \ctrl{2} & \ctrl{2} & \gate{R_y(3.142)} & \ctrl{-1} & \gate{R_y(-3.142)} & \qw \\
  \lstick{$q_2$} & \gate{H} & \qw & \gate{Y} & \ctrl{-1} & \ctrl{-2} & \gate{Z} & \gate{S} & \targX{} & \targX{} & \qw & \targX{} & \targX{} & \qw & \targX{} & \qw & \ctrl{-2} & \gate{R_z(-3.142)} & \gate{R_z(3.142)} & \qw & \qw \\
  \lstick{$q_3$} & \gate{H} & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \ctrl{-3} & \ctrl{-2} & \targX{} & \targX{} & \qw & \qw & \qw & \qw \\
\end{quantikz}
\end{document}
)";
  // clang-format on
  expected_str = expected_str.substr(1);
  std::string produced_str = cudaq::contrib::draw("latex", kernel);
  EXPECT_EQ(expected_str.size(), produced_str.size());
  EXPECT_EQ(expected_str, produced_str);
}
