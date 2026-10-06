/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cmath>
#include <complex>
#include <cstdint>
#include <cudaq/algorithms/draw.h>
#include <cudaq/algorithms/unitary.h>
#include <stdexcept>
#include <vector>

CUDAQ_TEST(DrawTester, checkEmpty) {

  auto kernel = []() __qpu__ {};

  std::string expected_str = "";
  auto produced_str = cudaq::contrib::draw(kernel);
  EXPECT_EQ(expected_str, produced_str);
}

namespace draw_tester {
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
} // namespace draw_tester

using namespace draw_tester;

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

CUDAQ_TEST(LatexDrawTester, skipsNonGateInstructions) {
  cudaq::Trace trace;
  trace.appendInstruction("h", {}, {}, {{2, 0}});
  trace.appendMeasurement("mz", {{2, 0}});
  trace.appendInstruction("x", {}, {}, {{2, 0}});

  const std::string expected_str = R"(\documentclass{minimal}
\usepackage{quantikz}
\begin{document}
\begin{quantikz}
  \lstick{$q_0$} & \gate{H} & \gate{X} & \qw \\
\end{quantikz}
\end{document}
)";

  EXPECT_EQ(expected_str, cudaq::detail::getLaTeXString(trace));
}

CUDAQ_TEST(DrawTester, ownsControlValues) {
  cudaq::Trace trace;
  std::vector<std::int32_t> values{0, 1};
  trace.appendInstruction("x", {}, {{2, 1}, {2, 0}}, {{2, 2}}, values);
  values[0] = 1;
  EXPECT_EQ(trace.begin()->controlValues, (std::vector<std::int32_t>{0, 1}));

  EXPECT_THROW(trace.appendInstruction("x", {}, {{2, 0}}, {{2, 1}}, {0, 1}),
               std::invalid_argument);
  EXPECT_THROW(trace.appendInstruction("x", {}, {{2, 0}}, {{2, 1}}, {-1}),
               std::invalid_argument);
  EXPECT_THROW(trace.appendInstruction("x", {}, {{2, 0}}, {{2, 1}}, {2}),
               std::invalid_argument);
  EXPECT_EQ(trace.getNumInstructions(), 1u);
}

CUDAQ_TEST(DrawTester, openControls) {
  // Note: Direct traces keep open controls intact without compiler expansion
  // into X-conjugated gates.
  // Exercise a controlled box, swap, and a control inside a multi-target box.
  std::vector<cudaq::Trace> traces(3);
  traces[0].appendInstruction("x", {}, {{2, 1}, {2, 0}}, {{2, 2}}, {0, 1});
  traces[1].appendInstruction("swap", {}, {{2, 1}, {2, 0}}, {{2, 2}, {2, 3}},
                              {0, 1});
  // q1 stays open when its position changes within the control list.
  traces[2].appendInstruction("swap", {}, {{2, 2}, {2, 1}}, {{2, 0}, {2, 3}},
                              {1, 0});
  for (std::size_t i = 0; i < traces.size(); ++i) {
    const auto text = cudaq::detail::draw(traces[i]);
    const auto rowStart = text.find("q1 : ");
    ASSERT_NE(rowStart, std::string::npos);
    const auto row =
        text.substr(rowStart, text.find('\n', rowStart) - rowStart);
    EXPECT_NE(row.find("○"), std::string::npos);
    EXPECT_EQ(row.find("●"), std::string::npos);
    EXPECT_NE(text.find("●"), std::string::npos);

    const auto latex = cudaq::detail::getLaTeXString(traces[i]);
    const auto openOffset = i == 0 ? 1 : 2;
    EXPECT_NE(latex.find("\\lstick{$q_1$} & \\octrl{" +
                         std::to_string(openOffset) + "}"),
              std::string::npos);
    EXPECT_NE(latex.find("\\ctrl{"), std::string::npos);
  }
}

CUDAQ_TEST(DrawTester, unitaryWithOpenControls) {
  cudaq::Trace openX;
  openX.appendInstruction("x", {}, {{2, 0}}, {{2, 1}}, {0});
  auto expectedX = cudaq::complex_matrix::identity(4);
  expectedX(0, 0) = expectedX(1, 1) = 0.;
  expectedX(0, 1) = expectedX(1, 0) = 1.;
  const auto actualX = cudaq::contrib::unitary_from_trace(openX);
  for (std::size_t row = 0; row < 4; ++row)
    for (std::size_t column = 0; column < 4; ++column)
      EXPECT_NEAR(std::abs(actualX(row, column) - expectedX(row, column)), 0.0,
                  1e-12);

  // Ordered controls q2=0, q0=1 and the R1 target q1=1 select |q0 q1 q2>=|110>.
  // This is matrix index 6, despite the different order of the control list.
  cudaq::Trace mixedPhase;
  mixedPhase.appendInstruction("r1", {0.37}, {{2, 2}, {2, 0}}, {{2, 1}},
                               {0, 1});
  auto expectedPhase = cudaq::complex_matrix::identity(8);
  expectedPhase(6, 6) = std::exp(std::complex<double>{0., 0.37});
  const auto actualPhase = cudaq::contrib::unitary_from_trace(mixedPhase);
  for (std::size_t row = 0; row < 8; ++row)
    for (std::size_t column = 0; column < 8; ++column)
      EXPECT_NEAR(
          std::abs(actualPhase(row, column) - expectedPhase(row, column)), 0.0,
          1e-12);

  const auto gate = cudaq::complex_matrix::identity(2);
  EXPECT_THROW(cudaq::contrib::make_controlled_unitary(gate, 1, {0, 1}),
               std::invalid_argument);
  EXPECT_THROW(cudaq::contrib::make_controlled_unitary(gate, 1, {-1}),
               std::invalid_argument);
}
