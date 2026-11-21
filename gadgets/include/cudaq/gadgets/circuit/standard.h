/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <string_view>

namespace cudaq::gadgets {
inline namespace v0 {

using u32 = std::uint32_t;

/// Flags for one declared port.
struct PortFlags {
  bool inactive = false;
  bool prepared = false;
  bool reset = false;
  bool discarded = false;
};

/// The standard abstract instruction set: Stim's fixed-width gates, plus
/// preparation, discard, T, and diagonal rotations.
enum class Opcode : u32 {
  // Paulis.
  I,
  X,
  Y,
  Z,
  // Single-qubit Cliffords.
  C_NXYZ,
  C_NZYX,
  C_XNYZ,
  C_XYNZ,
  C_XYZ,
  C_ZNYX,
  C_ZYNX,
  C_ZYX,
  H,
  H_NXY,
  H_NXZ,
  H_NYZ,
  H_XY,
  H_YZ,
  S,
  SQRT_X,
  SQRT_X_DAG,
  SQRT_Y,
  SQRT_Y_DAG,
  S_DAG,
  // Two-qubit Cliffords.
  CX,
  CXSWAP,
  CY,
  CZ,
  CZSWAP,
  ISWAP,
  ISWAP_DAG,
  SQRT_XX,
  SQRT_XX_DAG,
  SQRT_YY,
  SQRT_YY_DAG,
  SQRT_ZZ,
  SQRT_ZZ_DAG,
  SWAP,
  SWAPCX,
  XCX,
  XCY,
  XCZ,
  YCX,
  YCY,
  YCZ,
  // Measurement, reset, preparation, and discard.
  M,
  MX,
  MY,
  MR,
  MRX,
  MRY,
  MXX,
  MYY,
  MZZ,
  R,
  RX,
  RY,
  PREP_X,
  PREP_Y,
  PREP_Z,
  DISCARD,
  // Non-Clifford.
  T,
  T_DAG,
  // TODO: Multi-qubit rotations need variable-width support, or fixed widths
  // (e.g. 1-3).
  DIAGONAL_ROTATION, // Keep last.
};

inline constexpr u32 num_opcodes =
    static_cast<u32>(Opcode::DIAGONAL_ROTATION) + 1;

/// Signature of a standard operation. All ports share flags; input i
/// continues as output i.
struct OpcodeContract {
  Opcode opcode;
  const char *name;
  u32 q_in = 0;
  u32 q_out = 0;
  u32 c_out = 0;
  PortFlags in_flags;
  PortFlags out_flags;
  /// Float parameter name, e.g. "angle"; null if none.
  const char *param = nullptr;
};

const OpcodeContract &contract(Opcode opcode);

/// The contract with this name, or null.
const OpcodeContract *find_opcode(std::string_view name);

} // namespace v0
} // namespace cudaq::gadgets
