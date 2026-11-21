/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/gadgets/circuit/standard.h"

#include <iterator>

namespace cudaq::gadgets {
inline namespace v0 {
namespace {
constexpr PortFlags prepared_flags() {
  PortFlags flags;
  flags.prepared = true;
  return flags;
}
constexpr PortFlags reset_flags() {
  PortFlags flags;
  flags.reset = true;
  return flags;
}
constexpr PortFlags discarded_flags() {
  PortFlags flags;
  flags.discarded = true;
  return flags;
}
constexpr OpcodeContract unitary(Opcode op, const char *name, u32 width,
                                 const char *param = nullptr) {
  return {op, name, width, width, 0, {}, {}, param};
}
constexpr OpcodeContract measure(Opcode op, const char *name, u32 width) {
  return {op, name, width, width, 1, {}, {}};
}
constexpr OpcodeContract measure_reset(Opcode op, const char *name, u32 width) {
  return {op, name, width, width, 1, {}, reset_flags()};
}
constexpr OpcodeContract reset(Opcode op, const char *name, u32 width) {
  return {op, name, width, width, 0, {}, reset_flags()};
}
constexpr OpcodeContract prepare(Opcode op, const char *name, u32 width) {
  return {op, name, 0, width, 0, {}, prepared_flags()};
}
constexpr OpcodeContract discard(Opcode op, const char *name, u32 width) {
  return {op, name, width, 0, 0, discarded_flags(), {}};
}

/// Indexed by Opcode.
constexpr OpcodeContract contracts[] = {
    // Paulis.
    unitary(Opcode::I, "I", 1),
    unitary(Opcode::X, "X", 1),
    unitary(Opcode::Y, "Y", 1),
    unitary(Opcode::Z, "Z", 1),
    // Single-qubit Cliffords.
    unitary(Opcode::C_NXYZ, "C_NXYZ", 1),
    unitary(Opcode::C_NZYX, "C_NZYX", 1),
    unitary(Opcode::C_XNYZ, "C_XNYZ", 1),
    unitary(Opcode::C_XYNZ, "C_XYNZ", 1),
    unitary(Opcode::C_XYZ, "C_XYZ", 1),
    unitary(Opcode::C_ZNYX, "C_ZNYX", 1),
    unitary(Opcode::C_ZYNX, "C_ZYNX", 1),
    unitary(Opcode::C_ZYX, "C_ZYX", 1),
    unitary(Opcode::H, "H", 1),
    unitary(Opcode::H_NXY, "H_NXY", 1),
    unitary(Opcode::H_NXZ, "H_NXZ", 1),
    unitary(Opcode::H_NYZ, "H_NYZ", 1),
    unitary(Opcode::H_XY, "H_XY", 1),
    unitary(Opcode::H_YZ, "H_YZ", 1),
    unitary(Opcode::S, "S", 1),
    unitary(Opcode::SQRT_X, "SQRT_X", 1),
    unitary(Opcode::SQRT_X_DAG, "SQRT_X_DAG", 1),
    unitary(Opcode::SQRT_Y, "SQRT_Y", 1),
    unitary(Opcode::SQRT_Y_DAG, "SQRT_Y_DAG", 1),
    unitary(Opcode::S_DAG, "S_DAG", 1),
    // Two-qubit Cliffords.
    unitary(Opcode::CX, "CX", 2),
    unitary(Opcode::CXSWAP, "CXSWAP", 2),
    unitary(Opcode::CY, "CY", 2),
    unitary(Opcode::CZ, "CZ", 2),
    unitary(Opcode::CZSWAP, "CZSWAP", 2),
    unitary(Opcode::ISWAP, "ISWAP", 2),
    unitary(Opcode::ISWAP_DAG, "ISWAP_DAG", 2),
    unitary(Opcode::SQRT_XX, "SQRT_XX", 2),
    unitary(Opcode::SQRT_XX_DAG, "SQRT_XX_DAG", 2),
    unitary(Opcode::SQRT_YY, "SQRT_YY", 2),
    unitary(Opcode::SQRT_YY_DAG, "SQRT_YY_DAG", 2),
    unitary(Opcode::SQRT_ZZ, "SQRT_ZZ", 2),
    unitary(Opcode::SQRT_ZZ_DAG, "SQRT_ZZ_DAG", 2),
    unitary(Opcode::SWAP, "SWAP", 2),
    unitary(Opcode::SWAPCX, "SWAPCX", 2),
    unitary(Opcode::XCX, "XCX", 2),
    unitary(Opcode::XCY, "XCY", 2),
    unitary(Opcode::XCZ, "XCZ", 2),
    unitary(Opcode::YCX, "YCX", 2),
    unitary(Opcode::YCY, "YCY", 2),
    unitary(Opcode::YCZ, "YCZ", 2),
    // Measurement, reset, preparation, and discard.
    measure(Opcode::M, "M", 1),
    measure(Opcode::MX, "MX", 1),
    measure(Opcode::MY, "MY", 1),
    measure_reset(Opcode::MR, "MR", 1),
    measure_reset(Opcode::MRX, "MRX", 1),
    measure_reset(Opcode::MRY, "MRY", 1),
    measure(Opcode::MXX, "MXX", 2),
    measure(Opcode::MYY, "MYY", 2),
    measure(Opcode::MZZ, "MZZ", 2),
    reset(Opcode::R, "R", 1),
    reset(Opcode::RX, "RX", 1),
    reset(Opcode::RY, "RY", 1),
    prepare(Opcode::PREP_X, "PREP_X", 1),
    prepare(Opcode::PREP_Y, "PREP_Y", 1),
    prepare(Opcode::PREP_Z, "PREP_Z", 1),
    discard(Opcode::DISCARD, "DISCARD", 1),
    // Non-Clifford.
    unitary(Opcode::T, "T", 1),
    unitary(Opcode::T_DAG, "T_DAG", 1),
    unitary(Opcode::DIAGONAL_ROTATION, "DIAGONAL_ROTATION", 1, "angle"),
};

constexpr bool contracts_in_opcode_order() {
  for (u32 i = 0; i < std::size(contracts); ++i)
    if (static_cast<u32>(contracts[i].opcode) != i)
      return false;
  return true;
}
static_assert(contracts_in_opcode_order() &&
              std::size(contracts) == num_opcodes);
} // namespace

const OpcodeContract &contract(Opcode opcode) {
  return contracts[static_cast<u32>(opcode)];
}

const OpcodeContract *find_opcode(std::string_view name) {
  for (const auto &entry : contracts)
    if (name == entry.name)
      return &entry;
  return nullptr;
}

} // namespace v0
} // namespace cudaq::gadgets
