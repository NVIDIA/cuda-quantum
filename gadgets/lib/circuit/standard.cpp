/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/gadgets/circuit/standard.h"

#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cudaq::gadgets {
inline namespace v0 {
namespace {
/// Payload alternatives, in Payload's order.
enum class PayloadKind : u32 {
  Param,
  PauliString,
  CliffordImages,
  ConditionalPauli,
  DiagonalUnitary,
  Unitary,
};

struct Entry {
  Opcode opcode;
  const char *name;
  /// Unused for variable-width opcodes.
  Shape shape;
  std::optional<PayloadKind> payload;
  bool variable = false;
};

constexpr Entry gate(Opcode op, const char *name, std::size_t width) {
  return {op, name, {width, width, 0, 0}, {}};
}
constexpr Entry rotation(Opcode op, const char *name) {
  return {op, name, {1, 1, 0, 0}, PayloadKind::Param};
}
constexpr Entry prepare(Opcode op, const char *name, std::size_t width) {
  return {op, name, {0, width, 0, 0}, {}};
}
constexpr Entry measure(Opcode op, const char *name, std::size_t width) {
  return {op, name, {width, 0, 0, width}, {}};
}
constexpr Entry variable(Opcode op, const char *name, PayloadKind payload) {
  return {op, name, {}, payload, true};
}

/// Indexed by Opcode.
constexpr Entry entries[] = {
    gate(Opcode::I, "I", 1),
    gate(Opcode::X, "X", 1),
    gate(Opcode::Y, "Y", 1),
    gate(Opcode::Z, "Z", 1),
    gate(Opcode::H, "H", 1),
    gate(Opcode::S, "S", 1),
    gate(Opcode::SX, "SX", 1),
    gate(Opcode::SY, "SY", 1),
    gate(Opcode::CX, "CX", 2),
    gate(Opcode::CZ, "CZ", 2),
    gate(Opcode::CY, "CY", 2),
    gate(Opcode::SWAP, "SWAP", 2),
    variable(Opcode::CLIFFORD, "CLIFFORD", PayloadKind::CliffordImages),
    prepare(Opcode::PREPARE_X, "PREPARE_X", 1),
    prepare(Opcode::PREPARE_Y, "PREPARE_Y", 1),
    prepare(Opcode::PREPARE_Z, "PREPARE_Z", 1),
    measure(Opcode::MEASURE_X, "MEASURE_X", 1),
    measure(Opcode::MEASURE_Y, "MEASURE_Y", 1),
    measure(Opcode::MEASURE_Z, "MEASURE_Z", 1),
    measure(Opcode::BELL_MEASURE, "BELL_MEASURE", 2),
    prepare(Opcode::BELL_PREPARE, "BELL_PREPARE", 2),
    gate(Opcode::RESET, "RESET", 1),
    gate(Opcode::IDLE, "IDLE", 1),
    {Opcode::RANDOM_BIT, "RANDOM_BIT", {0, 0, 0, 1}, {}},
    variable(Opcode::PAULI_MEASUREMENT, "PAULI_MEASUREMENT",
             PayloadKind::PauliString),
    variable(Opcode::CONDITIONAL_PAULI, "CONDITIONAL_PAULI",
             PayloadKind::ConditionalPauli),
    gate(Opcode::T, "T", 1),
    gate(Opcode::SQRT_T, "SQRT_T", 1),
    gate(Opcode::CS, "CS", 2),
    gate(Opcode::CCZ, "CCZ", 3),
    gate(Opcode::CCX, "CCX", 3),
    gate(Opcode::CSWAP, "CSWAP", 3),
    rotation(Opcode::RX, "RX"),
    rotation(Opcode::RY, "RY"),
    rotation(Opcode::RZ, "RZ"),
    variable(Opcode::DIAGONAL_UNITARY, "DIAGONAL_UNITARY",
             PayloadKind::DiagonalUnitary),
    variable(Opcode::UNITARY, "UNITARY", PayloadKind::Unitary),
};

constexpr std::pair<const char *, Opcode> aliases[] = {
    {"NOT", Opcode::X},
    {"CNOT", Opcode::CX},
    {"PREPARE_ZERO", Opcode::PREPARE_Z},
    {"PREPARE_PLUS", Opcode::PREPARE_X},
    {"TOFFOLI", Opcode::CCX},
    {"FREDKIN", Opcode::CSWAP},
};

constexpr bool entries_in_opcode_order() {
  for (u32 i = 0; i < std::size(entries); ++i)
    if (static_cast<u32>(entries[i].opcode) != i)
      return false;
  return true;
}
static_assert(entries_in_opcode_order() && std::size(entries) == num_opcodes);
template <PayloadKind kind, typename T>
constexpr bool payload_is = std::is_same_v<
    std::variant_alternative_t<static_cast<std::size_t>(kind), Payload>, T>;
static_assert(payload_is<PayloadKind::Param, ParamValue> &&
              payload_is<PayloadKind::PauliString, PauliString> &&
              payload_is<PayloadKind::CliffordImages, CliffordImages> &&
              payload_is<PayloadKind::ConditionalPauli, ConditionalPauli> &&
              payload_is<PayloadKind::DiagonalUnitary, DiagonalUnitary> &&
              payload_is<PayloadKind::Unitary, Unitary>);
} // namespace

Shape get_shape(Opcode opcode, const std::optional<Payload> &payload) {
  const Entry &entry = entries[static_cast<u32>(opcode)];
  if (payload.has_value() != entry.payload.has_value() ||
      (payload && payload->index() != static_cast<std::size_t>(*entry.payload)))
    throw std::invalid_argument("payload does not match the opcode");
  if (!entry.variable)
    return entry.shape;

  switch (opcode) {
  case Opcode::CLIFFORD: {
    std::size_t n = std::get<CliffordImages>(*payload).size();
    return {n, n, 0, 0};
  }
  case Opcode::PAULI_MEASUREMENT: {
    const auto &pauli = std::get<PauliString>(*payload);
    if (!pauli.is_hermitian())
      throw std::invalid_argument("measured Pauli must be Hermitian");
    std::size_t n = pauli.size();
    return {n, n, 0, n};
  }
  case Opcode::CONDITIONAL_PAULI: {
    const auto &conditional = std::get<ConditionalPauli>(*payload);
    std::size_t n = conditional.pauli.size();
    return {n, n, conditional.num_bits, 0};
  }
  case Opcode::DIAGONAL_UNITARY: {
    std::size_t n = std::get<DiagonalUnitary>(*payload).size();
    return {n, n, 0, 0};
  }
  default: { // UNITARY
    std::size_t n = std::get<Unitary>(*payload).size();
    return {n, n, 0, 0};
  }
  }
}

std::string_view to_string(Opcode opcode) {
  return entries[static_cast<u32>(opcode)].name;
}

std::optional<Opcode> find_opcode(std::string_view name) {
  for (const Entry &entry : entries)
    if (name == entry.name)
      return entry.opcode;
  for (const auto &[alias, opcode] : aliases)
    if (name == alias)
      return opcode;
  return std::nullopt;
}

} // namespace v0
} // namespace cudaq::gadgets
