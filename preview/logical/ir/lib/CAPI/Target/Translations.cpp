/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx-c/Target/Translations.h"

#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

#include "qlx/Target/Fabric/EmitStim.h"

#include "qlx/Analysis/GadgetEDEM.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

// Wrap an MlirStringCallback as a tiny llvm::raw_string_ostream that buffers
// in std::string and ships the whole result through the callback at flush.
//
// We can't drive the callback incrementally because emit*() implementations
// expect an llvm::raw_ostream and we want a single call to land at the
// Python side; allocating one std::string and shipping it once is fine for
// these translations (output is typically a few KB to a few hundred KB).
inline void shipString(const std::string &s, MlirStringCallback callback,
                       void *userData) {
  callback(MlirStringRef{s.data(), s.size()}, userData);
}

} // namespace

MlirLogicalResult qlxDeriveGadgetEDEM(MlirModule module, MlirStringRef profile,
                                      MlirStringRef faults,
                                      cudaq_gf2_owned_buffer *output) {
  if (!output || output->data || output->size || output->deallocate ||
      output->context)
    return mlirLogicalResultFailure();
  auto result = cudaq::logical::deriveGadgetEDEM(
      unwrap(module), unwrap(profile), unwrap(faults));
  if (mlir::failed(result))
    return mlirLogicalResultFailure();
  *output = result->release();
  return mlirLogicalResultSuccess();
}

MlirLogicalResult qlxProjectCliffordJson(MlirModule module,
                                         MlirStringRef gadget,
                                         MlirStringCallback callback,
                                         void *userData) {
  auto projection =
      cudaq::logical::projectClifford(unwrap(module), unwrap(gadget));
  if (mlir::failed(projection))
    return mlirLogicalResultFailure();
  using Instruction = cudaq::logical::ProjectionInstruction;
  auto kindName = [](Instruction::Kind kind) -> const char * {
    switch (kind) {
    case Instruction::Kind::R:
      return "R";
    case Instruction::Kind::RX:
      return "RX";
    case Instruction::Kind::H:
      return "H";
    case Instruction::Kind::S:
      return "S";
    case Instruction::Kind::SDag:
      return "S_DAG";
    case Instruction::Kind::X:
      return "X";
    case Instruction::Kind::Z:
      return "Z";
    case Instruction::Kind::CX:
      return "CX";
    case Instruction::Kind::CZ:
      return "CZ";
    case Instruction::Kind::Swap:
      return "SWAP";
    case Instruction::Kind::M:
      return "M";
    case Instruction::Kind::MX:
      return "MX";
    case Instruction::Kind::MPP:
      return "MPP";
    }
    llvm_unreachable("unknown Clifford projection instruction");
  };
  llvm::json::Array instructions;
  for (const auto &instruction : projection->instructions) {
    llvm::json::Array paulis;
    for (const auto &factor : instruction.paulis)
      paulis.push_back(llvm::json::Array{
          factor.carrier, factor.pauli == Instruction::Pauli::X   ? "X"
                          : factor.pauli == Instruction::Pauli::Y ? "Y"
                                                                  : "Z"});
    bool measurement = instruction.kind == Instruction::Kind::M ||
                       instruction.kind == Instruction::Kind::MX ||
                       instruction.kind == Instruction::Kind::MPP;
    instructions.push_back(llvm::json::Object{
        {"kind", kindName(instruction.kind)},
        {"carriers", llvm::json::Array(instruction.carriers)},
        {"paulis", std::move(paulis)},
        {"inverted", instruction.inverted},
        {"measurement_start",
         measurement ? llvm::json::Value(instruction.measurementStart)
                     : llvm::json::Value(nullptr)},
        {"site_id", instruction.siteId}});
  }
  auto ports = [](const auto &values) {
    llvm::json::Array result;
    for (const auto &port : values) {
      llvm::json::Array partitions;
      for (const auto &[name, carriers] : port.partitions)
        partitions.push_back(
            llvm::json::Array{name, llvm::json::Array(carriers)});
      result.push_back(
          llvm::json::Object{{"port_ordinal", port.ordinal},
                             {"name", port.name},
                             {"direction", port.direction},
                             {"patch_id", port.patchId},
                             {"partitions", std::move(partitions)}});
    }
    return result;
  };
  llvm::json::Array measurements;
  for (const auto &record : projection->measurements)
    measurements.push_back(
        llvm::json::Object{{"index", record.index},
                           {"record", record.record},
                           {"symbol", record.symbol},
                           {"call_path", llvm::json::Array(record.callPath)},
                           {"field", record.field},
                           {"lane", record.lane},
                           {"carriers", llvm::json::Array(record.carriers)},
                           {"aliases", llvm::json::Array(record.aliases)}});
  llvm::json::Value value(
      llvm::json::Object{{"schema", "cudaq.logical.clifford-projection/v1"},
                         {"instructions", std::move(instructions)},
                         {"inputs", ports(projection->inputPorts)},
                         {"outputs", ports(projection->outputPorts)},
                         {"measurements", std::move(measurements)},
                         {"carrier_count", projection->carrierCount}});
  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << value;
  shipString(out, callback, userData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult qlxTranslateFabricToStim(MlirModule module,
                                           MlirStringCallback callback,
                                           void *userData) {
  auto mod = unwrap(module);
  std::string out;
  llvm::raw_string_ostream os(out);
  if (mlir::failed(qlx::fabric::emitStim(mod, os)))
    return mlirLogicalResultFailure();
  shipString(out, callback, userData);
  return mlirLogicalResultSuccess();
}
