/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Target/Emitter.h"
#include "cudaq/Target/IQM/IQMJsonEmitter.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatAdapters.h"

using namespace mlir;

namespace cudaq {

static LogicalResult emitOperation(nlohmann::json &json, Emitter &emitter,
                                   Operation &op);
static LogicalResult emitEntryPoint(nlohmann::json &json, Emitter &emitter,
                                    qtx::CircuitOp circuitOp) {
  if (circuitOp.getBody().getBlocks().size() != 1)
    circuitOp.emitError("Cannot map qtx Circuit op with more than 1 block to "
                        "IQM Json. Must be a flat circuit representation.");

  Emitter::Scope scope(emitter, /*isEntryPoint=*/true);
  json["name"] = circuitOp.getName().str();
  std::vector<nlohmann::json> instructions;
  for (Operation &op : circuitOp.getOps()) {
    nlohmann::json instruction = nlohmann::json::object();
    if (failed(emitOperation(instruction, emitter, op)))
      return failure();
    if (!instruction.empty())
      instructions.push_back(instruction);
  }
  json["instructions"] = instructions;
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json, Emitter &emitter,
                                   ModuleOp moduleOp) {
  qtx::CircuitOp entryPoint = nullptr;
  for (Operation &op : moduleOp) {
    if (op.hasAttr(cudaq::entryPointAttrName)) {
      if (entryPoint)
        return moduleOp.emitError("has multiple entrypoints");
      entryPoint = dyn_cast_or_null<qtx::CircuitOp>(op);
      continue;
    }
  }
  if (!entryPoint)
    return moduleOp.emitError("does not contain an entrypoint");
  return emitEntryPoint(json, emitter, entryPoint);
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json, Emitter &emitter,
                                   qtx::AllocaOp allocaOp) {
  Value wireOrArray = allocaOp.getWireOrArray();
  auto name = emitter.createName();
  emitter.getOrAssignName(wireOrArray, name);
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json, Emitter &emitter,
                                   qtx::OperatorInterface optor) {
  auto name = optor->getName().stripDialect();
  std::vector<std::string> validInstructions{"z", "phased_rx"};
  if (std::find(validInstructions.begin(), validInstructions.end(),
                name.str()) == validInstructions.end())
    optor.emitError(
        "Invalid operation, code not lowered to IQM native gate set (" + name +
        ").");

  json["name"] = name;
  std::vector<std::string> qubits;
  for (auto target : optor.getTargets())
    qubits.push_back(emitter.getOrAssignName(target).str());
  json["qubits"] = qubits;

  if (!optor.getParameters().empty()) {
    // has to be 2 parameters
    auto parameter0 = getParameterValueAsDouble(optor.getParameters()[0]);
    auto parameter1 = getParameterValueAsDouble(optor.getParameters()[1]);

    json["args"]["angle_t"] = *parameter0;
    json["args"]["phase_t"] = *parameter1;
  } else
    json["args"] = nlohmann::json::object();

  emitter.mapValuesName(optor.getTargets(), optor.getNewTargets());
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json, Emitter &emitter,
                                   qtx::MzOp op) {
  json["name"] = "measurement";
  std::vector<std::string> qubits;
  for (auto target : op.getTargets())
    qubits.push_back(emitter.getOrAssignName(target).str());

  json["qubits"] = qubits;

  emitter.mapValuesName(op.getTargets(), op.getNewTargets());
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json, Emitter &emitter,
                                   qtx::ArrayBorrowOp op) {
  for (auto [indexValue, wire] : llvm::zip(op.getIndices(), op.getWires())) {
    auto index = getIndexValueAsInt(indexValue);
    if (!index.has_value())
      return op.emitError("cannot translate runtime index to IQM Json");
    auto wireName = llvm::formatv("{0}{1}", "QB", *index);
    emitter.getOrAssignName(wire, wireName);
  }
  emitter.mapValuesName(op.getArray(), op.getNewArray());
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json, Emitter &emitter,
                                   Operation &op) {
  using namespace qtx;
  return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      .Case<ModuleOp>([&](auto op) { return emitOperation(json, emitter, op); })
      .Case<AllocaOp>([&](auto op) { return emitOperation(json, emitter, op); })
      // Arrays
      .Case<ArraySplitOp>([&](auto op) { return success(); })
      .Case<ArrayBorrowOp>(
          [&](auto op) { return emitOperation(json, emitter, op); })
      .Case<ArrayYieldOp>([&](auto op) { return success(); })
      // Operators
      .Case<OperatorInterface>(
          [&](auto optor) { return emitOperation(json, emitter, optor); })
      // Measurements
      .Case<MzOp>([&](auto op) { return emitOperation(json, emitter, op); })
      // Ignore
      .Case<DeallocOp>([&](auto op) { return success(); })
      .Case<ReturnOp>([&](auto op) { return success(); })
      .Case<arith::ConstantOp>([&](auto op) { return success(); })
      .Default([&](Operation *) -> LogicalResult {
        // allow LLVM dialect ops (for storing measure results)
        if (op.getName().getDialectNamespace().equals("llvm"))
          return success();
        return op.emitOpError("unable to translate op to IQM Json");
      });
}

mlir::LogicalResult translateToIQMJson(mlir::Operation *op,
                                       llvm::raw_ostream &os) {
  nlohmann::json j;
  Emitter emitter(os);
  auto ret = emitOperation(j, emitter, *op);
  os << j.dump(4);
  return ret;
}

} // namespace cudaq
