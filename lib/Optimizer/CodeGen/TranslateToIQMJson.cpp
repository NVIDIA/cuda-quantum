/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/CodeGen/Emitter.h"
#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatAdapters.h"
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace mlir;

static LogicalResult emitOperation(nlohmann::json &json,
                                   cudaq::Emitter &emitter, Operation &op);

static LogicalResult emitEntryPoint(nlohmann::json &json,
                                    cudaq::Emitter &emitter, func::FuncOp op) {
  if (op.getBody().getBlocks().size() != 1)
    op.emitError("Cannot translate kernels with more than 1 block to IQM Json. "
                 "Must be a straight-line representation.");

  cudaq::Emitter::Scope scope(emitter, /*isEntryPoint=*/true);
  json["name"] = op.getName().str();
  std::vector<nlohmann::json> instructions;
  for (Operation &op : op.getOps()) {
    nlohmann::json instruction = nlohmann::json::object();
    if (failed(emitOperation(instruction, emitter, op)))
      return failure();
    if (!instruction.empty())
      instructions.push_back(instruction);
  }
  json["instructions"] = instructions;
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json,
                                   cudaq::Emitter &emitter, ModuleOp moduleOp) {
  func::FuncOp entryPoint = nullptr;
  for (Operation &op : moduleOp) {
    if (op.hasAttr(cudaq::entryPointAttrName)) {
      if (entryPoint)
        return moduleOp.emitError("has multiple entrypoints");
      entryPoint = dyn_cast_or_null<func::FuncOp>(op);
      continue;
    }
  }
  if (!entryPoint)
    return moduleOp.emitError("does not contain an entrypoint");
  return emitEntryPoint(json, emitter, entryPoint);
}

static LogicalResult emitOperation(nlohmann::json &json,
                                   cudaq::Emitter &emitter,
                                   quake::AllocaOp op) {
  Value refOrVeq = op.getRefOrVec();
  auto name = emitter.createName("QB", 1);
  emitter.getOrAssignName(refOrVeq, name);
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json,
                                   cudaq::Emitter &emitter,
                                   quake::ExtractRefOp op) {
  std::optional<int64_t> index = std::nullopt;
  if (op.hasConstantIndex())
    index = op.getConstantIndex();
  else
    index = cudaq::getIndexValueAsInt(op.getIndex());

  if (!index.has_value())
    return op.emitError("cannot translate runtime index to IQM Json");
  auto qrefName = llvm::formatv("{0}{1}", "QB", *index + 1);
  emitter.getOrAssignName(op.getRef(), qrefName);
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json,
                                   cudaq::Emitter &emitter,
                                   quake::OperatorInterface optor) {
  auto name = optor->getName().stripDialect();
  std::vector<std::string> validInstructions{"z", "phased_rx"};
  if (std::find(validInstructions.begin(), validInstructions.end(),
                name.str()) == validInstructions.end())
    optor.emitError(
        "Invalid operation, code not lowered to IQM native gate set (" + name +
        ").");

  std::vector<std::string> qubits;

  if (name == "z") {
    if (optor.getControls().size() != 1)
      optor.emitError(
          "IQM gate set only supports Z gates with exactly one control.");
    json["name"] = "cz";
    json["args"] = nlohmann::json::object();
    for (auto control : optor.getControls())
      qubits.push_back(emitter.getOrAssignName(control).str());
  } else {
    json["name"] = name;

    if (optor.getParameters().size() != 2)
      optor.emitError("IQM phased_rx gate expects exactly two parameters.");

    auto parameter0 =
        cudaq::getParameterValueAsDouble(optor.getParameters()[0]);
    auto parameter1 =
        cudaq::getParameterValueAsDouble(optor.getParameters()[1]);

    auto convertToFullTurns = [](double &angleInRadians) {
      return angleInRadians / (2 * M_PI);
    };
    json["args"]["angle_t"] = convertToFullTurns(*parameter0);
    json["args"]["phase_t"] = convertToFullTurns(*parameter1);
  }

  if (optor.getTargets().size() != 1)
    optor.emitError("IQM operation " + name + " supports exactly one target.");

  qubits.push_back(emitter.getOrAssignName(optor.getTargets().front()).str());

  json["qubits"] = qubits;

  return success();
}

static LogicalResult emitOperation(nlohmann::json &json,
                                   cudaq::Emitter &emitter, quake::MzOp op) {
  json["name"] = "measurement";
  std::vector<std::string> qubits;
  for (auto target : op.getTargets())
    qubits.push_back(emitter.getOrAssignName(target).str());

  json["qubits"] = qubits;
  json["args"] = nlohmann::json::object();
  auto join_lambda = [](std::string a, std::string b) {
    return a + std::string("_") + b;
  };
  json["args"]["key"] =
      "m_" + (qubits.empty() ? ""
                             : std::accumulate(++qubits.begin(), qubits.end(),
                                               *qubits.begin(), join_lambda));
  return success();
}

static LogicalResult emitOperation(nlohmann::json &json,
                                   cudaq::Emitter &emitter, Operation &op) {
  using namespace quake;
  return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      .Case<ModuleOp>([&](auto op) { return emitOperation(json, emitter, op); })
      // Quake
      .Case<AllocaOp>([&](auto op) { return emitOperation(json, emitter, op); })
      .Case<ExtractRefOp>(
          [&](auto op) { return emitOperation(json, emitter, op); })
      .Case<OperatorInterface>(
          [&](auto optor) { return emitOperation(json, emitter, optor); })
      .Case<MzOp>([&](auto op) { return emitOperation(json, emitter, op); })
      // Ignore
      .Case<DeallocOp>([&](auto op) { return success(); })
      .Case<func::ReturnOp>([&](auto op) { return success(); })
      .Case<arith::ConstantOp>([&](auto op) { return success(); })
      .Default([&](Operation *) -> LogicalResult {
        // Allow LLVM and cc dialect ops (for storing measure results).
        if (op.getName().getDialectNamespace().equals("llvm") ||
            op.getName().getDialectNamespace().equals("cc") ||
            op.getName().getDialectNamespace().equals("arith"))
          return success();
        return op.emitOpError() << "unable to translate op to IQM Json "
                                << op.getName().getIdentifier().str();
      });
}

LogicalResult cudaq::translateToIQMJson(Operation *op, llvm::raw_ostream &os) {
  nlohmann::json j;
  Emitter emitter(os);
  auto ret = emitOperation(j, emitter, *op);
  os << j.dump(4);
  return ret;
}
