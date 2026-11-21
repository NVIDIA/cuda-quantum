/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include <limits>

using namespace mlir;
namespace cudaq_internal::compiler {

cudaq::CompiledModule Compiler::compileOutcomeProgram(
    const std::string &kernelName, const std::string &source,
    cudaq::KernelArgs args, std::shared_ptr<MLIRContext> context) {
  if (!context)
    throw std::runtime_error(
        "outcome program requires an owning compiler context");
  auto original = parseSourceString<ModuleOp>(source, context.get());
  if (!original)
    throw std::runtime_error("cannot parse the explicit outcome program");
  auto [program, entry, specialized] =
      prepareModule(kernelName, *original, args, /*isEntryPoint=*/true);
  applyPipeline("func.func(memtoreg{quantum=0}),canonicalize,cse,symbol-dce",
                program, kernelName);
  if (!specialized || entry.getNumArguments() != 0)
    throw std::runtime_error(
        "ServerHelper outcome program requires bound arguments");
  return CompiledModuleHelper::createCompiledModule(
      kernelName, {},
      {CompiledModuleHelper::createMlirArtifact("cudaq.outcome_program",
                                                program, std::move(context))});
}

ServerHelperFrequencies
compileServerHelperFrequencies(const cudaq::CompiledModule &program,
                               const cudaq::CompileTarget &target) {
  auto artifact = program.getMlir("cudaq.outcome_program");
  if (!artifact)
    throw std::runtime_error("missing explicit outcome program");
  auto context = artifact->getContext();
  OwningOpRef<ModuleOp> module =
      CompiledModuleHelper::getMlirModuleOp(*artifact).clone();
  auto entry = module->lookupSymbol<func::FuncOp>(
      cudaq::runtime::cudaqGenPrefixName + program.getName());
  auto unsupported = []() -> void {
    throw std::runtime_error(
        "ServerHelper prototype supports exactly one top-level frequencies "
        "request followed by emit of its histogram");
  };
  if (!entry || !entry.getBody().hasOneBlock())
    unsupported();
  cudaq::quake::FrequenciesOp request;
  cudaq::quake::EmitOp publication;
  for (auto &op : entry.front()) {
    if (auto frequencies = dyn_cast<cudaq::quake::FrequenciesOp>(op)) {
      if (request || publication)
        unsupported();
      request = frequencies;
    } else if (auto emit = dyn_cast<cudaq::quake::EmitOp>(op)) {
      if (!request || publication || emit.getValue() != request.getHistogram())
        unsupported();
      publication = emit;
    } else if (!isa<arith::ConstantOp, func::ReturnOp>(op)) {
      unsupported();
    }
  }
  if (!request || !publication || !request.getHistogram().hasOneUse())
    unsupported();
  APInt shotCount;
  if (!matchPattern(request.getShots(), m_ConstantInt(&shotCount)) ||
      shotCount.isNegative() || shotCount.getActiveBits() > 63)
    throw std::runtime_error(
        "ServerHelper frequencies requires a bound nonnegative shot count");
  auto shots = static_cast<std::size_t>(shotCount.getZExtValue());
  auto prepare = module->lookupSymbol<func::FuncOp>(request.getPreparation());
  auto reg = cast<cudaq::quake::VeqType>(prepare.getArgumentTypes()[0]);
  bool hasIntermediateReadout = false;
  (*module)->walk([&](Operation *operation) {
    hasIntermediateReadout |=
        isa<cudaq::quake::MeasurementInterface, cudaq::quake::ResetOp>(
            operation);
  });
  if (hasIntermediateReadout)
    throw std::runtime_error("ServerHelper prototype does not yet map "
                             "intermediate measurements or resets");

  // Provider code is one preparation and its explicit final readout. Repetition
  // and accumulation are fulfilled by provider shot batching.
  OpBuilder b(request);
  auto loc = request.getLoc();
  Value q = cudaq::quake::AllocaOp::create(b, loc, reg);
  SmallVector<Value> arguments{q};
  llvm::append_range(arguments, request.getArgs());
  func::CallOp::create(b, loc, prepare, arguments);
  for (std::size_t index = 0; index < reg.getSize(); ++index) {
    Value bit = cudaq::quake::ExtractRefOp::create(b, loc, q, index);
    cudaq::quake::MzOp::create(
        b, loc, cudaq::quake::MeasureType::get(context.get()), bit);
  }
  cudaq::quake::DeallocOp::create(b, loc, q);
  publication.erase();
  request.erase();

  // Emissions performed by a preparation cannot be represented by a sampling-
  // only helper. Reject them before generating or submitting provider code.
  bool hasEmissions = false;
  bool hasNoise = false;
  (*module)->walk([&](Operation *operation) {
    hasEmissions |= isa<cudaq::quake::EmitOp>(operation);
    hasNoise |= isa<cudaq::quake::ApplyNoiseOp>(operation);
  });
  if (hasNoise)
    throw std::runtime_error(
        "ServerHelper prototype cannot submit explicit preparation noise");
  if (hasEmissions)
    throw std::runtime_error(
        "ServerHelper prototype cannot transport preparation emissions");
  cudaq::CompileOptions options;
  options.storeReorderIdx = true;
  Compiler compiler(target, options);
  auto circuit = compiler.compile(
      program.getName(), module->getAsOpaquePointer(), {}, true, context);
  return {std::move(circuit), shots};
}
} // namespace cudaq_internal::compiler
