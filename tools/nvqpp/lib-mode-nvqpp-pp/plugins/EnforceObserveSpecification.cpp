/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../AnalysisPlugin.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"

using namespace clang;

namespace {

/// @brief This AnalysisPlugin will report an error if it sees a
/// CUDA Quantum kernel with measurements specified being used by
/// `cudaq::observe`.
class EnforceObserveSpecification : public cudaq::AnalysisPlugin {
public:
  void traverseTree(const cudaq::AnalysisPlugin::Function &function,
                    clang::DiagnosticsEngine &de, Rewriter &rewriter) override {
    auto &characteristics = function.first;
    auto *decl = function.second;
    if (characteristics.hasMeasurements && characteristics.isObserved) {
      auto id =
          de.getCustomDiagID(clang::DiagnosticsEngine::Error,
                             "CUDA Quantum kernel passed to cudaq::observe "
                             "cannot have measurements specified.");
      de.Report(decl->getLocation(), id);
    }
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::AnalysisPlugin, EnforceObserveSpecification,
                    observe_spec)