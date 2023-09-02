/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Support/Verifier.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/BackendUtil.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

namespace cudaq {
namespace nvqpp {
enum class OutputGen { EmitMlir, EmitLlvm, EmitObject };
}

/// Consumer that runs a pair of AST consumers at the same time.
class CudaQASTConsumer : public clang::ASTConsumer {
public:
  CudaQASTConsumer(clang::CompilerInstance &ci, llvm::StringRef inFile,
                   std::unique_ptr<clang::ASTConsumer> &consumer0,
                   std::unique_ptr<clang::ASTConsumer> &consumer1,
                   nvqpp::OutputGen genAction,
                   mlir::OwningOpRef<mlir::ModuleOp> &module,
                   mlir::MLIRContext *context,
                   MangledKernelNamesMap &kernelNames,
                   const std::string &quakeOptPineline = "");

  virtual ~CudaQASTConsumer();

  // The following is boilerplate to override all the virtual functions that
  // appear in the union of the two AST consumers, namely clang::BackendConsumer
  // and nvqpp::MLIRASTConsumer.
  template <typename A, typename B>
  inline void applyConsumers(void (clang::ASTConsumer::*fun)(A), B &&arg) {
    for (auto *c : consumers)
      (c->*fun)(arg);
  }
  void HandleTranslationUnit(clang::ASTContext &ctxt) override;
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override;
  void Initialize(clang::ASTContext &Ctx) override;
  bool HandleTopLevelDecl(clang::DeclGroupRef D) override;
  void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override;
  void HandleInterestingDecl(clang::DeclGroupRef D) override;
  void HandleTagDeclDefinition(clang::TagDecl *D) override;
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override;
  void CompleteTentativeDefinition(clang::VarDecl *D) override;
  void CompleteExternalDeclaration(clang::VarDecl *D) override;
  void AssignInheritanceModel(clang::CXXRecordDecl *RD) override;
  void HandleVTable(clang::CXXRecordDecl *RD) override;

private:
  /// Apply Quake optimization (according to the opt pipeline configuration)
  void applyOpt();
  /// Apply Quake -> LLVM IR translation
  void applyTranslate();
  /// Emit output file from Quake (for the quantum kernel part) according the
  /// the output option.
  void emitOutput();

private:
  clang::CompilerInstance &ci;
  std::string inFilename;
  llvm::SmallVector<clang::ASTConsumer *, 2> consumers;
  nvqpp::OutputGen outputAction;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::MLIRContext *context;
  MangledKernelNamesMap &kernelNames;
  std::string optPipeline;
  std::unique_ptr<llvm::Module> llvmModule;
  std::unique_ptr<llvm::LLVMContext> llvmContext;
};

/// Action to create both the LLVM IR for the entire C++ compilation unit and
/// to translate the CUDA Quantum kernels.
template <typename ClangFEAction>
class CudaQAction : public ClangFEAction {
public:
  using Base = ClangFEAction;
  using MangledKernelNamesMap = cudaq::ASTBridgeAction::MangledKernelNamesMap;

  CudaQAction(clang::CompilerInstance &ci,
              mlir::OwningOpRef<mlir::ModuleOp> &module,
              mlir::MLIRContext *context, MangledKernelNamesMap &kernelNames,
              nvqpp::OutputGen genAction,
              const std::string &quakeOptPineline = "")
      : ci(ci), mlirAction(module, kernelNames), module(module),
        context(context), kernelNames(kernelNames), outputAction(genAction),
        optPipeline(quakeOptPineline) {}
  virtual ~CudaQAction() = default;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci,
                    llvm::StringRef inFile) override {
    auto llvmConsumer = this->Base::CreateASTConsumer(ci, inFile);
    auto mlirConsumer = mlirAction.CreateASTConsumer(ci, inFile);
    return std::make_unique<CudaQASTConsumer>(
        ci, inFile, llvmConsumer, mlirConsumer, outputAction, module, context,
        kernelNames, optPipeline);
  }

private:
  clang::CompilerInstance &ci;
  cudaq::ASTBridgeAction mlirAction;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::MLIRContext *context;
  MangledKernelNamesMap &kernelNames;
  nvqpp::OutputGen outputAction;
  std::string optPipeline;
};

} // namespace cudaq