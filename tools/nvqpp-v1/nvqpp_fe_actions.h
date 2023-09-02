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
                   const std::string &quakeOptPineline = "")
      : ci(ci), inFilename(llvm::sys::path::filename(inFile)),
        outputAction(genAction), module(module), context(context),
        kernelNames(kernelNames), optPineline(quakeOptPineline) {
    assert(consumer0 && consumer1 && "AST Consumers must be instantiated");
    consumers.emplace_back(consumer1.release());
    consumers.emplace_back(consumer0.release());
  }

  virtual ~CudaQASTConsumer() {
    for (auto *p : consumers)
      delete p;
    consumers.clear();
    llvm::BuryPointer(std::move(llvmModule));
  }

  // The following is boilerplate to override all the virtual functions that
  // appear in the union of the two AST consumers, namely clang::BackendConsumer
  // and nvqpp::MLIRASTConsumer.
  template <typename A, typename B>
  inline void applyConsumers(void (clang::ASTConsumer::*fun)(A), B &&arg) {
    for (auto *c : consumers)
      (c->*fun)(arg);
  }
  void HandleTranslationUnit(clang::ASTContext &ctxt) override {
    applyConsumers(&clang::ASTConsumer::HandleTranslationUnit, std::move(ctxt));
    // entire translation unit have been parsed => output
    emitOutput();
  }
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override {
    applyConsumers(&clang::ASTConsumer::HandleCXXStaticMemberVarInstantiation,
                   std::move(VD));
  }
  void Initialize(clang::ASTContext &Ctx) override {
    applyConsumers(&clang::ASTConsumer::Initialize, std::move(Ctx));
  }
  bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
    bool result = true;
    for (auto *c : consumers)
      result = result && c->HandleTopLevelDecl(D);
    return result;
  }
  void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override {
    applyConsumers(&clang::ASTConsumer::HandleInlineFunctionDefinition,
                   std::move(D));
  }
  void HandleInterestingDecl(clang::DeclGroupRef D) override {
    applyConsumers(&clang::ASTConsumer::HandleInterestingDecl, std::move(D));
  }
  void HandleTagDeclDefinition(clang::TagDecl *D) override {
    applyConsumers(&clang::ASTConsumer::HandleTagDeclDefinition, std::move(D));
  }
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override {
    applyConsumers(&clang::ASTConsumer::HandleTagDeclRequiredDefinition,
                   std::move(D));
  }
  void CompleteTentativeDefinition(clang::VarDecl *D) override {
    applyConsumers(&clang::ASTConsumer::CompleteTentativeDefinition,
                   std::move(D));
  }
  void CompleteExternalDeclaration(clang::VarDecl *D) override {
    applyConsumers(&clang::ASTConsumer::CompleteExternalDeclaration,
                   std::move(D));
  }
  void AssignInheritanceModel(clang::CXXRecordDecl *RD) override {
    applyConsumers(&clang::ASTConsumer::AssignInheritanceModel, std::move(RD));
  }
  void HandleVTable(clang::CXXRecordDecl *RD) override {
    applyConsumers(&clang::ASTConsumer::HandleVTable, std::move(RD));
  }

private:
  void applyOpt() {
    mlir::PassManager pm(context);
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    if (mlir::failed(mlir::parsePassPipeline(optPineline, pm, os)))
      throw std::runtime_error("Failed to add passes to pipeline (" + errMsg +
                               ").");
    if (mlir::failed(pm.run(module.get())))
      throw std::runtime_error("Quake optimization failed.");
  }

  void applyTranslate() {
    const auto addPipelineToQIR = [](mlir::PassManager &pm,
                                     bool baseProfile = false) {
      using namespace mlir;
      cudaq::opt::addAggressiveEarlyInlining(pm);
      pm.addPass(createCanonicalizerPass());
      pm.addPass(cudaq::opt::createExpandMeasurementsPass());
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLoweringPass());
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createLowerToCFGPass());
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize());
      pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopUnroll());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      pm.addNestedPass<func::FuncOp>(
          cudaq::opt::createCombineQuantumAllocations());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      pm.addPass(cudaq::opt::createConvertToQIRPass());
      if (baseProfile) {
        cudaq::opt::addBaseProfilePipeline(pm);
      }
    };
    mlir::PassManager pm(context);
    addPipelineToQIR(pm);

    if (mlir::failed(pm.run(module.get())))
      throw("pipeline failed");

    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerLLVMDialectTranslation(*context);

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvmContext = std::make_unique<llvm::LLVMContext>();
    llvmContext->setOpaquePointers(false);
    llvmModule = mlir::translateModuleToLLVMIR(module.get(), *llvmContext);
    if (!llvmModule)
      throw("Failed to emit LLVM IR");

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    // Optionally run an optimization pipeline over the llvm module.
    // Get the level from the compiler instance
    auto &codegenOpts = ci.getCodeGenOpts();
    const auto optLevel = codegenOpts.OptimizationLevel;
    const auto sizeLevel = codegenOpts.OptimizeSize;
    auto optPipeline =
        mlir::makeOptimizingTransformer(optLevel, sizeLevel,
                                        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
      llvm::errs() << "Failed to optimize LLVM IR " << err << '\n';
      std::exit(1);
    }
  }

  void emitOutput() {
    constexpr static const char mangledKernelNameMapAttrName[] =
        "quake.mangled_name_map";
    if (!module->getBody()->empty()) {
      if (!kernelNames.empty()) {
        llvm::SmallVector<mlir::NamedAttribute> names;
        for (auto [key, value] : kernelNames)
          names.emplace_back(mlir::StringAttr::get(context, key),
                             mlir::StringAttr::get(context, value));
        auto mapAttr = mlir::DictionaryAttr::get(context, names);
        module.get()->setAttr(mangledKernelNameMapAttrName, mapAttr);
      }

      // Running the verifier to make it easier to track down errors.
      mlir::PassManager pm(context);
      pm.addPass(std::make_unique<cudaq::VerifierPass>());
      if (mlir::failed(pm.run(module.get()))) {
        module.get()->dump();
        llvm::errs() << "Passes failed!\n";
        exit(1);
      }
      if (outputAction == nvqpp::OutputGen::EmitMlir) {
        // Write the MLIR output then exit
        auto outputStream =
            ci.createOutputFile(inFilename + ".qke", false, true, false);
        mlir::OpPrintingFlags opf;
        opf.enableDebugInfo(/*enable=*/true,
                            /*pretty=*/false);
        module.get().print(*outputStream, opf);
        return;
      }
      // Apply cudaq-opt
      applyOpt();
      // MLIR -> LLVM translate
      applyTranslate();
      const auto emitAction = (outputAction == nvqpp::OutputGen::EmitLlvm)
                                  ? clang::BackendAction::Backend_EmitLL
                                  : clang::BackendAction::Backend_EmitObj;

      const std::string fileNameSuffix =
          (outputAction == nvqpp::OutputGen::EmitLlvm) ? ".qke.ll" : ".qke.o";
      const bool isBinaryFile =
          (outputAction == nvqpp::OutputGen::EmitLlvm) ? false : true;
      // Emit
      auto &headerSearchOpts = ci.getHeaderSearchOpts();
      auto &codegenOpts = ci.getCodeGenOpts();
      auto &targetOpts = ci.getTargetOpts();
      auto &langOpts = ci.getLangOpts();
      auto &diags = ci.getDiagnostics();
      auto &targetInfo = ci.getTarget();
      auto outputStream = ci.createOutputFile(inFilename + fileNameSuffix,
                                              isBinaryFile, true, false);
      clang::EmitBackendOutput(diags, headerSearchOpts, codegenOpts, targetOpts,
                               langOpts, targetInfo.getDataLayoutString(),
                               llvmModule.get(), emitAction,
                               std::move(outputStream));
    }
  }

private:
  clang::CompilerInstance &ci;
  std::string inFilename;
  llvm::SmallVector<clang::ASTConsumer *, 2> consumers;
  nvqpp::OutputGen outputAction;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::MLIRContext *context;
  MangledKernelNamesMap &kernelNames;
  std::string optPineline;
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
        optPineline(quakeOptPineline) {}
  virtual ~CudaQAction() = default;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci,
                    llvm::StringRef inFile) override {
    auto llvmConsumer = this->Base::CreateASTConsumer(ci, inFile);
    auto mlirConsumer = mlirAction.CreateASTConsumer(ci, inFile);
    return std::make_unique<CudaQASTConsumer>(
        ci, inFile, llvmConsumer, mlirConsumer, outputAction, module, context,
        kernelNames, optPineline);
  }

private:
  clang::CompilerInstance &ci;
  cudaq::ASTBridgeAction mlirAction;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::MLIRContext *context;
  MangledKernelNamesMap &kernelNames;
  nvqpp::OutputGen outputAction;
  std::string optPineline;
};

} // namespace cudaq