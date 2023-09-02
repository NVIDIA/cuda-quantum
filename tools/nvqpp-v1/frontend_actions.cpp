/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "frontend_actions.h"

namespace cudaq {

CudaQASTConsumer::CudaQASTConsumer(
    clang::CompilerInstance &ci, llvm::StringRef inFile,
    std::unique_ptr<clang::ASTConsumer> &consumer0,
    std::unique_ptr<clang::ASTConsumer> &consumer1, nvqpp::OutputGen genAction,
    mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext *context,
    MangledKernelNamesMap &kernelNames, const std::string &quakeOptPineline)
    : ci(ci), inFilename(llvm::sys::path::filename(inFile)),
      outputAction(genAction), module(module), context(context),
      kernelNames(kernelNames), optPipeline(quakeOptPineline) {
  assert(consumer0 && consumer1 && "AST Consumers must be instantiated");
  consumers.emplace_back(consumer1.release());
  consumers.emplace_back(consumer0.release());
}

CudaQASTConsumer::~CudaQASTConsumer() {
  for (auto *p : consumers)
    delete p;
  consumers.clear();
  llvm::BuryPointer(std::move(llvmModule));
}

void CudaQASTConsumer::HandleTranslationUnit(clang::ASTContext &ctxt) {
  applyConsumers(&clang::ASTConsumer::HandleTranslationUnit, std::move(ctxt));
  // entire translation unit have been parsed => output
  emitOutput();
}
void CudaQASTConsumer::HandleCXXStaticMemberVarInstantiation(
    clang::VarDecl *VD) {
  applyConsumers(&clang::ASTConsumer::HandleCXXStaticMemberVarInstantiation,
                 std::move(VD));
}
void CudaQASTConsumer::Initialize(clang::ASTContext &Ctx) {
  applyConsumers(&clang::ASTConsumer::Initialize, std::move(Ctx));
}
bool CudaQASTConsumer::HandleTopLevelDecl(clang::DeclGroupRef D) {
  bool result = true;
  for (auto *c : consumers)
    result = result && c->HandleTopLevelDecl(D);
  return result;
}
void CudaQASTConsumer::HandleInlineFunctionDefinition(clang::FunctionDecl *D) {
  applyConsumers(&clang::ASTConsumer::HandleInlineFunctionDefinition,
                 std::move(D));
}
void CudaQASTConsumer::HandleInterestingDecl(clang::DeclGroupRef D) {
  applyConsumers(&clang::ASTConsumer::HandleInterestingDecl, std::move(D));
}
void CudaQASTConsumer::HandleTagDeclDefinition(clang::TagDecl *D) {
  applyConsumers(&clang::ASTConsumer::HandleTagDeclDefinition, std::move(D));
}
void CudaQASTConsumer::HandleTagDeclRequiredDefinition(
    const clang::TagDecl *D) {
  applyConsumers(&clang::ASTConsumer::HandleTagDeclRequiredDefinition,
                 std::move(D));
}
void CudaQASTConsumer::CompleteTentativeDefinition(clang::VarDecl *D) {
  applyConsumers(&clang::ASTConsumer::CompleteTentativeDefinition,
                 std::move(D));
}
void CudaQASTConsumer::CompleteExternalDeclaration(clang::VarDecl *D) {
  applyConsumers(&clang::ASTConsumer::CompleteExternalDeclaration,
                 std::move(D));
}
void CudaQASTConsumer::AssignInheritanceModel(clang::CXXRecordDecl *RD) {
  applyConsumers(&clang::ASTConsumer::AssignInheritanceModel, std::move(RD));
}
void CudaQASTConsumer::HandleVTable(clang::CXXRecordDecl *RD) {
  applyConsumers(&clang::ASTConsumer::HandleVTable, std::move(RD));
}

void CudaQASTConsumer::applyOpt() {
  mlir::PassManager pm(context);
  std::string errMsg;
  llvm::raw_string_ostream os(errMsg);
  // Parse the pipeline and apply
  if (mlir::failed(mlir::parsePassPipeline(optPipeline, pm, os)))
    throw std::runtime_error("Failed to add passes to pipeline (" + errMsg +
                             ").");
  if (mlir::failed(pm.run(module.get())))
    throw std::runtime_error("Quake optimization failed.");
}

void CudaQASTConsumer::applyTranslate() {
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
  auto optPipeline = mlir::makeOptimizingTransformer(optLevel, sizeLevel,
                                                     /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << '\n';
    std::exit(1);
  }
}

void CudaQASTConsumer::emitOutput() {
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
      // If only requested MLIR (Quake) output, write the MLIR output then exit
      auto outputStream =
          ci.createOutputFile(inFilename + ".qke", false, true, false);
      mlir::OpPrintingFlags opf;
      opf.enableDebugInfo(/*enable=*/true,
                          /*pretty=*/false);
      module.get().print(*outputStream, opf);
      return;
    }

    // Further lowering is needed:
    // Apply cudaq-opt
    applyOpt();
    // MLIR -> LLVM translate
    applyTranslate();
    // Are we emitting LLVM IR or Obj file?
    const auto emitAction = (outputAction == nvqpp::OutputGen::EmitLlvm)
                                ? clang::BackendAction::Backend_EmitLL
                                : clang::BackendAction::Backend_EmitObj;

    const std::string fileNameSuffix =
        (outputAction == nvqpp::OutputGen::EmitLlvm) ? ".qke.ll" : ".qke.o";
    const bool isBinaryFile =
        (outputAction == nvqpp::OutputGen::EmitLlvm) ? false : true;
    // Emit (write to file)
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
} // namespace cudaq