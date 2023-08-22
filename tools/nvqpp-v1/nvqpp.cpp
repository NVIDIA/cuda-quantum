/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Support/Verifier.h"
#include "cudaq/Support/Version.h"
#include "nvqpp_config.h"
#include "nvqpp_diagnostics.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "nvqpp_driver.h"
#include <iostream>
extern "C" {
void get_this_executable_path() { return; }
}

namespace {
/// Consumer that runs a pair of AST consumers at the same time.
class CudaQASTConsumer : public clang::ASTConsumer {
public:
  CudaQASTConsumer(std::unique_ptr<clang::ASTConsumer> &consumer0,
                   std::unique_ptr<clang::ASTConsumer> &consumer1) {
    assert(consumer0 && consumer1 && "AST Consumers must be instantiated");
    consumers.emplace_back(consumer1.release());
    consumers.emplace_back(consumer0.release());
  }

  virtual ~CudaQASTConsumer() {
    for (auto *p : consumers)
      delete p;
    consumers.clear();
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
  llvm::SmallVector<clang::ASTConsumer *, 2> consumers;
};

/// Action to create both the LLVM IR for the entire C++ compilation unit and to
/// translate the CUDA Quantum kernels.
template <typename ClangFEAction>
class CudaQAction : public ClangFEAction {
public:
  using Base = ClangFEAction;
  using MangledKernelNamesMap = cudaq::ASTBridgeAction::MangledKernelNamesMap;

  CudaQAction(mlir::OwningOpRef<mlir::ModuleOp> &module,
              MangledKernelNamesMap &kernelNames)
      : mlirAction(module, kernelNames) {}
  virtual ~CudaQAction() = default;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci,
                    llvm::StringRef inFile) override {
    auto llvmConsumer = this->Base::CreateASTConsumer(ci, inFile);
    auto mlirConsumer = mlirAction.CreateASTConsumer(ci, inFile);
    return std::make_unique<CudaQASTConsumer>(llvmConsumer, mlirConsumer);
  }

private:
  cudaq::ASTBridgeAction mlirAction;
};
} // namespace
namespace cudaq {

void preprocess_arguments(argv_storage &args) {
  // const auto plugin_arg = "-Xclang";
  // auto is_plugin_argument = [&](auto it) {
  //   return llvm::StringRef(*std::prev(it)) == plugin_arg;
  // };

  // auto make_plugin_argument = [&](auto it) {
  //   if (is_plugin_argument(it))
  //     return it;
  //   return std::next(args.insert(it, plugin_arg));
  // };
}
static void error_handler(void *user_data, const char *msg,
                          bool get_crash_diag) {
  auto &diags = *static_cast<clang::DiagnosticsEngine *>(user_data);
  diags.Report(clang::diag::err_drv_command_failure) << msg;
  llvm::sys::RunInterruptHandlers();
  llvm::sys::Process::Exit(get_crash_diag ? 70 : 1);
}

std::unique_ptr<clang::FrontendAction> create_frontend_action(
    clang::CompilerInstance &ci, const cudaq_args &vargs,
    mlir::OwningOpRef<mlir::ModuleOp> &module,
    cudaq::ASTBridgeAction::MangledKernelNamesMap &cxx_mangled) {
  auto &opts = ci.getFrontendOpts();
  auto act = opts.ProgramAction;

  switch (act) {
  case (clang::frontend::ActionKind::ASTDump):
    // AST action: no need to invoke CUDAQ
    return std::make_unique<clang::ASTDumpAction>();
  case (clang::frontend::ActionKind::ASTPrint):
    // AST action: no need to invoke CUDAQ
    return std::make_unique<clang::ASTPrintAction>();
  case (clang::frontend::ActionKind::EmitAssembly):
    return std::make_unique<CudaQAction<clang::EmitAssemblyAction>>(
        module, cxx_mangled);
  case (clang::frontend::ActionKind::EmitBC):
    return std::make_unique<CudaQAction<clang::EmitBCAction>>(module,
                                                              cxx_mangled);
  case (clang::frontend::ActionKind::EmitLLVM):
    return std::make_unique<CudaQAction<clang::EmitLLVMAction>>(module,
                                                                cxx_mangled);
  case (clang::frontend::ActionKind::EmitObj):
    return std::make_unique<CudaQAction<clang::EmitObjAction>>(module,
                                                               cxx_mangled);
  default:
    throw std::runtime_error("Not supported!!!");
  }

  return nullptr;
}

bool execute_compiler_invocation(clang::CompilerInstance *ci,
                                 const cudaq_args &vargs) {
  auto &opts = ci->getFrontendOpts();

  // -help.
  if (opts.ShowHelp) {
    clang::driver::getDriverOptTable().printHelp(
        llvm::outs(), "nvq++ -cc1 [options] file...",
        "NVQ++ Compiler: https://github.com/NVIDIA/cuda-quantum",
        /*Include=*/clang::driver::options::CC1Option,
        /*Exclude=*/0, /*ShowAllAliases=*/false);
    return true;
  }

  // -version.
  //
  // FIXME: Use a better -version message?
  if (opts.ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return true;
  }

  ci->LoadRequestedPlugins();

  // If there were errors in processing arguments, don't do anything else.
  if (ci->getDiagnostics().hasErrorOccurred())
    return false;
  cudaq::ASTBridgeAction::MangledKernelNamesMap mangledKernelNameMap;
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<cudaq::cc::CCDialect, quake::QuakeDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OpBuilder builder(&context);
  auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(moduleOp);
  // Create and execute the frontend action.
  auto action =
      create_frontend_action(*ci, vargs, module, mangledKernelNameMap);
  if (!action)
    return false;

  bool success = ci->ExecuteAction(*action);

  // module->dump();
  const auto outputFilename = [&]() -> std::string {
    for (const auto &feInput : opts.Inputs) {
      if (feInput.isFile()) {
        llvm::SmallString<128> inputFile =
            llvm::sys::path::filename(feInput.getFile());
        return std::string(inputFile.c_str()) + ".qke";
      }
    }
    return "";
  }();

  assert(!outputFilename.empty());

  std::error_code ec;
  llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Failed to open output file '" << outputFilename << "'\n";
    return ec.value();
  }
  constexpr static const char mangledKernelNameMapAttrName[] =
      "quake.mangled_name_map";
  if (!moduleOp.getBody()->empty()) {
    if (!mangledKernelNameMap.empty()) {
      llvm::SmallVector<mlir::NamedAttribute> names;
      for (auto [key, value] : mangledKernelNameMap)
        names.emplace_back(mlir::StringAttr::get(&context, key),
                           mlir::StringAttr::get(&context, value));
      auto mapAttr = mlir::DictionaryAttr::get(&context, names);
      moduleOp->setAttr(mangledKernelNameMapAttrName, mapAttr);
    }

    // Running the verifier to make it easier to track down errors.
    mlir::PassManager pm(&context);
    pm.addPass(std::make_unique<cudaq::VerifierPass>());
    if (failed(pm.run(moduleOp))) {
      moduleOp->dump();
      llvm::errs() << "Passes failed!\n";
    } else {
      mlir::OpPrintingFlags opf;
      opf.enableDebugInfo(/*enable=*/true,
                          /*pretty=*/false);
      moduleOp.print(out.os(), opf);
      out.os() << '\n';
      out.keep();
    }
  }

  if (opts.DisableFree) {
    llvm::BuryPointer(std::move(action));
  }

  return success;
}

int cc1(const cudaq_args &vargs, argv_t ccargs, arg_t tool, void *main_addr) {
  auto comp = std::make_unique<clang::CompilerInstance>();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  cudaq::buffered_diagnostics diags(ccargs);
  bool success = clang::CompilerInvocation::CreateFromArgs(
      comp->getInvocation(), ccargs, diags.engine, tool);

  auto &frontend_opts = comp->getFrontendOpts();
  auto &header_opts = comp->getHeaderSearchOpts();

  if (frontend_opts.TimeTrace || !frontend_opts.TimeTracePath.empty()) {
    frontend_opts.TimeTrace = 1;
    llvm::timeTraceProfilerInitialize(frontend_opts.TimeTraceGranularity, tool);
  }

  // Infer the builtin include path if unspecified.
  if (header_opts.UseBuiltinIncludes && header_opts.ResourceDir.empty()) {
    header_opts.ResourceDir =
        clang::CompilerInvocation::GetResourcesPath(tool, main_addr);
  }

  // Create the actual diagnostics engine.
  if (comp->createDiagnostics(); !comp->hasDiagnostics()) {
    return 1;
  }

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(
      error_handler, static_cast<void *>(&comp->getDiagnostics()));

  diags.flush();
  if (!success) {
    comp->getDiagnosticClient().finish();
    return 1;
  }

  // Execute the frontend actions.
  try {
    llvm::TimeTraceScope TimeScope("ExecuteCompiler");
    success = execute_compiler_invocation(comp.get(), vargs);
  } catch (...) {
    comp->setSema(nullptr);
    comp->setASTConsumer(nullptr);
    comp->clearOutputFiles(true);
    throw;
  }

  // If any timers were active but haven't been destroyed yet, print their
  // results now.
  llvm::TimerGroup::printAll(llvm::errs());
  llvm::TimerGroup::clearAll();

  using small_string = llvm::SmallString<128>;

  if (llvm::timeTraceProfilerEnabled()) {
    small_string path(frontend_opts.OutputFile);
    llvm::sys::path::replace_extension(path, "json");

    if (!frontend_opts.TimeTracePath.empty()) {
      small_string trace_path(frontend_opts.TimeTracePath);
      if (llvm::sys::fs::is_directory(trace_path)) {
        llvm::sys::path::append(trace_path, llvm::sys::path::filename(path));
      }

      path.assign(trace_path);
    }

    if (auto profiler_output = comp->createOutputFile(
            path.str(), /*Binary=*/false, /*RemoveFileOnSignal=*/false,
            /*useTemporary=*/false)) {
      llvm::timeTraceProfilerWrite(*profiler_output);
      profiler_output.reset();
      llvm::timeTraceProfilerCleanup();
      comp->clearOutputFiles(false);
    }
  }
  llvm::remove_fatal_error_handler();

  // When running with -disable-free, don't do any destruction or shutdown.
  if (frontend_opts.DisableFree) {
    llvm::BuryPointer(std::move(comp));
    return !success;
  }

  return !success;
}
std::pair<cudaq_args, argv_storage> filter_args(const argv_storage_base &args) {
  cudaq_args vargs;
  argv_storage rest;
  // TODO: just an idea
  for (auto arg : args) {
    if (std::string_view(arg).starts_with("cudaq-")) {
      vargs.push_back(arg);
    } else {
      rest.push_back(arg);
    }
  }

  return {vargs, rest};
}
} // namespace cudaq
static int execute_cc1_tool(argv_storage_base &cmd_args) {
  llvm::cl::ResetAllOptionOccurrences();
  llvm::BumpPtrAllocator pointer_allocator;
  llvm::StringSaver saver(pointer_allocator);
  llvm::cl::ExpandResponseFiles(saver, &llvm::cl::TokenizeGNUCommandLine,
                                cmd_args);

  llvm::StringRef tool = cmd_args[1];

  void *get_executable_path_ptr = (void *)(intptr_t)get_this_executable_path;

  auto [vargs, ccargs] = cudaq::filter_args(cmd_args);

  if (tool == "-cc1") {
    auto ccargs_ref = llvm::ArrayRef(ccargs).slice(2);
    return cudaq::cc1(vargs, ccargs_ref, cmd_args[0], get_executable_path_ptr);
  }

  llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
               << "Valid tools include '-cc1'.\n";
  return 1;
}
int main(int argc, char **argv) {
  try {
    // Initialize variables to call the driver
    llvm::InitLLVM x(argc, argv);
    argv_storage cmd_args(argv, argv + argc);
    if (llvm::sys::Process::FixupStandardFileDescriptors()) {
      return 1;
    }

    llvm::InitializeAllTargets();

    llvm::BumpPtrAllocator pointer_allocator;
    llvm::StringSaver saver(pointer_allocator);

    // TODO: support both modes: now just do MLIR

    // Check if cudaq-front is in the frontend mode
    auto first_arg = llvm::find_if(llvm::drop_begin(cmd_args),
                                   [](auto a) { return a != nullptr; });
    if (first_arg != cmd_args.end()) {
      if (std::string_view(cmd_args[1]).starts_with("-cc1")) {
        return execute_cc1_tool(cmd_args);
      }
    }

    void *p = (void *)(intptr_t)get_this_executable_path;
    const std::string driver_path =
        llvm::sys::fs::getMainExecutable(cmd_args[0], p);
    // std::cout << "Driver path: " << driver_path << "\n";
    cudaq::preprocess_arguments(cmd_args);
    cudaq::driver driver(driver_path, cmd_args, &execute_cc1_tool);
    return driver.execute();
  } catch (std::exception &e) {
    llvm::errs() << "error: " << e.what() << '\n';
    std::exit(1);
  }
}