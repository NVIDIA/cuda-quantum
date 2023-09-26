/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// This tool takes a snippet of CUDA Quantum code, translates it to the Quake
/// dialect, runs user-specified passes, and prints the result. (Note that Quake
/// is a "minimalist" dialect, so translating to Quake can and does make use of
/// other MLIR dialects as well. These other dialects are convenient but not the
/// salient part of this tool.)

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Support/Verifier.h"
#include "cudaq/Support/Version.h"
#include "nvqpp_config.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
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
#include <filesystem>
#include <sstream>

using namespace llvm;

constexpr static const char toolName[] = "cudaq-quake";
constexpr static const char mangledKernelNameMapAttrName[] =
    "quake.mangled_name_map";

//===----------------------------------------------------------------------===//
// Command line options.
//===----------------------------------------------------------------------===//

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::list<std::string>
    kernelNames("filter", cl::desc("Names of quantum kernels to convert."));

static cl::opt<bool>
    emitLLVM("emit-llvm-file",
             cl::desc("Emit the LLVM IR for the C++ input to <input>.ll file."),
             cl::init(false));

static cl::opt<bool> llvmOnly(
    "llvm-only",
    cl::desc("Emit the LLVM IR for the C++ input to <input>.ll file and do not "
             "emit the MLIR code. --emit-llvm-file has higher precedence."),
    cl::init(false));

static cl::opt<bool>
    verifyMode("verify", cl::desc("Run in diagnostic verification mode."),
               cl::init(false));

static cl::opt<bool> noSimplify(
    "no-simplify",
    cl::desc("Disable passes to simplify the code after the bridge."),
    cl::init(false));

static cl::opt<bool> astDump("ast-dump", cl::desc("Dump the ast."),
                             cl::init(false));

static cl::opt<bool> showVersion("nvqpp-version",
                                 cl::desc("Print the version."),
                                 cl::init(false));

static cl::opt<bool> verboseClang("v",
                                  cl::desc("Add -v to clang tool arguments."),
                                  cl::init(false));

static cl::opt<bool> debugMode("g", cl::desc("Add -g to clang tool arguments."),
                               cl::init(false));

static cl::opt<bool> dumpToStderr("llvm-to-stderr",
                                  cl::desc("Echo the LLVM IR to stderr"),
                                  cl::init(false));

static cl::opt<std::string>
    resourceDir("resource-dir", cl::desc("Specify output filename"),
                cl::init(LLVM_ROOT "/lib/clang/" CUDAQ_LLVM_VERSION));

static cl::list<std::string>
    macroDefines("D", cl::desc("Define preprocessor macro."));

static cl::list<std::string> includePath("I", cl::desc("Include file path."));

static cl::list<std::string>
    systemIncludePath("J", cl::desc("System include file path."));

static cl::list<std::string>
    extraClangArgs("Xcudaq", cl::desc("Extra options to pass to clang++"));

inline bool isStdinInput(StringRef str) { return str == "-"; }

//===----------------------------------------------------------------------===//
// Helper classes.
//===----------------------------------------------------------------------===//

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
  SmallVector<clang::ASTConsumer *, 2> consumers;
};

/// Action to create both the LLVM IR for the entire C++ compilation unit and to
/// translate the CUDA Quantum kernels.
class CudaQAction : public clang::EmitLLVMAction {
public:
  using Base = clang::EmitLLVMAction;
  using MangledKernelNamesMap = cudaq::ASTBridgeAction::MangledKernelNamesMap;

  CudaQAction(mlir::OwningOpRef<mlir::ModuleOp> &module,
              MangledKernelNamesMap &kernelNames)
      : mlirAction(module, kernelNames) {}
  virtual ~CudaQAction() = default;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci, StringRef inFile) override {
    auto llvmConsumer = this->Base::CreateASTConsumer(ci, inFile);
    auto mlirConsumer = mlirAction.CreateASTConsumer(ci, inFile);
    return std::make_unique<CudaQASTConsumer>(llvmConsumer, mlirConsumer);
  }

private:
  cudaq::ASTBridgeAction mlirAction;
};

/// This is the front-end action to convert the C++ code to LLVM IR to a file.
class InterceptCudaQAction : public clang::EmitLLVMAction {
public:
  using MangledKernelNamesMap = cudaq::ASTBridgeAction::MangledKernelNamesMap;

  InterceptCudaQAction(mlir::OwningOpRef<mlir::ModuleOp> &,
                       MangledKernelNamesMap &)
      : clang::EmitLLVMAction{} {}
  virtual ~InterceptCudaQAction() = default;

  void EndSourceFileAction() override {
    clang::EmitLLVMAction::EndSourceFileAction();
    // Dump the IR to stderr if the command-line option was given.
    if (dumpToStderr)
      takeModule().get()->print(errs(), nullptr, false, true);
  }
};
} // namespace

template <typename ACTION>
bool runTool(mlir::OwningOpRef<mlir::ModuleOp> &module,
             CudaQAction::MangledKernelNamesMap &mangledKernelNameMap,
             StringRef cplusplusCode, std::vector<std::string> &clArgs,
             const std::string &inputFileName) {
  assert(cplusplusCode.size() > 0);
  assert(clArgs.size() > 0);
  if (!clang::tooling::runToolOnCodeWithArgs(
          std::make_unique<ACTION>(module, mangledKernelNameMap), cplusplusCode,
          clArgs, inputFileName, toolName)) {
    errs() << "error: could not translate ";
    if (inputFilename == "-")
      errs() << "input";
    else
      errs() << "file '" << inputFilename;
    errs() << '\n';
    return true; // failed
  }
  return false;
}

/// @brief Retrieve the path of this executable, borrowed from
/// the Clang Driver
std::string getExecutablePath(const char *argv0, bool canonicalPrefixes) {
  if (!canonicalPrefixes) {
    SmallString<128> executablePath(argv0);
    if (!llvm::sys::fs::exists(executablePath))
      if (llvm::ErrorOr<std::string> p =
              llvm::sys::findProgramByName(executablePath))
        executablePath = *p;
    return std::string(executablePath.str());
  }
  void *p = (void *)(intptr_t)getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, p);
}

//===----------------------------------------------------------------------===//
// Main entry point into the cudaq-quake tool.
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues on
  // nvidia/cuda-quantum
  llvm::setBugReportMsg(cudaq::bugReportMsg);

  // We need the location of this cudaq-quake executable so that we can get the
  // install path
  std::string executablePath = getExecutablePath(argv[0], true);
  std::filesystem::path cudaqQuakePath{executablePath};
  auto installBinPath = cudaqQuakePath.parent_path();
  auto cudaqInstallPath = installBinPath.parent_path();

  // Default to the internal resource-dir in the absence of
  // the one in the LLVM_BINARY_DIR
  std::filesystem::path resourceDirPath{resourceDir.getValue()};
  if (!std::filesystem::exists(resourceDirPath))
    resourceDirPath = cudaqInstallPath / "lib" / "clang" / CUDAQ_LLVM_VERSION;

  if (!std::filesystem::exists(resourceDirPath)) {
    llvm::errs() << "Could not find a valid clang resource-dir.\n";
    return 1;
  }

  // Process the command-line options, including reading in a file.
  [[maybe_unused]] llvm::InitLLVM unused(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, toolName);
  if (showVersion)
    llvm::errs() << "nvq++ Version " << cudaq::getVersion() << '\n';
  ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrError =
      MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrError.getError()) {
    errs() << "Could not open file: " << ec.message() << '\n';
    return ec.value();
  }

  mlir::registerAllPasses();

  // Read the code into a memory buffer and setup MLIR.
  auto cplusplusCode = fileOrError.get()->getBuffer();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<cudaq::cc::CCDialect, quake::QuakeDialect>();
  mlir::MLIRContext context(registry);
  // TODO: Consider only loading the dialects we know we'll use.
  context.loadAllAvailableDialects();
  mlir::OpBuilder builder(&context);
  auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(moduleOp);
  // Register trivial diagnostic handler so that notes, warnings, and
  // remark messages are not elided by default.
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    const char *severity = "";
    switch (diag.getSeverity()) {
    case mlir::DiagnosticSeverity::Note:
      severity = "note";
      break;
    case mlir::DiagnosticSeverity::Warning:
      severity = "warning";
      break;
    case mlir::DiagnosticSeverity::Remark:
      severity = "remark";
      break;
    case mlir::DiagnosticSeverity::Error:
      severity = "error";
      break;
    }
    llvm::errs() << diag.getLocation() << ':' << severity << ": " << diag.str()
                 << '\n';
  });

  // Process arguments.
  std::vector<std::string> clArgs = {"-std=c++20", "-resource-dir",
                                     resourceDirPath.string()};
  if (verboseClang)
    clArgs.push_back("-v");

  if (debugMode)
    clArgs.push_back("-g");

  // Update the include path.
  for (auto &path : systemIncludePath) {
    clArgs.push_back("-isystem");
    clArgs.push_back(path);
  }

  // `cudaq-quake` is a clang tool, and thus it will search for C++ headers
  // using clang builtin paths, i.e., it will look for `../include/c++/v1` and
  // fallback to the system paths.  Since this tool is not installed with clang,
  // this is the wrong thing to do.  This tools needs to look for
  // `${LLVM_ROOT}/include/c++/v1` and then fallback to system paths.
  //
  // I have not found a way to change the builtin search paths for a particular
  // tool.  So the workaround involves checking whether
  // `${LLVM_ROOT}/include/c++/v1` exists, and forcing the tool to use it:
  if (std::filesystem::exists(LLVM_LIBCXX_INCLUDE_DIR)) {
    clArgs.push_back("-stdlib++-isystem");
    clArgs.push_back(LLVM_LIBCXX_INCLUDE_DIR);
  }

  // If the cudaq.h does not exist in the installation directory,
  // fallback onto the source install.
  std::filesystem::path cudaqIncludeDir = cudaqInstallPath / "include";
  auto cudaqHeader = cudaqIncludeDir / "cudaq.h";
  if (!std::filesystem::exists(cudaqHeader))
    // need to fall back to the build environment.
    cudaqIncludeDir = std::string(FALLBACK_CUDAQ_INCLUDE_DIR);

  // One final check here, do we have this header,
  // if not we cannot proceed.
  if (!std::filesystem::exists(cudaqIncludeDir / "cudaq.h")) {
    llvm::errs() << "Invalid CUDA Quantum install configuration, cannot find "
                    "CUDA Quantum "
                    "include directory.\n";
    return 1;
  }

  // Attach the relative path of the file, if any. The directory the file
  // resides in has highest priority.
  std::string relativePath = sys::path::parent_path(inputFilename).str();
  if (!relativePath.empty())
    clArgs.push_back("-I" + relativePath);

  for (auto &path : includePath)
    clArgs.push_back("-I" + path);

  // Add the default path to the cudaq headers.
  clArgs.push_back("-I" + cudaqIncludeDir.string());

  // Add preprocessor macro definitions, if any.
  for (auto &def : macroDefines)
    clArgs.push_back("-D" + def);

  // Pass verify mode if requested.
  if (verifyMode) {
    clArgs.push_back("-Xclang");
    clArgs.push_back("-verify");
  }
  // Pass -ast-dump if requested.
  if (astDump) {
    clArgs.push_back("-Xclang");
    clArgs.push_back("-ast-dump");
  }

  for (auto &xtra : extraClangArgs)
    clArgs.push_back(xtra);

  // Allow a user to specify extra args for clang via
  // an environment variable.
  if (auto extraArgs = std::getenv("CUDAQ_CLANG_EXTRA_ARGS")) {
    std::stringstream ss;
    ss << extraArgs;
    std::string part;
    std::vector<std::string> localArgs;
    while (std::getline(ss, part, ' '))
      clArgs.push_back(part);
  }

  // Set the mangled kernel names map.
  CudaQAction::MangledKernelNamesMap mangledKernelNameMap;

  std::string inputFile = isStdinInput(inputFilename)
                              ? std::string("input.cc")
                              : sys::path::filename(inputFilename).str();
  if (auto rc = emitLLVM
                    ? runTool<CudaQAction>(module, mangledKernelNameMap,
                                           cplusplusCode, clArgs, inputFile)
                    : (llvmOnly ? runTool<InterceptCudaQAction>(
                                      module, mangledKernelNameMap,
                                      cplusplusCode, clArgs, inputFile)
                                : runTool<cudaq::ASTBridgeAction>(
                                      module, mangledKernelNameMap,
                                      cplusplusCode, clArgs, inputFile)))
    return rc;

  // Success! Dump the IR and exit.
  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);
  if (ec) {
    errs() << "Failed to open output file '" << outputFilename << "'\n";
    return ec.value();
  }
  if (!moduleOp.getBody()->empty()) {
    if (!mangledKernelNameMap.empty()) {
      SmallVector<mlir::NamedAttribute> names;
      for (auto [key, value] : mangledKernelNameMap)
        names.emplace_back(mlir::StringAttr::get(&context, key),
                           mlir::StringAttr::get(&context, value));
      auto mapAttr = mlir::DictionaryAttr::get(&context, names);
      moduleOp->setAttr(mangledKernelNameMapAttrName, mapAttr);
    }

    // Running the verifier to make it easier to track down errors.
    // The canonicalizer and symbol DCE passes cleanup unused artifacts
    // generated by the bridge.
    mlir::PassManager pm(&context);
    pm.addPass(std::make_unique<cudaq::VerifierPass>());
    if (!noSimplify) {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createSymbolDCEPass());
    }
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

  return 0;
}
