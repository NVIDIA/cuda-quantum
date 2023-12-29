/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// The nvq++-pp tool provides a preprocessor on CUDA Quantum C++ source files
/// that are to be compiled in library mode. The preprocessor provides an
/// opportunity for source code rewrites and specification enforcement /
/// diagnostics.

#include "common/FmtCore.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>

#include "nvqpp_config.h"

#include "KernelCharacteristics.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"

#include "AnalysisPlugin.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"

LLVM_INSTANTIATE_REGISTRY(cudaq::AnalysisPlugin::RegistryType)

using namespace clang;

constexpr static const char toolName[] = "nvq++-pp";

static llvm::cl::opt<bool>
    removeComments("remove-comments",
                   llvm::cl::desc("Run in diagnostic verification mode."),
                   llvm::cl::init(false));

static llvm::cl::opt<bool>
    verifyMode("verify", llvm::cl::desc("Run in diagnostic verification mode."),
               llvm::cl::init(false));

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Specify output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    resourceDir("resource-dir", llvm::cl::desc("Specify output filename"),
                llvm::cl::init(LLVM_ROOT "/lib/clang/" CUDAQ_LLVM_VERSION));

static llvm::cl::list<std::string>
    includePath("I", llvm::cl::desc("Include file path."));

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

class NvqppPreprocessorConsumer : public clang::ASTConsumer {
protected:
  /// @brief Reference to the Clang Rewriter
  Rewriter &rewriter;

  /// @brief Utility visitor for collecting pertinent CUDA Quantum kernel
  /// and host `FunctionDecl` instances.
  struct CollectQuantumKernelsAndHostFunctions
      : public RecursiveASTVisitor<CollectQuantumKernelsAndHostFunctions> {
    Rewriter &rewriter;
    std::vector<FunctionDecl *> kernelFunctions;
    std::vector<FunctionDecl *> hostFunctions;
    CollectQuantumKernelsAndHostFunctions(Rewriter &r) : rewriter(r) {}

    /// Find CUDA Quantum lambda kernels
    bool VisitLambdaExpr(LambdaExpr *lambda) {
      if (const auto *cxxMethodDecl = lambda->getCallOperator())
        if (const auto *f = cxxMethodDecl->getAsFunction()->getDefinition())
          if (auto attr = f->getAttr<clang::AnnotateAttr>())
            if (attr->getAnnotation().str() == "quantum")
              kernelFunctions.push_back(const_cast<FunctionDecl *>(f));

      return true;
    }
    bool VisitFunctionDecl(FunctionDecl *decl) {
      // add host functions
      if (auto attr = decl->getAttr<clang::AnnotateAttr>()) {
        if (attr->getAnnotation().str() == "quantum") {
          kernelFunctions.push_back(decl);
          return true;
        }
      }

      if (rewriter.getSourceMgr().isInMainFile(decl->getBeginLoc()))
        hostFunctions.push_back(decl);

      return true;
    }
  };

public:
  NvqppPreprocessorConsumer(Rewriter &r) : rewriter(r) {}
  void HandleTranslationUnit(ASTContext &context) override {
    // Create the diagnostic engine
    auto &de = context.getDiagnostics();

    // Collect quantum kernels and host functions in the main file
    CollectQuantumKernelsAndHostFunctions vis(rewriter);
    vis.TraverseDecl(context.getTranslationUnitDecl());

    // Create a KernelCharacteristics instance for every kernel, store
    // them in a map with their corresponding `FunctionDecl` as the key
    std::map<FunctionDecl *, cudaq::KernelCharacteristics> characteristics;
    for (auto *decl : vis.kernelFunctions)
      characteristics.emplace(decl, cudaq::KernelCharacteristics());

    // Set the characteristics for the kernel, if we haven't already
    for (auto *d : vis.kernelFunctions)
      if (characteristics[d].name.empty())
        cudaq::ApplyQuantumKernelCharacteristics(characteristics[d])
            .TraverseDecl(d);

    // Look at the host functions and see if there are any other
    // kernel characteristics we can specify from them (e.g. called from
    // observe())
    for (auto *d : vis.hostFunctions)
      cudaq::ApplyHostFunctionCharacteristics(characteristics).TraverseDecl(d);

    // Loop over all kernels and apply any analysis plugins we have
    for (auto *decl : vis.kernelFunctions)
      for (auto iter = cudaq::AnalysisPlugin::RegistryType::begin();
           iter != cudaq::AnalysisPlugin::RegistryType::end(); ++iter)
        iter->instantiate()->traverseTree({characteristics[decl], decl}, de,
                                          rewriter);
  }
};

/// @brief Interface with the Clang Frontend via this sub-type.
/// Return our own custom `ASTConsumer`.
class NvqppPreprocessorAction : public clang::PluginASTAction {
protected:
  Rewriter &rewriter;

public:
  NvqppPreprocessorAction(Rewriter &r) : rewriter(r) {}
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci, StringRef inFile) override {
    auto &ctx = ci.getASTContext();
    rewriter.setSourceMgr(ctx.getSourceManager(), ctx.getLangOpts());
    return std::make_unique<NvqppPreprocessorConsumer>(rewriter);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

int main(int argc, char *argv[]) {

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
    llvm::errs() << "[nvq++-pp] Could not find a valid clang resource-dir.\n";
    return 1;
  }

  // Process the command-line options, including reading in a file.
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrError.getError()) {
    llvm::errs() << "[nvq++-pp] Could not open file: " << ec.message() << '\n';
    return ec.value();
  }

  std::filesystem::path cudaqIncludeDir = cudaqInstallPath / "include";
  auto cudaqHeader = cudaqIncludeDir / "cudaq.h";
  if (!std::filesystem::exists(cudaqHeader))
    // need to fall back to the build environment.
    cudaqIncludeDir = std::string(FALLBACK_CUDAQ_INCLUDE_DIR);

  // One final check here, do we have this header, if not we cannot proceed.
  if (!std::filesystem::exists(cudaqIncludeDir / "cudaq.h")) {
    llvm::errs()
        << "[nvq++-pp] Invalid CUDA Quantum install configuration, cannot find "
           "CUDA Quantum include directory.\n";
    return 1;
  }

  auto cudaqPCH = cudaqIncludeDir / "pch" / "libmode" / "cudaq.h.pch";
  if (!std::filesystem::exists(cudaqPCH))
    // need to fall back to the build environment.
    cudaqPCH = std::filesystem::path(std::string(FALLBACK_CUDAQ_PCH_DIR)) /
               "libmode" / "cudaq.h.pch";

  // One final check here, do we have this header, if not we cannot proceed.
  if (!std::filesystem::exists(cudaqPCH)) {
    llvm::errs()
        << "[nvq++-pp] Invalid CUDA Quantum install configuration, cannot find "
           "CUDA Quantum pre-compiled header.\n";
    return 1;
  }

  // Create the Rewriter
  Rewriter rewriter;

  // Read the code into a memory buffer and setup MLIR.
  auto cplusplusCode = fileOrError.get()->getBuffer();
  std::vector<std::string> clArgs{"-std=c++20", "-fsyntax-only", "-include-pch",
                                  cudaqPCH.string(),
                                  "-I" + cudaqIncludeDir.string()};
  if (verifyMode) {
    clArgs.push_back("-Xclang");
    clArgs.push_back("-verify");
  }
  for (auto &path : includePath)
    clArgs.push_back("-I" + path);
  if (!clang::tooling::runToolOnCodeWithArgs(
          std::make_unique<NvqppPreprocessorAction>(rewriter), cplusplusCode,
          clArgs, inputFilename, toolName)) {
    return -1;
  }

  const RewriteBuffer *rewriteBuffer =
      rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());

  // If rewriteBuffer is null, than we will always just output the inputFile
  // contents

  auto maybeRemoveComments = [&](const std::string &contents) {
    if (!removeComments)
      return contents;
    std::stringstream ss(contents.c_str()), newss;
    std::string line;
    while (std::getline(ss, line, '\n')) {
      StringRef tmp(line);
      if (tmp.contains("//"))
        tmp = tmp.trim();
      if (StringRef(tmp).starts_with("//")) {
        newss << "\n";
      } else {
        newss << tmp.str() << "\n";
      }
    }
    return newss.str();
  };

  // If no output file given, output results to stdout
  if (outputFilename.empty()) {
    if (!rewriteBuffer) {
      std::ifstream in(inputFilename);
      std::stringstream ss;
      ss << in.rdbuf();
      llvm::outs() << maybeRemoveComments(ss.str()) << "\n";
    } else {
      llvm::outs() << maybeRemoveComments(std::string(rewriteBuffer->begin(),
                                                      rewriteBuffer->end()))
                   << "\n";
    }
    return 0;
  }

  // If output name provided, write to that file
  {
    std::ofstream out(outputFilename);
    if (rewriteBuffer)
      out << maybeRemoveComments(
          std::string(rewriteBuffer->begin(), rewriteBuffer->end()));
    else {
      std::ifstream in(inputFilename);
      std::stringstream ss;
      ss << in.rdbuf();
      out << maybeRemoveComments(ss.str());
    }
  }
  return 0;
}