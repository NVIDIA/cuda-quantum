/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "nvqpp_diagnostics.h"
#include "nvqpp_targets.h"
#include "clang/Driver/Driver.h"
#include <iostream>
namespace cudaq {
using CudaqArgs = llvm::opt::InputArgList;
class Driver {
  /// Arguments originated from command line.
  llvm::opt::InputArgList clOptions;
  llvm::opt::ArgStringList hostCompilerArgs;
  std::string driverPath;
  ErrorsDiagnostics diag;
  clang::driver::Driver clangDriver;
  std::string cudaqLibPath;
  std::string cudaqTargetsPath;
  TargetPlatformConfig targetPlatformExtraArgs;
  std::string targetName;
  // Storage of arg strings to have reliable const char*
  mutable std::list<std::string> synthesizedArgStrings;

public:
  Driver(ArgvStorageBase &cmdArgs);
  /// Driver execution
  int execute();
  /// Main entry to the CC1 frontend tool
  static int executeCC1Tool(ArgvStorageBase &cmdArgs);
  static std::string
  constructCudaqOptPipeline(const llvm::opt::InputArgList &clOptions);

private:
  /// Construct a constant string pointer whose
  /// lifetime will match that of the Driver.
  const char *makeArgStringRef(llvm::StringRef argStr);
  /// Parse/filter nvq++ args.
  // Strategy:
  // - Parse CLI args into an InputArgList based on nvq++ options (tblgen).
  // - Filter them out of the list of arguments passing to the clang driver (it
  // will reject unknown arguments).
  // - Once clang driver has built the compilation job list, re-inject arguments
  // that are needed for the frontend (-cc1) in its invocation. e.g., the quake
  // optimization options, etc.
  static std::pair<llvm::opt::InputArgList, llvm::opt::ArgStringList>
  preProcessCudaQArguments(ArgvStorageBase &cmdArgs);
  /// Handle simple immediate actions without the need to construct a
  /// 'Compilation' instance.
  // e.g., show help, list targets, etc.
  bool handleImmediateArgs();
  /// Construct/build the compilation pipeline (e.g., whether to do linking or
  /// not).
  // Passing the filter args list to the clang driver for it to build.
  std::unique_ptr<clang::driver::Compilation> makeCompilation();
  std::optional<clang::driver::Driver::ReproLevel> getClangReproLevel(
      const std::unique_ptr<clang::driver::Compilation> &comp) const;
  /// Find/locate directories of the cudaq installation
  void setInstallDir(ArgvStorageBase &argv);
  // nvq++ cc1 tool main
  static int cc1Main(const CudaqArgs &cudaqArgs, ArgvT ccargs, ArgT tool,
                     void *mainAddr);
  static bool executeCompilerInvocation(clang::CompilerInstance *ci,
                                        const CudaqArgs &cudaqArgs);
};
} // namespace cudaq