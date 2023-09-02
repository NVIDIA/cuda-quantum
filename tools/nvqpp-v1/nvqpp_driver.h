/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "nvqpp_args.h"
#include "nvqpp_diagnostics.h"
#include "clang/Driver/Driver.h"
#include <iostream>
namespace cudaq {
using CudaqArgs = llvm::opt::InputArgList;
class Driver {
  /// Arguments originated from command line.
  llvm::opt::InputArgList clOptions;
  llvm::opt::ArgStringList hostCompilerArgs;
  // ArgvStorageBase &cmdArgs;
  std::string driverPath;
  ErrorsDiagnostics diag;
  clang::driver::Driver drv;
  std::string cudaqOptExe;
  std::string cudaqTranslateExe;
  std::string cudaqLibPath;
  std::string cudaqTargetsPath;
  TargetPlatformArgs::Data targetPlatformExtraArgs;
  std::string targetConfig;
  // Storage of arg strings to have reliable const char*
  mutable std::list<std::string> synthesizedArgStrings;

public:
  Driver(ArgvStorageBase &cmdArgs);
  int execute();
  static int executeCC1Tool(ArgvStorageBase &cmdArgs);
  static std::string
  constructCudaqOptPipeline(const llvm::opt::InputArgList &clOptions);

private:
  /// Construct a constant string pointer whose
  /// lifetime will match that of the Driver.
  const char *makeArgStringRef(llvm::StringRef argStr);
  /// Parse the given list of strings into an InputArgList.
  // llvm::opt::InputArgList parseArgStrings(ArgvStorageBase &args);
  static std::pair<llvm::opt::InputArgList, llvm::opt::ArgStringList>
  preProcessCudaQArguments(ArgvStorageBase &cmdArgs);
  bool handleImmediateArgs();

  std::unique_ptr<clang::driver::Compilation> makeCompilation();
  std::optional<clang::driver::Driver::ReproLevel> getClangReproLevel(
      const std::unique_ptr<clang::driver::Compilation> &comp) const;
  void setInstallDir(ArgvStorageBase &argv);

  static int cc1Main(const CudaqArgs &cudaqArgs, ArgvT ccargs, ArgT tool,
                     void *mainAddr);

  static bool executeCompilerInvocation(clang::CompilerInstance *ci,
                                        const CudaqArgs &cudaqArgs);
};
} // namespace cudaq