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
namespace cudaq {
class Driver {
  ArgvStorageBase &cmdArgs;
  ErrorsDiagnostics diag;
  clang::driver::Driver drv;
  ExecCompileFuncT cc1EntryPoint;
  std::string cudaqOptExe;
  std::string cudaqTranslateExe;
  std::string cudaqLibPath;
  std::string cudaqTargetsPath;
  TargetPlatformArgs::Data targetPlatformExtraArgs;
  std::string targetConfig;
  CudaqArgs cudaqArgs;
  std::string cudaqOptPipeline;

public:
  Driver(const std::string &path, ArgvStorageBase &cmdArgs,
         ExecCompileFuncT cc1);
  int execute();

private:
  void preProcessCudaQArguments(ArgvStorageBase &cmdArgs);
  std::string processOptPipeline(ArgvStorageBase &args, bool doLink);
  std::unique_ptr<clang::driver::Compilation> makeCompilation();
  std::optional<clang::driver::Driver::ReproLevel> getClangReproLevel(
      const std::unique_ptr<clang::driver::Compilation> &comp) const;
  void setInstallDir(ArgvStorageBase &argv);
};
} // namespace cudaq