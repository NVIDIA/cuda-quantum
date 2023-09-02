/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/Version.h"
#include "nvqpp_config.h"
#include "nvqpp_diagnostics.h"
#include "nvqpp_driver.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  try {
    // Initialize variables to call the driver
    llvm::InitLLVM x(argc, argv);
    ArgvStorage cmdArgs(argv, argv + argc);
    if (llvm::sys::Process::FixupStandardFileDescriptors()) {
      return 1;
    }

    llvm::InitializeAllTargets();

    llvm::BumpPtrAllocator pointerAllocator;
    llvm::StringSaver saver(pointerAllocator);
    auto firstArg = llvm::find_if(llvm::drop_begin(cmdArgs),
                                  [](auto a) { return a != nullptr; });
    if (firstArg != cmdArgs.end() &&
        std::string_view(cmdArgs[1]).starts_with("-cc1")) {
      return cudaq::Driver::executeCC1Tool(cmdArgs);
    }

    cudaq::Driver driver(cmdArgs);
    return driver.execute();
  } catch (std::exception &e) {
    llvm::errs() << "Error: " << e.what() << '\n';
    std::exit(1);
  }
}