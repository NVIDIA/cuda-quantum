/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include <memory>

using namespace llvm;
using namespace llvm::orc;

namespace cudaq {

// The KernelJIT class wraps the LLVM JIT utility types to
// take as input a llvm::Module and enable one to extract
// a function pointer for the contained llvm::Functions.
class KernelJIT {
private:
  // The LLVM ExecutionSession representing the JIT program
  std::unique_ptr<ExecutionSession> ES;

  // LLVM helper for object linking
  RTDyldObjectLinkingLayer ObjectLayer;

  // LLVM helper for compiling Modules
  IRCompileLayer CompileLayer;

  // Representation of the target triple
  DataLayout DL;

  // Thread-safe LLVM Context
  ThreadSafeContext Ctx;

  // LLVM Helper representing JIT dynamic library
  JITDylib &MainJD;

public:
  // The constructor, not meant to be used publicly, see KernelJIT::Create()
  KernelJIT(std::unique_ptr<ExecutionSession> ES, JITTargetMachineBuilder JTMB,
            DataLayout DL, const std::vector<std::string> &extraLibraries,
            std::unique_ptr<LLVMContext> ctx = std::make_unique<LLVMContext>());

  // The destructor
  ~KernelJIT();

  // Static creation method for the KernelJIT
  static Expected<std::unique_ptr<KernelJIT>>
  Create(const std::vector<std::string> &extraLibraries);

  // Add an LLVM Module to be JIT compiled, optionally
  // provide extra linker paths to search.
  Error addModule(std::unique_ptr<llvm::Module> M,
                  std::vector<std::string> extra_paths = {});

  // Lookup and return a symbol JIT compiled from the Module
  // i.e. get a handle to a specific compiled function
  // and cast to a function pointer to invoke.
  Expected<JITEvaluatedSymbol> lookup(StringRef Name);
};

} // namespace cudaq
