/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "KernelJIT.h"

using namespace llvm;
using namespace llvm::orc;

namespace cudaq {
KernelJIT::KernelJIT(std::unique_ptr<ExecutionSession> ES,
                     JITTargetMachineBuilder JTMB, DataLayout DL,
                     const std::vector<std::string> &extraLibraries,
                     std::unique_ptr<LLVMContext> ctx)
    : ES(std::move(ES)),
      ObjectLayer(*this->ES,
                  []() { return std::make_unique<SectionMemoryManager>(); }),
      CompileLayer(*this->ES, ObjectLayer,
                   std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
      DL(std::move(DL)), Ctx(std::move(ctx)),
      MainJD(this->ES->createBareJITDylib("<main>")) {
  MainJD.addGenerator(
      cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
          DL.getGlobalPrefix())));
  for (auto &library : extraLibraries) {
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        library.data(), DL.getGlobalPrefix())));
  }
}

KernelJIT::~KernelJIT() {
  // End the session.
  if (auto Err = ES->endSession()) {
    ES->reportError(std::move(Err));
  }
}

Expected<std::unique_ptr<KernelJIT>>
KernelJIT::Create(const std::vector<std::string> &extraLibraries) {
  auto EPC = SelfExecutorProcessControl::Create();
  if (!EPC)
    return EPC.takeError();

  auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));
  JITTargetMachineBuilder JTMB(
      ES->getExecutorProcessControl().getTargetTriple());

  auto DL = JTMB.getDefaultDataLayoutForTarget();
  if (!DL)
    return DL.takeError();

  return std::make_unique<KernelJIT>(std::move(ES), std::move(JTMB),
                                     std::move(*DL), extraLibraries);
}

Error KernelJIT::addModule(std::unique_ptr<llvm::Module> M,
                           std::vector<std::string> extra_paths) {
  auto rt = MainJD.getDefaultResourceTracker();
  return CompileLayer.add(rt, ThreadSafeModule(std::move(M), Ctx));
}

Expected<JITEvaluatedSymbol> KernelJIT::lookup(StringRef Name) {
  return ES->lookup({&MainJD}, Name.str());
}
} // namespace cudaq
