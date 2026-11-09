/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "llvm/Support/ThreadPool.h"

namespace {
/// A thread pool that runs each task on the thread that submits it.
///
/// MLIR hands every function body in a nested pass pipeline to its context's
/// thread pool. Kernel modules are typically too small to benefit from
/// multithreading. Running the tasks inline keeps the same work on one thread
/// with no wakeups to pay for.
///
/// Leaving threading disabled through MLIRContext::disableMultithreading is not
/// an option. It switches the attribute and type uniquers out of their locking
/// mode, so a context shared between threads stops being safe (e.g. mqpu,
/// where several threads compile against one context). LLVM's own
/// SingleThreadExecutor will also not serve: it defers tasks into an
/// unsynchronized deque that wait() drains, so two threads compiling against
/// one context would race on it.
///
/// Here we hold no state at all to avoid these issues.
class InlineExecutor : public llvm::ThreadPoolInterface {
public:
  /// Nothing is ever outstanding, so waiting has nothing to do.
  void wait() override {}
  void wait(llvm::ThreadPoolTaskGroup &) override {}

  unsigned getMaxConcurrency() const override { return 1; }

private:
  void asyncEnqueue(llvm::unique_function<void()> task,
                    llvm::ThreadPoolTaskGroup *) override {
    task();
  }
};
} // namespace

llvm::ThreadPoolInterface &cudaq_internal::compiler::getInlineExecutor() {
  // Never destroyed: it must outlive every context that borrows it, including
  // ones deliberately leaked at a longer lifetime than this file's statics.
  static auto *executor = new InlineExecutor();
  return *executor;
}
