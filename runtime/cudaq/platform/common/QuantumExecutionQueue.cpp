/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/runtime/logger/logger.h"
#include <sstream>
#include <stdexcept>

namespace cudaq {

QuantumExecutionQueue::QuantumExecutionQueue() : lock() {
  thread = std::thread(&QuantumExecutionQueue::handler, this);
  detail::registerExecutionQueue(*this);
}

QuantumExecutionQueue::~QuantumExecutionQueue() {
  shutdown();
  // Unconditional, unlike `shutdown()`: a queue already stopped by
  // `shutdownExecutionQueues()` must still leave the registry when destroyed.
  detail::unregisterExecutionQueue(*this);
}

void QuantumExecutionQueue::shutdown() {
  // call_once guard (rather than a `quit` check): a second caller must block
  // until the join below completes, not return while it is still in progress.
  std::call_once(shutdownFlag, [this] {
    {
      std::unique_lock<std::mutex> l(lock);
      quit = true;
      cv.notify_all();
    }
    // Captured before the join, which resets the id to "not a thread".
    std::ostringstream threadId;
    threadId << getExecutionThreadId();
    if (thread.joinable()) {
      thread.join();
    }

    // Destroy the discarded tasks now
    // e.g., Python tasks may capture MLIR modules whose erasure needs a Python
    // MLIR context.
    std::queue<QuantumTask> discarded;
    {
      std::unique_lock<std::mutex> l(lock);
      queue.swap(discarded);
    }
    if (!discarded.empty())
      CUDAQ_WARN("Execution queue on thread {} shut down with {} task(s) still "
                 "queued; they were discarded.",
                 threadId.str(), discarded.size());
  });
}

void QuantumExecutionQueue::enqueue(QuantumTask &t) {
  std::unique_lock<std::mutex> l(lock);
  if (quit)
    throw std::runtime_error(
        "cannot schedule new asynchronous tasks after shutdown");
  queue.push(t);
  cv.notify_one();
  return;
}

std::thread::id QuantumExecutionQueue::getExecutionThreadId() const {
  return thread.get_id();
}

void QuantumExecutionQueue::handler(void) {
  std::unique_lock<std::mutex> l(lock);

  do {
    // Wait until we have data or a quit signal
    cv.wait(l, [this] { return (queue.size() || quit); });

    // after wait, we own the lock
    if (!quit && queue.size()) {

      auto op = std::move(queue.front());
      queue.pop();

      // unlock now that we're done messing with the queue
      l.unlock();

      op();
      l.lock();
    }
  } while (!quit);
}

} // namespace cudaq
